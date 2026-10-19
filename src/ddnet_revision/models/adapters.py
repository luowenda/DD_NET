from __future__ import annotations

import math
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..losses import HighFocalFrequencyLoss, mse_mae_loss
from .base import StepOutput, TrainableNowcastAdapter


def _namespace(
    experiment: dict[str, Any],
    data: dict[str, Any],
    model: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
) -> SimpleNamespace:
    input_frames = int(experiment["input_frames"])
    output_frames = int(experiment["output_frames"])
    patch_size = int(model.get("patch_size", 1))
    return SimpleNamespace(
        in_shape=(input_frames, 1, data["height"], data["width"]),
        pre_seq_length=input_frames,
        aft_seq_length=output_frames,
        total_length=input_frames + output_frames,
        input_length=input_frames,
        img_width=data["width"],
        img_height=data["height"],
        img_channel=1,
        patch_size=patch_size,
        filter_size=int(model.get("filter_size", 5)),
        stride=1,
        layer_norm=bool(model.get("layer_norm", True)),
        reverse_scheduled_sampling=int(bool(model.get("reverse_scheduled_sampling", False))),
        r_sampling_step_1=int(model.get("r_sampling_step_1", 25000)),
        r_sampling_step_2=int(model.get("r_sampling_step_2", 50000)),
        r_exp_alpha=float(model.get("r_exp_alpha", 5000)),
        decouple_beta=float(model.get("decouple_beta", 0.1)),
        batch_size=int(training["batch_size"]),
        device=device,
        embed_dim=int(model.get("embed_dim", 128)),
        window_size=int(model.get("window_size", 4)),
        depths=model.get("depths", [2, 6, 6, 2]),
        num_heads=model.get("num_heads", [4, 8]),
    )


def _sampling_mask(
    batch_size: int,
    output_frames: int,
    height: int,
    width: int,
    channels: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    shape = (batch_size, output_frames - 1, height, width, channels)
    if ratio <= 0:
        return torch.zeros(shape, device=device)
    if ratio >= 1:
        return torch.ones(shape, device=device)
    sampled = (torch.rand((batch_size, output_frames - 1, 1, 1, 1), device=device) < ratio)
    return sampled.expand(shape).float()


def _reverse_sampling_mask(
    batch_size: int,
    input_frames: int,
    output_frames: int,
    height: int,
    width: int,
    channels: int,
    global_step: int,
    step_1: int,
    step_2: int,
    exp_alpha: float,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    if not training:
        context = torch.ones(
            (batch_size, input_frames - 1, height, width, channels), device=device
        )
        future = torch.zeros(
            (batch_size, output_frames - 1, height, width, channels), device=device
        )
        return torch.cat([context, future], dim=1)

    if global_step < step_1:
        context_ratio = 0.5
        future_ratio = 0.5
    elif global_step < step_2:
        context_ratio = 1.0 - 0.5 * math.exp(-(global_step - step_1) / exp_alpha)
        future_ratio = 0.5 - 0.5 * (global_step - step_1) / (step_2 - step_1)
    else:
        context_ratio = 1.0
        future_ratio = 0.0

    context = (
        torch.rand((batch_size, input_frames - 1, 1, 1, 1), device=device) < context_ratio
    ).expand(batch_size, input_frames - 1, height, width, channels).float()
    future = (
        torch.rand((batch_size, output_frames - 1, 1, 1, 1), device=device) < future_ratio
    ).expand(batch_size, output_frames - 1, height, width, channels).float()
    return torch.cat([context, future], dim=1)


def _reshape_patch(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return frames
    batch, length, height, width, channels = frames.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Frame size {(height, width)} is not divisible by patch size {patch_size}")
    frames = frames.reshape(
        batch,
        length,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    return frames.transpose(3, 4).reshape(
        batch,
        length,
        height // patch_size,
        width // patch_size,
        patch_size * patch_size * channels,
    )


def _reshape_patch_back(frames: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return frames
    batch, length, height, width, channels = frames.shape
    image_channels = channels // (patch_size * patch_size)
    frames = frames.reshape(
        batch,
        length,
        height,
        width,
        patch_size,
        patch_size,
        image_channels,
    )
    return frames.transpose(3, 4).reshape(
        batch,
        length,
        height * patch_size,
        width * patch_size,
        image_channels,
    )


class RecurrentSequenceAdapter(TrainableNowcastAdapter):
    def __init__(
        self,
        model: nn.Module,
        normalization_scale: float,
        output_frames: int,
        *,
        supports_return_loss: bool = False,
        forecast_loss_name: str | None = None,
        hffl_weight: float = 0.001,
    ) -> None:
        super().__init__(normalization_scale, output_frames)
        self.model = model
        self.patch_size = int(model.configs.patch_size)
        self.reverse_scheduled_sampling = bool(model.configs.reverse_scheduled_sampling)
        self.supports_return_loss = supports_return_loss
        self.forecast_loss_name = forecast_loss_name
        self.hffl = HighFocalFrequencyLoss(loss_weight=hffl_weight)

    def _forecast_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.forecast_loss_name == "mse_mae":
            return mse_mae_loss(prediction, target)
        loss = F.mse_loss(prediction, target)
        if self.forecast_loss_name == "mse_hffl":
            loss = loss + self.hffl(prediction, target)
        return loss

    def _forward(
        self,
        input_dbz: torch.Tensor,
        target_dbz: torch.Tensor | None,
        ratio: float,
        global_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_value = self.normalize(input_dbz)
        if target_dbz is None:
            target_value = torch.zeros(
                input_value.shape[0],
                self.output_frames,
                *input_value.shape[2:],
                device=input_value.device,
                dtype=input_value.dtype,
            )
        else:
            target_value = self.normalize(target_dbz)
        frames = torch.cat([input_value, target_value], dim=1).permute(0, 1, 3, 4, 2)
        frames = _reshape_patch(frames, self.patch_size)
        if self.reverse_scheduled_sampling:
            cfg = self.model.configs
            mask = _reverse_sampling_mask(
                frames.shape[0],
                int(cfg.pre_seq_length),
                self.output_frames,
                frames.shape[2],
                frames.shape[3],
                frames.shape[4],
                global_step,
                int(cfg.r_sampling_step_1),
                int(cfg.r_sampling_step_2),
                float(cfg.r_exp_alpha),
                frames.device,
                target_dbz is not None,
            )
        else:
            mask = _sampling_mask(
                frames.shape[0],
                self.output_frames,
                frames.shape[2],
                frames.shape[3],
                frames.shape[4],
                ratio,
                frames.device,
            )
        if self.supports_return_loss:
            generated, loss = self.model(
                frames, mask, return_loss=target_dbz is not None
            )
        else:
            generated, loss = self.model(frames, mask)
        generated = _reshape_patch_back(generated, self.patch_size)
        prediction = generated[:, -self.output_frames :].permute(0, 1, 4, 2, 3)
        return prediction, loss

    def training_step(
        self,
        input_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        teacher_forcing_ratio: float,
        global_step: int = 0,
    ) -> StepOutput:
        prediction, loss = self._forward(
            input_dbz, target_dbz, teacher_forcing_ratio, global_step
        )
        if self.forecast_loss_name is not None:
            loss = self._forecast_loss(prediction, self.normalize(target_dbz))
        elif loss is None:
            loss = F.mse_loss(prediction, self.normalize(target_dbz))
        return StepOutput(loss=loss, prediction_dbz=self.denormalize(prediction))

    def predict(self, input_dbz: torch.Tensor) -> torch.Tensor:
        prediction, _ = self._forward(input_dbz, None, 0.0, 0)
        return self.denormalize(prediction)


class SwinLSTMAdapter(TrainableNowcastAdapter):
    def __init__(self, model: nn.Module, normalization_scale: float, output_frames: int) -> None:
        super().__init__(normalization_scale, output_frames)
        self.model = model

    def _forward(
        self, input_dbz: torch.Tensor, target_dbz: torch.Tensor | None, return_loss: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_value = self.normalize(input_dbz)
        target_value = (
            torch.zeros_like(input_value[:, : self.output_frames])
            if target_dbz is None
            else self.normalize(target_dbz)
        )
        frames = torch.cat([input_value, target_value], dim=1).permute(0, 1, 3, 4, 2)
        generated, loss = self.model(frames, return_loss=return_loss)
        prediction = generated[:, -self.output_frames :].permute(0, 1, 4, 2, 3)
        return prediction, loss

    def training_step(
        self,
        input_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        teacher_forcing_ratio: float,
        global_step: int = 0,
    ) -> StepOutput:
        del teacher_forcing_ratio, global_step
        prediction, loss = self._forward(input_dbz, target_dbz, True)
        assert loss is not None
        return StepOutput(loss=loss, prediction_dbz=self.denormalize(prediction))

    def predict(self, input_dbz: torch.Tensor) -> torch.Tensor:
        prediction, _ = self._forward(input_dbz, None, False)
        return self.denormalize(prediction)


class DirectSequenceAdapter(TrainableNowcastAdapter):
    def __init__(
        self,
        model: nn.Module,
        normalization_scale: float,
        output_frames: int,
        loss_name: str,
        hffl_weight: float = 0.001,
    ) -> None:
        super().__init__(normalization_scale, output_frames)
        self.model = model
        self.loss_name = loss_name
        self.hffl = HighFocalFrequencyLoss(loss_weight=hffl_weight)

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse_mae":
            return mse_mae_loss(prediction, target)
        loss = F.mse_loss(prediction, target)
        if self.loss_name == "mse_hffl":
            loss = loss + self.hffl(prediction, target)
        return loss

    def training_step(
        self,
        input_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        teacher_forcing_ratio: float,
        global_step: int = 0,
    ) -> StepOutput:
        del teacher_forcing_ratio, global_step
        prediction = self.model(self.normalize(input_dbz))
        loss = self._loss(prediction, self.normalize(target_dbz))
        return StepOutput(loss=loss, prediction_dbz=self.denormalize(prediction))

    def predict(self, input_dbz: torch.Tensor) -> torch.Tensor:
        return self.denormalize(self.model(self.normalize(input_dbz)))


class TrajGRUAdapter(DirectSequenceAdapter):
    def training_step(
        self,
        input_dbz: torch.Tensor,
        target_dbz: torch.Tensor,
        teacher_forcing_ratio: float,
        global_step: int = 0,
    ) -> StepOutput:
        del teacher_forcing_ratio, global_step
        prediction = self.model(self.normalize(input_dbz).permute(1, 0, 2, 3, 4))
        prediction = prediction.permute(1, 0, 2, 3, 4)
        target = self.normalize(target_dbz)
        return StepOutput(
            loss=F.mse_loss(prediction, target), prediction_dbz=self.denormalize(prediction)
        )

    def predict(self, input_dbz: torch.Tensor) -> torch.Tensor:
        prediction = self.model(self.normalize(input_dbz).permute(1, 0, 2, 3, 4))
        return self.denormalize(prediction.permute(1, 0, 2, 3, 4))


def _build_trajgru(model_cfg: dict[str, Any], batch_size: int, output_frames: int) -> nn.Module:
    from .external import load_external_module

    package = load_external_module("trajgru")
    EF, Encoder, Forecaster, TrajGRU = package.EF, package.Encoder, package.Forecaster, package.TrajGRU

    hidden = [int(value) for value in model_cfg["hidden_dims"]]
    links = [int(value) for value in model_cfg["links"]]
    encoder_subnets = [
        OrderedDict({"conv1_leaky_1": [1, 8, 3, 2, 1]}),
        OrderedDict({"conv2_leaky_1": [hidden[0], 192, 3, 2, 1]}),
        OrderedDict({"conv3_leaky_1": [hidden[1], 192, 3, 2, 1]}),
    ]
    encoder_rnns = [
        TrajGRU(8, hidden[0], (batch_size, 32, 32), L=links[0]),
        TrajGRU(192, hidden[1], (batch_size, 16, 16), L=links[1]),
        TrajGRU(192, hidden[2], (batch_size, 8, 8), L=links[2], h2h_kernel=(3, 3)),
    ]
    forecaster_subnets = [
        OrderedDict({"deconv1_leaky_1": [hidden[2], 192, 4, 2, 1]}),
        OrderedDict({"deconv2_leaky_1": [hidden[1], 64, 4, 2, 1]}),
        OrderedDict(
            {
                "deconv3_leaky_1": [hidden[0], 8, 4, 2, 1],
                "conv3_leaky_2": [8, 8, 3, 1, 1],
                "conv3_3": [8, 1, 1, 1, 0],
            }
        ),
    ]
    forecaster_rnns = [
        TrajGRU(192, hidden[2], (batch_size, 8, 8), L=links[0], h2h_kernel=(3, 3)),
        TrajGRU(192, hidden[1], (batch_size, 16, 16), L=links[1]),
        TrajGRU(64, hidden[0], (batch_size, 32, 32), L=links[2]),
    ]
    return EF(
        Encoder(encoder_subnets, encoder_rnns),
        Forecaster(forecaster_subnets, forecaster_rnns, output_frames=output_frames),
    )


def build_trainable_adapter(
    config: dict[str, Any], device: torch.device
) -> TrainableNowcastAdapter:
    from ..config import validate_config

    validate_config(config)
    experiment = config["experiment"]
    data = config["data"]
    model_cfg = config["model"]
    training = config["training"]
    name = model_cfg["name"]
    scale = float(experiment["normalization_scale"])
    output_frames = int(experiment["output_frames"])
    namespace = _namespace(experiment, data, model_cfg, training, device)

    if name in {'convlstm_ddnet_recipe_e100'}:
        from .vendor.convlstm import ConvLSTM_Model

        hidden = [int(value) for value in model_cfg["hidden_dims"]]
        model = ConvLSTM_Model(len(hidden), hidden, namespace)
        if name == "convlstm_ddnet_recipe_e100":
            return RecurrentSequenceAdapter(
                model,
                scale,
                output_frames,
                forecast_loss_name=str(training["loss"]),
                hffl_weight=float(training.get("hffl_weight", 0.001)),
            )
        return RecurrentSequenceAdapter(model, scale, output_frames)
    if name in {'predrnn_openstl_e100'}:
        from .vendor.predrnn_openstl import PredRNN_Model

        hidden = [int(value) for value in model_cfg["hidden_dims"]]
        return RecurrentSequenceAdapter(
            PredRNN_Model(len(hidden), hidden, namespace),
            scale,
            output_frames,
            supports_return_loss=True,
        )
    if name in {'predrnnv2_openstl_e100'}:
        from .vendor.predrnnv2_openstl import PredRNNv2_Model

        hidden = [int(value) for value in model_cfg["hidden_dims"]]
        return RecurrentSequenceAdapter(
            PredRNNv2_Model(len(hidden), hidden, namespace),
            scale,
            output_frames,
            supports_return_loss=True,
        )
    if name in {'mim_e100'}:
        from .vendor.mim import MIM_Model

        hidden = [int(value) for value in model_cfg["hidden_dims"]]
        return RecurrentSequenceAdapter(
            MIM_Model(len(hidden), hidden, namespace), scale, output_frames
        )
    if name in {'motionrnn_e100'}:
        from .external import load_external_module

        RNN = load_external_module("motionrnn").RNN

        hidden = [int(value) for value in model_cfg["hidden_dims"]]
        return RecurrentSequenceAdapter(RNN(len(hidden), hidden, namespace), scale, output_frames)
    if name in {'swinlstm_e100'}:
        from .vendor.swinlstm import SwinLSTM_D_Model

        model = SwinLSTM_D_Model(
            model_cfg["depths_downsample"],
            model_cfg["depths_upsample"],
            model_cfg["num_heads"],
            namespace,
        )
        return SwinLSTMAdapter(model, scale, output_frames)
    if name in {'wast_historical_recipe'}:
        from .vendor.wast import WaST_level1

        model = WaST_level1(
            in_shape=(experiment["input_frames"], 1, data["height"], data["width"]),
            encoder_dim=model_cfg["encoder_dim"],
            block_list=model_cfg["block_list"],
            drop_path_rate=model_cfg["drop_path_rate"],
            mlp_ratio=model_cfg["mlp_ratio"],
            wavelet_subband_grouping=model_cfg.get(
                "wavelet_subband_grouping", "openstl"
            ),
        )
        return DirectSequenceAdapter(
            model, scale, output_frames, training["loss"], training["hffl_weight"]
        )
    if name in {'ddnet_ablate_dcn_only_e100'}:
        from .vendor.ddnet import DDNetComponentAblation

        model = DDNetComponentAblation(
            in_shape=(experiment["input_frames"], 1, data["height"], data["width"]),
            encoder_dim=model_cfg["encoder_dim"],
            block_list=model_cfg["block_list"],
            use_hypergraph_encoder=model_cfg["use_hypergraph_encoder"],
            use_dcn_core=model_cfg["use_dcn_core"],
            drop_path_rate=model_cfg["drop_path_rate"],
            mlp_ratio=model_cfg["mlp_ratio"],
            hypergraph_scales=model_cfg["hypergraph_scales"],
            hypergraph_scale_gates=model_cfg.get("hypergraph_scale_gates"),
            hyperedge_weighting=model_cfg.get("hyperedge_weighting", "adaptive"),
            wavelet_subband_grouping=model_cfg["wavelet_subband_grouping"],
            residual_layer_scale_mode=model_cfg["residual_layer_scale_mode"],
        )
        return DirectSequenceAdapter(
            model,
            scale,
            output_frames,
            training["loss"],
            training.get("hffl_weight", 0.001),
        )
    if name in {
        'ddnet_incidence_attention_large_e100', 'ddnet_corrected_uniform_large_e100',
        'ddnet_attention_v1_dual_linear_e100', 'ddnet_ablate_v1_small_gate_e100',
        'ddnet_ablate_v1_no_modulation_e100', 'ddnet_ablate_v1_no_temporal_mixer_e100',
        'ddnet_ablate_v1_no_wavelet_e100', 'ddnet_ablate_v1_mse_only_e100',
        'ddnet_ablate_v1_hypergraph_only_e100',
    }:
        from .vendor.ddnet import DD_Net

        model_type = DD_Net
        component_options = {}
        if name == "ddnet_ablate_v1_hypergraph_only_e100":
            from .vendor.ddnet.attention_component import AttentionV1HypergraphOnly

            model_type = AttentionV1HypergraphOnly
            component_options["use_dcn_core"] = model_cfg["use_dcn_core"]
        if name == "ddnet_ablate_v1_no_wavelet_e100":
            from .vendor.ddnet.attention_ablation import AttentionV1WaveletReplacement

            model_type = AttentionV1WaveletReplacement
        model = model_type(
            in_shape=(experiment["input_frames"], 1, data["height"], data["width"]),
            encoder_dim=model_cfg["encoder_dim"],
            block_list=model_cfg["block_list"],
            attention_mode=model_cfg["attention_mode"],
            batching_mode=model_cfg["batching_mode"],
            drop_path_rate=model_cfg["drop_path_rate"],
            mlp_ratio=model_cfg["mlp_ratio"],
            wavelet_subband_grouping=model_cfg["wavelet_subband_grouping"],
            residual_layer_scale_mode=model_cfg["residual_layer_scale_mode"],
            hypergraph_scales=model_cfg.get(
                "hypergraph_scales", ["small", "large"]
            ),
            hypergraph_scale_gates=model_cfg.get("hypergraph_scale_gates"),
            hyperedge_weighting=model_cfg.get(
                "hyperedge_weighting", "adaptive"
            ),
            incidence_attention_heads=model_cfg.get(
                "incidence_attention_heads", 2
            ),
            incidence_attention_channels=model_cfg.get(
                "incidence_attention_channels", 8
            ),
            incidence_attention_logit_scale=model_cfg.get(
                "incidence_attention_logit_scale", 0.1
            ),
            hypergraph_scale_fusion=model_cfg.get("hypergraph_scale_fusion", "linear"),
            scale_gate_hidden_channels=model_cfg.get("scale_gate_hidden_channels", 16),
            hypergraph_input_stem_channels=model_cfg.get("hypergraph_input_stem_channels", 0),
            hypergraph_input_stem_kernel_size=model_cfg.get("hypergraph_input_stem_kernel_size", 3),
            use_modulation_mask=model_cfg.get("use_modulation_mask", True),
            use_temporal_mixer=model_cfg.get("use_temporal_mixer", True),
            modulation_mask_gate=model_cfg.get("modulation_mask_gate"),
            temporal_mixer_gate=model_cfg.get("temporal_mixer_gate"),
            use_wavelet=model_cfg.get("use_wavelet", True),
            **component_options,
        )
        return DirectSequenceAdapter(
            model,
            scale,
            output_frames,
            training["loss"],
            training.get("hffl_weight", 0.001),
        )
    if name in {'trajgru_e100'}:
        model = _build_trajgru(model_cfg, int(training["batch_size"]), output_frames)
        return TrajGRUAdapter(model, scale, output_frames, "mse")
    raise KeyError(f"Unsupported released model: {name}")
