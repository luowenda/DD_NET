from __future__ import annotations

from itertools import accumulate

import torch
from einops import rearrange
from timm.layers import DropPath
from torch import nn

from ..wast.model import (
    TF_AwareBlocks as WaSTBlocks,
    Wavelet_3D_Embedding,
    Wavelet_3D_Reconstruction,
)
from .model import (
    CorrectedAdaptiveHypergraph,
    LocalHypergraph_BroadcastBatching,
    NodeAttention,
    StrictUniformCorrectedHypergraph,
    StrictUniformNodeAttention,
    TF_AwareBlocks as DDNetDCNBlocks,
)


class DDNetComponentAblation(nn.Module):
    """Compose the WaST backbone with exactly one or both DD-Net components."""

    def __init__(
        self,
        in_shape,
        encoder_dim,
        block_list,
        *,
        use_hypergraph_encoder,
        use_dcn_core,
        drop_path_rate=0.1,
        mlp_ratio=4.0,
        hypergraph_scales=("small", "large"),
        hypergraph_scale_gates=None,
        hyperedge_weighting="adaptive",
        wavelet_subband_grouping="openstl",
        residual_layer_scale_mode="shared_openstl",
    ):
        super().__init__()
        frame, in_dim, height, width = in_shape
        self.block_list = list(block_list)
        self.use_hypergraph_encoder = bool(use_hypergraph_encoder)
        self.use_dcn_core = bool(use_dcn_core)

        if self.use_hypergraph_encoder:
            if hyperedge_weighting == "adaptive":
                attention_module = NodeAttention
            elif hyperedge_weighting == "strict_uniform":
                attention_module = StrictUniformNodeAttention
            elif hyperedge_weighting == "corrected_adaptive":
                attention_module = CorrectedAdaptiveHypergraph
            elif hyperedge_weighting == "corrected_strict_uniform":
                attention_module = StrictUniformCorrectedHypergraph
            else:
                raise ValueError(
                    "component ablations support 'adaptive', 'strict_uniform', "
                    "'corrected_adaptive', or 'corrected_strict_uniform' "
                    "hyperedge weighting"
                )
            self.conv_in = LocalHypergraph_BroadcastBatching(
                in_channels=in_dim,
                out_channels=encoder_dim,
                spatial_size=(height, width),
                temporal_size=frame,
                attention_module=attention_module,
                scales=hypergraph_scales,
                scale_gates=hypergraph_scale_gates,
            )
        else:
            self.conv_in = nn.Sequential(
                nn.Conv3d(
                    in_dim,
                    encoder_dim,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                ),
                nn.BatchNorm3d(encoder_dim),
                nn.GELU(),
            )

        drop_paths = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(self.block_list))
        ]
        indexes = list(accumulate(self.block_list))
        drop_paths = [
            drop_paths[start:end]
            for start, end in zip([0] + indexes, indexes)
        ]

        block_type = DDNetDCNBlocks if self.use_dcn_core else WaSTBlocks
        common_options = {}
        if self.use_dcn_core:
            common_options["residual_layer_scale_mode"] = residual_layer_scale_mode

        self.translator1 = block_type(
            dim=encoder_dim * frame,
            num_blocks=self.block_list[0],
            drop_path=drop_paths[0],
            mlp_ratio=mlp_ratio,
            large_kernel=51,
            small_kernel=5,
            **common_options,
        )
        self.wavelet_embed1 = Wavelet_3D_Embedding(
            in_dim=encoder_dim,
            out_dim=encoder_dim * 2,
            emb_dim=in_dim,
            subband_grouping=wavelet_subband_grouping,
        )
        self.bottleneck_translator = block_type(
            dim=encoder_dim * 2 * frame,
            num_blocks=self.block_list[1],
            drop_path=drop_paths[1],
            use_bottleneck=True,
            mlp_ratio=mlp_ratio,
            large_kernel=21,
            small_kernel=5,
            **common_options,
        )
        self.wavelet_recon1 = Wavelet_3D_Reconstruction(
            in_dim=encoder_dim * 2,
            out_dim=encoder_dim,
            hi_dim=encoder_dim,
        )
        self.translator2 = block_type(
            dim=encoder_dim * frame,
            num_blocks=self.block_list[2],
            drop_path=drop_paths[2],
            use_hid=True,
            mlp_ratio=mlp_ratio,
            large_kernel=51,
            small_kernel=5,
            **common_options,
        )
        self.conv_out = nn.Sequential(
            nn.BatchNorm3d(encoder_dim),
            nn.GELU(),
            nn.Conv3d(
                encoder_dim,
                in_dim,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
            ),
        )

    def update_drop_path(self, drop_path_rate):
        drop_paths = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(self.block_list))
        ]
        indexes = list(accumulate(self.block_list))
        grouped = [
            drop_paths[start:end]
            for start, end in zip([0] + indexes, indexes)
        ]
        block_groups = (
            self.translator1.blocks,
            self.bottleneck_translator.blocks,
            self.translator2.blocks,
        )
        for blocks, rates in zip(block_groups, grouped):
            for block, rate in zip(blocks, rates):
                if hasattr(block.drop_path, "drop_prob"):
                    block.drop_path.drop_prob = rate
                elif rate != 0.0:
                    block.drop_path = DropPath(rate)

    def forward(self, x):
        x = rearrange(x, "b t c h w -> b c t h w")
        original = x
        x = self.conv_in(x)
        x, temporal_skip = self.translator1(x)
        x, wavelet_skip = self.wavelet_embed1(x, x_emb=original)
        x = self.bottleneck_translator(x)
        x = self.wavelet_recon1(x, wavelet_skip)
        x = self.translator2(x, temporal_skip)
        x = self.conv_out(x)
        return rearrange(x, "b c t h w -> b t c h w")
