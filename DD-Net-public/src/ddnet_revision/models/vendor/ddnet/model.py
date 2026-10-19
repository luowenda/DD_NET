import math

import torch
import torch.nn as nn
import pywt
import os
from einops import rearrange
from functools import partial
from itertools import accumulate
from timm.layers import DropPath, activations
from timm.models._efficientnet_blocks import SqueezeExcite, InvertedResidual
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import softmax as segment_softmax

from torchvision.ops import DeformConv2d

from .scale_fusion import NodeScaleGate



# 获取主要版本号部分（忽略开发版本标识）
version_parts = torch.__version__.split('+')[0].split('.')
# 只处理前三个部分（主版本、次版本、修订号）
major, minor, patch = version_parts[:3]
# 尝试将每个部分转换为整数，如果失败则设为0
try:
    major = int(major)
except ValueError:
    major = 0
try:
    minor = int(minor)
except ValueError:
    minor = 0
try:
    patch = int(patch)
except ValueError:
    patch = 0

IS_HIGH_VERSION = (major, minor, patch) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft





class HighFocalFrequencyLoss(nn.Module):
    """ Example:
        fake = torch.randn(4, 3, 128, 64)
        real = torch.randn(4, 3, 128, 64)
        hffl = HighFocalFrequencyLoss()

        loss = hffl(fake, real)
        print(loss)
    """

    def __init__(self, loss_weight=0.001, level=1, tau=0.1, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=True, batch_matrix=False):
        super(HighFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.level = level
        self.tau = tau
        self.DWT = WaveletTransform2D().cuda()

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def build_freq_mask(self, shape):
        H, W = shape[-2:]
        radius = self.tau * max(H, W)
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))

        mask = torch.ones_like(X, dtype=torch.float32).cuda()

        centers = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]

        for center in centers:
            distance = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
            mask[distance <= radius] = 0
        return mask

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        mask = self.build_freq_mask(weight_matrix.shape)
        loss = weight_matrix * freq_distance * mask
        return torch.mean(loss)

    def frequency_loss(self, pred, target, matrix=None):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix)

    def forward(self, pred, target, matrix=None, **kwargs):
        pred = rearrange(pred, 'b t c h w -> (b t) c h w') if kwargs["reshape"] is True else pred
        target = rearrange(target, 'b t c h w -> (b t) c h w') if kwargs["reshape"] is True else target

        loss = 0
        for level in range(self.level):
            pred, _, _, _ = self.DWT(pred)
            target, _, _, _ = self.DWT(target)
            loss += self.frequency_loss(pred, target, matrix)
        return loss * self.loss_weight



class NodeAttention(nn.Module):
    """
    方案A: 您的原始版本，直接对节点特征计算注意力。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hyper_conv = HypergraphConv(in_channels, out_channels)
        self.attn = nn.Linear(in_channels, 1)
        
    def forward(self, x, hyperedge_index, edges_per_graph=None):
        edge_weight = self.attn(x).sigmoid()
        return self.hyper_conv(x, hyperedge_index, edge_weight)


class UniformHypergraph(nn.Module):
    """Hypergraph convolution with the library-default unit hyperedge weights."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hyper_conv = HypergraphConv(in_channels, out_channels)

    def forward(self, x, hyperedge_index, edges_per_graph=None):
        return self.hyper_conv(x, hyperedge_index)


class StrictUniformNodeAttention(NodeAttention):
    """Retain adaptive-attention parameters while forcing unit edge weights."""

    def forward(self, x, hyperedge_index, edges_per_graph=None):
        adaptive_weight = self.attn(x).sigmoid()
        edge_weight = adaptive_weight * 0.0 + 1.0
        return self.hyper_conv(x, hyperedge_index, edge_weight)


class CorrectedAdaptiveHypergraph(nn.Module):
    """Content-aware hyperedge weighting with explicit weighted propagation."""

    def __init__(self, in_channels, out_channels, weight_range=0.5, eps=1e-8):
        super().__init__()
        if not 0.0 < weight_range < 1.0:
            raise ValueError("weight_range must be between zero and one")
        self.hyper_conv = HypergraphConv(in_channels, out_channels)
        hidden_channels = max(8, in_channels * 2)
        self.edge_gate = nn.Sequential(
            nn.Linear(in_channels * 4, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.weight_range = float(weight_range)
        self.eps = float(eps)

        # Start as the unit-weight operator. The final layer learns departures
        # from the uniform reference without changing the initial feature scale.
        nn.init.zeros_(self.edge_gate[-1].weight)
        nn.init.zeros_(self.edge_gate[-1].bias)

    @staticmethod
    def _edge_count(hyperedge_index):
        if hyperedge_index.numel() == 0:
            return 0
        return int(hyperedge_index[1].max()) + 1

    def edge_descriptors(self, x, hyperedge_index):
        node_index, edge_index = hyperedge_index
        num_edges = self._edge_count(hyperedge_index)
        if num_edges != x.shape[0]:
            raise ValueError(
                "Corrected local hypergraphs require one center hyperedge per node"
            )

        members = x[node_index]
        counts = x.new_zeros((num_edges, 1))
        counts.index_add_(0, edge_index, x.new_ones((edge_index.numel(), 1)))

        edge_sum = x.new_zeros((num_edges, x.shape[-1]))
        edge_sum.index_add_(0, edge_index, members)
        edge_mean = edge_sum / counts.clamp_min(1.0)

        edge_square_sum = x.new_zeros((num_edges, x.shape[-1]))
        edge_square_sum.index_add_(0, edge_index, members.square())
        edge_variance = (
            edge_square_sum / counts.clamp_min(1.0) - edge_mean.square()
        ).clamp_min(0.0)
        edge_std = torch.sqrt(edge_variance + self.eps)

        expanded_edge_index = edge_index[:, None].expand(-1, x.shape[-1])
        edge_max = x.new_full((num_edges, x.shape[-1]), -torch.inf)
        edge_max.scatter_reduce_(
            0, expanded_edge_index, members, reduce="amax", include_self=True
        )

        # create_hyperedges assigns edge e to the neighborhood centered at node e.
        center = x[torch.arange(num_edges, device=x.device)]
        return torch.cat((center, edge_mean, edge_std, edge_max), dim=-1)

    def compute_edge_weights(self, x, hyperedge_index, edges_per_graph=None):
        descriptors = self.edge_descriptors(x, hyperedge_index)
        delta = self.edge_gate(descriptors).squeeze(-1)
        weights = 1.0 + self.weight_range * torch.tanh(delta)

        if edges_per_graph is None:
            return weights / weights.mean().clamp_min(self.eps)
        if edges_per_graph <= 0 or weights.numel() % edges_per_graph:
            raise ValueError("Invalid edges_per_graph for per-sample normalization")
        grouped = weights.view(-1, edges_per_graph)
        grouped = grouped / grouped.mean(dim=1, keepdim=True).clamp_min(self.eps)
        return grouped.reshape(-1)

    def weighted_convolution(self, x, hyperedge_index, edge_weight):
        node_index, edge_index = hyperedge_index
        num_edges = self._edge_count(hyperedge_index)
        if edge_weight.shape != (num_edges,):
            raise ValueError("edge_weight must contain one scalar per hyperedge")

        transformed = self.hyper_conv.lin(x)
        edge_counts = transformed.new_zeros((num_edges, 1))
        edge_counts.index_add_(
            0, edge_index, transformed.new_ones((edge_index.numel(), 1))
        )
        edge_messages = transformed.new_zeros((num_edges, transformed.shape[-1]))
        edge_messages.index_add_(0, edge_index, transformed[node_index])
        edge_messages = edge_messages / edge_counts.clamp_min(1.0)
        edge_messages = edge_messages * edge_weight[:, None]

        weighted_degree = transformed.new_zeros((x.shape[0],))
        weighted_degree.index_add_(0, node_index, edge_weight[edge_index])
        output = transformed.new_zeros((x.shape[0], transformed.shape[-1]))
        output.index_add_(0, node_index, edge_messages[edge_index])
        output = output / weighted_degree.clamp_min(self.eps)[:, None]
        if self.hyper_conv.bias is not None:
            output = output + self.hyper_conv.bias
        return output

    def forward(self, x, hyperedge_index, edges_per_graph=None):
        edge_weight = self.compute_edge_weights(
            x, hyperedge_index, edges_per_graph=edges_per_graph
        )
        return self.weighted_convolution(x, hyperedge_index, edge_weight)


class StrictUniformCorrectedHypergraph(CorrectedAdaptiveHypergraph):
    """Keep the corrected gate architecture while forcing unit weights."""

    def compute_edge_weights(self, x, hyperedge_index, edges_per_graph=None):
        adaptive = super().compute_edge_weights(
            x, hyperedge_index, edges_per_graph=edges_per_graph
        )
        return adaptive * 0.0 + 1.0


class IncidenceAttentionHypergraph(nn.Module):
    """Directly replace uniform two-stage aggregation with incidence attention.

    Attention is normalized first over the nodes in each hyperedge and then over
    the hyperedges incident to each node. The value path stays in the input
    feature space, so uniform attention exactly recovers the corrected uniform
    operator before the shared HypergraphConv linear projection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=2,
        attention_channels=8,
        initial_logit_scale=0.1,
    ):
        super().__init__()
        if num_heads <= 0 or attention_channels <= 0:
            raise ValueError("attention head counts and dimensions must be positive")
        if initial_logit_scale <= 0.0:
            raise ValueError("initial_logit_scale must be positive")

        self.hyper_conv = HypergraphConv(in_channels, out_channels)
        self.num_heads = int(num_heads)
        self.attention_channels = int(attention_channels)
        projection_channels = self.num_heads * self.attention_channels

        def projector():
            hidden_channels = max(8, in_channels * 4)
            return nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, projection_channels),
            )

        self.node_to_edge_node = projector()
        self.node_to_edge_edge = projector()
        self.edge_to_node_node = projector()
        self.edge_to_node_edge = projector()

        raw_scale = torch.log(torch.expm1(torch.tensor(float(initial_logit_scale))))
        self.node_to_edge_raw_scale = nn.Parameter(raw_scale.clone())
        self.edge_to_node_raw_scale = nn.Parameter(raw_scale.clone())

    @staticmethod
    def _edge_count(hyperedge_index):
        if hyperedge_index.numel() == 0:
            return 0
        return int(hyperedge_index[1].max()) + 1

    @staticmethod
    def _mean_by_index(values, index, count):
        result = values.new_zeros((count, values.shape[-1]))
        result.index_add_(0, index, values)
        denominator = values.new_zeros((count, 1))
        denominator.index_add_(
            0, index, values.new_ones((index.numel(), 1))
        )
        return result / denominator.clamp_min(1.0)

    def _project(self, module, values):
        return module(values).view(
            values.shape[0], self.num_heads, self.attention_channels
        )

    def compute_attention(self, x, hyperedge_index):
        node_index, edge_index = hyperedge_index
        num_edges = self._edge_count(hyperedge_index)
        if num_edges != x.shape[0]:
            raise ValueError(
                "Incidence attention requires one center hyperedge per local node"
            )

        edge_context = self._mean_by_index(x[node_index], edge_index, num_edges)
        node_key = self._project(self.node_to_edge_node, x)
        edge_query = self._project(self.node_to_edge_edge, edge_context)
        node_to_edge_score = (
            node_key[node_index] * edge_query[edge_index]
        ).sum(dim=-1) / self.attention_channels**0.5
        node_to_edge_scale = torch.nn.functional.softplus(
            self.node_to_edge_raw_scale
        )
        node_to_edge = segment_softmax(
            node_to_edge_scale * node_to_edge_score,
            edge_index,
            num_nodes=num_edges,
        ).mean(dim=-1)

        edge_message = x.new_zeros((num_edges, x.shape[-1]))
        edge_message.index_add_(
            0, edge_index, node_to_edge[:, None] * x[node_index]
        )

        node_query = self._project(self.edge_to_node_node, x)
        edge_key = self._project(self.edge_to_node_edge, edge_message)
        edge_to_node_score = (
            node_query[node_index] * edge_key[edge_index]
        ).sum(dim=-1) / self.attention_channels**0.5
        edge_to_node_scale = torch.nn.functional.softplus(
            self.edge_to_node_raw_scale
        )
        edge_to_node = segment_softmax(
            edge_to_node_scale * edge_to_node_score,
            node_index,
            num_nodes=x.shape[0],
        ).mean(dim=-1)
        return edge_message, node_to_edge, edge_to_node

    def forward(self, x, hyperedge_index, edges_per_graph=None):
        node_index, edge_index = hyperedge_index
        edge_message, _, edge_to_node = self.compute_attention(x, hyperedge_index)
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(
            0, node_index, edge_to_node[:, None] * edge_message[edge_index]
        )
        output = self.hyper_conv.lin(aggregated)
        if self.hyper_conv.bias is not None:
            output = output + self.hyper_conv.bias
        return output


class _AdditiveIncidenceScorer(nn.Module):
    """Bahdanau-style additive scorer for node-hyperedge incidences."""

    def __init__(
        self,
        in_channels,
        num_heads,
        attention_channels,
        initial_logit_scale,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attention_channels = int(attention_channels)
        projection_channels = self.num_heads * self.attention_channels
        self.node_projection = nn.Linear(in_channels, projection_channels)
        self.edge_projection = nn.Linear(in_channels, projection_channels)
        self.score_vector = nn.Parameter(
            torch.empty(self.num_heads, self.attention_channels)
        )
        nn.init.xavier_uniform_(self.score_vector)
        self.logit_scale = nn.Parameter(
            torch.full((self.num_heads,), math.log(float(initial_logit_scale)))
        )

    def forward(self, node_values, edge_values, node_index, edge_index, groups):
        node_embedding = self.node_projection(node_values).view(
            node_values.shape[0], self.num_heads, self.attention_channels
        )
        edge_embedding = self.edge_projection(edge_values).view(
            edge_values.shape[0], self.num_heads, self.attention_channels
        )
        joint = torch.tanh(
            node_embedding[node_index] + edge_embedding[edge_index]
        )
        score = (joint * self.score_vector[None]).sum(dim=-1)
        score = score / self.attention_channels**0.5
        scale = self.logit_scale.exp().clamp(max=20.0)
        return segment_softmax(
            score * scale[None], groups, num_nodes=edge_values.shape[0]
        ).mean(dim=-1)


class AdditiveIncidenceAttentionHypergraph(nn.Module):
    """Two-stage additive incidence attention with an exact uniform limit."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=2,
        attention_channels=8,
        initial_logit_scale=1.0,
    ):
        super().__init__()
        if num_heads <= 0 or attention_channels <= 0:
            raise ValueError("attention head counts and dimensions must be positive")
        if initial_logit_scale <= 0.0:
            raise ValueError("initial_logit_scale must be positive")
        self.hyper_conv = HypergraphConv(in_channels, out_channels)
        self.num_heads = int(num_heads)
        self.attention_channels = int(attention_channels)
        scorer_kwargs = {
            "in_channels": in_channels,
            "num_heads": num_heads,
            "attention_channels": attention_channels,
            "initial_logit_scale": initial_logit_scale,
        }
        self.node_to_edge_attention = _AdditiveIncidenceScorer(**scorer_kwargs)
        self.edge_to_node_attention = _AdditiveIncidenceScorer(**scorer_kwargs)

    @staticmethod
    def _edge_count(hyperedge_index):
        if hyperedge_index.numel() == 0:
            return 0
        return int(hyperedge_index[1].max()) + 1

    @staticmethod
    def _mean_by_index(values, index, count):
        result = values.new_zeros((count, values.shape[-1]))
        result.index_add_(0, index, values)
        denominator = values.new_zeros((count, 1))
        denominator.index_add_(
            0, index, values.new_ones((index.numel(), 1))
        )
        return result / denominator.clamp_min(1.0)

    def compute_attention(self, x, hyperedge_index):
        node_index, edge_index = hyperedge_index
        num_edges = self._edge_count(hyperedge_index)
        if num_edges != x.shape[0]:
            raise ValueError(
                "Incidence attention requires one center hyperedge per local node"
            )

        edge_context = self._mean_by_index(x[node_index], edge_index, num_edges)
        node_to_edge = self.node_to_edge_attention(
            x, edge_context, node_index, edge_index, edge_index
        )
        edge_message = x.new_zeros((num_edges, x.shape[-1]))
        edge_message.index_add_(
            0, edge_index, node_to_edge[:, None] * x[node_index]
        )

        edge_to_node = self.edge_to_node_attention(
            x, edge_message, node_index, edge_index, node_index
        )
        return edge_message, node_to_edge, edge_to_node

    def forward(self, x, hyperedge_index, edges_per_graph=None):
        node_index, edge_index = hyperedge_index
        edge_message, _, edge_to_node = self.compute_attention(x, hyperedge_index)
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(
            0, node_index, edge_to_node[:, None] * edge_message[edge_index]
        )
        output = self.hyper_conv.lin(aggregated)
        if self.hyper_conv.bias is not None:
            output = output + self.hyper_conv.bias
        return output


class LocalHypergraph(nn.Module):
    """局部超图的基类，定义通用方法。"""
    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        temporal_size,
        attention_module: nn.Module,
        scales=("small", "large"),
        scale_gates=None,
        scale_fusion="linear",
        scale_gate_hidden_channels=16,
    ):
        super().__init__()
        scales = tuple(scales)
        if not scales or any(scale not in {"small", "large"} for scale in scales):
            raise ValueError("hypergraph scales must contain 'small' and/or 'large'")
        if len(set(scales)) != len(scales):
            raise ValueError("hypergraph scales must not contain duplicates")
        if scale_gates is None:
            scale_gates = (1.0,) * len(scales)
        scale_gates = tuple(float(gate) for gate in scale_gates)
        if len(scale_gates) != len(scales):
            raise ValueError("hypergraph scale gates must match the configured scales")
        if any(gate not in {0.0, 1.0} for gate in scale_gates):
            raise ValueError("hypergraph scale gates must be fixed binary values")
        if not any(scale_gates):
            raise ValueError("at least one hypergraph scale gate must be enabled")
        self.scales = scales
        self.register_buffer(
            "scale_gates", torch.tensor(scale_gates, dtype=torch.float32), persistent=False
        )
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size

        if "small" in scales:
            self.register_buffer('hyperedge_small', self.create_hyperedges(3))
            self.conv_small = attention_module(in_channels, out_channels)
        if "large" in scales:
            self.register_buffer('hyperedge_large', self.create_hyperedges(5))
            self.conv_large = attention_module(in_channels, out_channels)

        self.fusion = nn.Linear(len(scales) * out_channels, out_channels)
        if scale_fusion not in {"linear", "node_gate", "fixed_uniform_gate"}:
            raise ValueError("Unsupported hypergraph scale fusion")
        self.scale_fusion = scale_fusion
        self.scale_gate = None
        if scale_fusion != "linear":
            # Adding the scorer must not shift backbone initialization or RNG state.
            with torch.random.fork_rng(devices=[]):
                self.scale_gate = NodeScaleGate(
                    out_channels,
                    scale_gates,
                    hidden_channels=scale_gate_hidden_channels,
                    adaptive=scale_fusion == "node_gate",
                )

    def fuse_scale_outputs(self, outputs):
        features = (
            torch.cat(outputs, dim=-1)
            if self.scale_gate is None
            else self.scale_gate(outputs)
        )
        return self.fusion(features)

    def create_hyperedges(self, kernel_size):
        H, W = self.spatial_size; T = self.temporal_size; pad = (kernel_size - 1) // 2
        hyperedges = []; edge_idx = 0
        for t in range(T):
            for i in range(H):
                for j in range(W):
                    min_t, max_t = max(0, t - pad), min(T - 1, t + pad)
                    min_i, max_i = max(0, i - pad), min(H - 1, i + pad)
                    min_j, max_j = max(0, j - pad), min(W - 1, j + pad)
                    indices = [dt*H*W + di*W + dj for dt in range(min_t, max_t + 1) for di in range(min_i, max_i + 1) for dj in range(min_j, max_j + 1)]
                    if indices:
                        hyperedges.extend([(node_idx, edge_idx) for node_idx in indices]); edge_idx += 1
        return torch.tensor(hyperedges, dtype=torch.long).t().contiguous()

    def forward(self, x):
        raise NotImplementedError

class LocalHypergraph_BroadcastBatching(LocalHypergraph):
    def forward(self, x):
        B, C, T, H, W = x.shape
        num_nodes = T * H * W
        nodes = x.permute(0, 2, 3, 4, 1).reshape(B * num_nodes, C)
        
        offsets = (torch.arange(B, device=x.device) * num_nodes).reshape(-1, 1, 1)
        outputs = []
        for index, scale in enumerate(self.scales):
            gate = self.scale_gates[index].to(dtype=nodes.dtype)
            if gate.item() == 0.0:
                disabled_branch = getattr(self, f"conv_{scale}")
                zero_parameter_dependency = sum(
                    (parameter.sum() * 0.0 for parameter in disabled_branch.parameters()),
                    nodes.new_zeros(()),
                )
                outputs.append(
                    nodes.new_zeros((B * num_nodes, self.fusion.out_features))
                    + zero_parameter_dependency
                )
                continue
            hyperedge = getattr(self, f"hyperedge_{scale}")
            batched = (hyperedge.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            output = getattr(self, f"conv_{scale}")(
                nodes, batched, edges_per_graph=num_nodes
            )
            outputs.append(output * gate)

        fused = self.fuse_scale_outputs(outputs)
        return fused.view(B, T, H, W, -1).permute(0, 4, 1, 2, 3)


class LocalHypergraph_LegacyBroadcastBatching(LocalHypergraph):
    """Reproduce the historical DD-Net batch-coupled index layout exactly."""

    def forward(self, x):
        B, C, T, H, W = x.shape
        num_nodes = T * H * W
        nodes = x.permute(0, 2, 3, 4, 1).reshape(B * num_nodes, C)

        offsets = (torch.arange(B, device=x.device) * num_nodes).reshape(-1, 1, 1)
        outputs = []
        for index, scale in enumerate(self.scales):
            hyperedge = getattr(self, f"hyperedge_{scale}")
            batched = (hyperedge.unsqueeze(0) + offsets).reshape(2, -1)
            output = getattr(self, f"conv_{scale}")(
                nodes, batched, edges_per_graph=num_nodes
            )
            outputs.append(output * self.scale_gates[index].to(dtype=output.dtype))

        fused = self.fusion(torch.cat(outputs, dim=-1))
        return fused.view(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
    



class WaveletTransform2D(nn.Module):
    """Compute a two-dimensional wavelet transform.
        loss = nn.MSELoss()
        data = torch.rand(1, 3, 128, 256)
        DWT = WaveletTransform2D()
        IDWT = WaveletTransform2D(inverse=True)

        LL, LH, HL, HH = DWT(data)
        recdata = IDWT([LL, LH, HL, HH])
        print(loss(data, recdata))
    """
    def __init__(self, inverse=False, wavelet="haar", mode="constant"):
        super(WaveletTransform2D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        # construct 2d filter
        self.dim_size = lo.shape[-1]
        ll = self.outer(lo, lo)
        lh = self.outer(hi, lo)
        hl = self.outer(lo, hi)
        hh = self.outer(hi, hi)
        filters = torch.stack([ll, lh, hl, hh],dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)  # [4, 1, height, width]

    def outer(self, a: torch.Tensor, b: torch.Tensor):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        # pad to even singal length.
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        padb, padt = self.get_pad(data.shape[-2], self.dim_size)
        padr, padl = self.get_pad(data.shape[-1], self.dim_size)

        data_pad = torch.nn.functional.pad(data, [padl, padr, padt, padb], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(torch.nn.functional.conv2d(data, filter.repeat(c, 1, 1, 1), stride=2, groups=c))
            return dec_res
        else:
            b, c, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, h, w)
            rec_res = torch.nn.functional.conv_transpose2d(data, self.filters.repeat(c, 1, 1, 1), stride=2, groups=c)
            return rec_res



class WaveletTransform3D(nn.Module):
    """Compute a three-dimensional wavelet transform.
        Example:
            loss = nn.MSELoss()
            data = torch.rand(1, 3, 10, 128, 256)
            DWT = WaveletTransform3D()
            IDWT = WaveletTransform3D(inverse=True)

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = DWT(data)
            recdata = IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
            print(loss(data, recdata))

            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = DWT_3D(data)
            recdata = IDWT_3D(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
            print(loss(data, recdata))
        """
    def __init__(self, inverse=False, wavelet="haar", mode="constant"):
        super(WaveletTransform3D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        # construct 3d filter
        self.dim_size = lo.shape[-1]
        size = [self.dim_size] * 3
        lll = self.outer(lo, self.outer(lo, lo)).reshape(size)
        llh = self.outer(lo, self.outer(lo, hi)).reshape(size)
        lhl = self.outer(lo, self.outer(hi, lo)).reshape(size)
        lhh = self.outer(lo, self.outer(hi, hi)).reshape(size)
        hll = self.outer(hi, self.outer(lo, lo)).reshape(size)
        hlh = self.outer(hi, self.outer(lo, hi)).reshape(size)
        hhl = self.outer(hi, self.outer(hi, lo)).reshape(size)
        hhh = self.outer(hi, self.outer(hi, hi)).reshape(size)
        filters = torch.stack([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)  # [8, 1, length, height, width]
        
    def outer(self, a: torch.Tensor, b: torch.Tensor):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        # pad to even singal length.
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        pad_back, pad_front = self.get_pad(data.shape[-3], self.dim_size)
        pad_bottom, pad_top = self.get_pad(data.shape[-2], self.dim_size)
        pad_right, pad_left = self.get_pad(data.shape[-1], self.dim_size)
        data_pad = torch.nn.functional.pad(
            data, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, t, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(torch.nn.functional.conv3d(data, filter.repeat(c, 1, 1, 1, 1), stride=2, groups=c))
            return dec_res
        else:
            b, c, t, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, t, h, w)
            rec_res = torch.nn.functional.conv_transpose3d(data, self.filters.repeat(c, 1, 1, 1, 1), stride=2, groups=c)
            return rec_res



class FrequencyAttention(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=32):
        super(FrequencyAttention, self).__init__()
        self.avgpool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgpool_w = nn.AdaptiveAvgPool2d((1, None))

        hidden_dim = max(8, in_dim // reduction)

        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act = activations.HardSwish(inplace=True)

        self.conv_h = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.avgpool_h(x)  # b c h 1
        x_w = self.avgpool_w(x).permute(0, 1, 3, 2)  # b c w 1

        y = torch.cat([x_h, x_w], dim=2)  # b c (h+w) 1
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class TF_AwareBlock_DCN(nn.Module):
    """
    时空可变形卷积
    """
    def __init__(
        self,
        dim,
        mlp_ratio=4.,
        drop=0.,
        ls_init_value=1e-2,
        drop_path=0.1,
        residual_layer_scale_mode="shared_openstl",
        use_modulation_mask=True,
        use_temporal_mixer=True,
        modulation_mask_gate=None,
        temporal_mixer_gate=None,
    ):
        super().__init__()
        if residual_layer_scale_mode not in {"shared_openstl", "independent"}:
            raise ValueError(
                "residual_layer_scale_mode must be 'shared_openstl' or 'independent'"
            )
        self.residual_layer_scale_mode = residual_layer_scale_mode
        self.use_modulation_mask = use_modulation_mask
        self.use_temporal_mixer = use_temporal_mixer
        for name, value in (
            ("modulation_mask_gate", modulation_mask_gate),
            ("temporal_mixer_gate", temporal_mixer_gate),
        ):
            if value is not None and float(value) not in {0.0, 1.0}:
                raise ValueError(f"{name} must be a fixed binary value")
        if modulation_mask_gate is not None and not use_modulation_mask:
            raise ValueError("modulation_mask_gate requires the mask branch to be retained")
        if temporal_mixer_gate is not None and not use_temporal_mixer:
            raise ValueError("temporal_mixer_gate requires the temporal branch to be retained")
        self.modulation_mask_gate = (
            None if modulation_mask_gate is None else float(modulation_mask_gate)
        )
        self.temporal_mixer_gate = (
            None if temporal_mixer_gate is None else float(temporal_mixer_gate)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        # --- DCN 相关层 ---
        # 定义DCN的卷积核大小和padding
        self.dcn_kernel_size = 3
        self.dcn_padding = 1
        
        # 1. 一个标准卷积层，用于从输入特征动态生成offset和mask
        # The default branch predicts x/y offsets and one modulation mask.
        offset_channels = (3 if use_modulation_mask else 2) * self.dcn_kernel_size ** 2
        self.offset_mask_conv = nn.Conv2d(
            dim, 
            offset_channels,
            kernel_size=self.dcn_kernel_size,
            padding=self.dcn_padding
        )

        # 2. 可变形卷积层
        self.deformable_conv = DeformConv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.dcn_kernel_size,
            padding=self.dcn_padding,
            bias=False
        )
        self.norm_dcn = nn.BatchNorm2d(dim)
        
        # 保留并行的时序混合器分支
        if use_temporal_mixer:
            self.temporal_mixer = InvertedResidual(
                in_chs=dim, out_chs=dim, dw_kernel_size=7, exp_ratio=mlp_ratio,
                se_layer=partial(SqueezeExcite, rd_ratio=0.25), noskip=True
            )

        # 保留LayerScale
        self.layer_scale_1 = nn.Parameter(ls_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(ls_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        identity = x

        # --- 空间自适应分支 ---
        attn = self.norm1(x)
        
        # 1. 生成offset和mask
        offset_mask = self.offset_mask_conv(attn)
        if self.use_modulation_mask:
            o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((o1, o2), dim=1)
            mask = mask.sigmoid()
            if self.modulation_mask_gate == 0.0:
                mask = None
        else:
            offset = offset_mask
            mask = None

        # 2. 应用可变形卷积
        spatial_features = self.deformable_conv(attn, offset, mask)
        spatial_features = self.norm_dcn(spatial_features)

        # 第一个残差连接
        x = identity + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * spatial_features)

        # --- 时序混合分支 ---
        if self.use_temporal_mixer:
            temporal_features = self.temporal_mixer(self.norm2(x))
            # OpenSTL WaST applies layer_scale_1 to both residual branches.
            temporal_scale = (
                self.layer_scale_1
                if self.residual_layer_scale_mode == "shared_openstl"
                else self.layer_scale_2
            )
            temporal_residual = self.drop_path(
                temporal_scale.unsqueeze(-1).unsqueeze(-1) * temporal_features
            )
            temporal_gate = (
                1.0 if self.temporal_mixer_gate is None else self.temporal_mixer_gate
            )
            x = x + temporal_gate * temporal_residual
        
        return x
    

class TF_AwareBlocks(nn.Module):
    def __init__(self, dim, num_blocks, drop_path, use_bottleneck=None, use_hid=False, mlp_ratio=4., drop=0., ls_init_value=1e-2, large_kernel=51, small_kernel=5, use_dcn=True, residual_layer_scale_mode="shared_openstl", use_modulation_mask=True, use_temporal_mixer=True, modulation_mask_gate=None, temporal_mixer_gate=None): # 新增 use_dcn 开关
        super().__init__()
        assert len(drop_path) == num_blocks, "drop_path list doesn't match num_blocks"
        self.use_hid = use_hid
        self.use_bottleneck = use_bottleneck

        blocks = []
        for i in range(num_blocks):
            # --- 根据 use_dcn 开关选择要创建的Block ---
            if use_dcn:
                # 如果为True, 创建DCN版本的Block
                # 注意：DCN版本不需要 large_kernel 和 small_kernel 参数
                block = TF_AwareBlock_DCN(
                    dim,
                    mlp_ratio,
                    drop,
                    ls_init_value,
                    drop_path[i],
                    residual_layer_scale_mode=residual_layer_scale_mode,
                    use_modulation_mask=use_modulation_mask,
                    use_temporal_mixer=use_temporal_mixer,
                    modulation_mask_gate=modulation_mask_gate,
                    temporal_mixer_gate=temporal_mixer_gate,
                )
            # else:
            #     # 否则，创建原始版本的Block
            #     block = TF_AwareBlock(dim, mlp_ratio, drop, ls_init_value, drop_path[i], large_kernel, small_kernel)
            blocks.append(block)
        
        self.blocks = nn.Sequential(*blocks)
        self.concat_block = nn.Conv2d(dim * 2, dim, 3, 1, 1) if use_hid==True else None

        self.DWT = WaveletTransform3D(inverse=False) if use_bottleneck == "decompose" else None
        self.IDWT = WaveletTransform3D(inverse=True) if use_bottleneck == "decompose" else None

    def forward(self, x, skip=None):  # 前向传播逻辑保持不变
        if self.concat_block is not None and self.use_bottleneck is None:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x = self.concat_block(torch.cat([x, skip], dim=1))
            x = self.blocks(x)
            x = rearrange(x, 'b (c t) h w -> b c t h w', t=t)
            return x
        elif self.concat_block is None and self.use_bottleneck is None:
            b, c, t, h, w = x.shape
            x = rearrange(x, 'b c t h w -> b (c t) h w')
            x_skip = self.blocks(x) # 修正了一个小bug, x = skip= self.blocks(x) -> x_skip = self.blocks(x)
            x = rearrange(x_skip, 'b (c t) h w -> b c t h w', t=t)
            return x, x_skip
        elif self.use_bottleneck is not None:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x) if self.use_bottleneck == "decompose" else [x, None, None, None, None, None, None, None]
            b, c, t, h, w = LLL.shape
            LLL = rearrange(LLL, 'b c t h w -> b (c t) h w')
            LLL = self.blocks(LLL)
            LLL = rearrange(LLL, 'b (c t) h w -> b c t h w', t=t)
            x = self.IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH]) if self.use_bottleneck == "decompose" else LLL
            return x

    
            
class Wavelet_3D_Embedding(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim=None, subband_grouping="openstl"):
        super().__init__()
        if subband_grouping not in {"openstl", "axis_consistent"}:
            raise ValueError("subband_grouping must be 'openstl' or 'axis_consistent'")
        self.subband_grouping = subband_grouping
        emb_dim = in_dim if emb_dim==None else emb_dim
        self.conv_0 = nn.Sequential(nn.Conv3d(in_dim, in_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),),
                    nn.BatchNorm3d(in_dim),
                    nn.GELU(),)
        self.conv_1 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),),
                    nn.BatchNorm3d(out_dim),
                    nn.GELU(),)

        self.conv_emb = nn.Conv3d(emb_dim * 4, out_dim, kernel_size=(3, 3, 3),stride=(1, 1, 1),padding=(1, 1, 1),)

        self.DWT = WaveletTransform3D(inverse=False)

    def forward(self, x, x_emb=None):
        # embedding branch
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x_emb)
        if self.subband_grouping == "openstl":
            lo_temp = torch.cat([LLL, LHL, HLL, HHL], dim=1)
            hi_temp = torch.cat([LLH, LHH, HLH, HHH], dim=1)
        else:
            lo_temp = torch.cat([LLL, LLH, LHL, LHH], dim=1)
            hi_temp = torch.cat([HLL, HLH, HHL, HHH], dim=1)
        x_emb = torch.cat([lo_temp, hi_temp], dim=2)
        x_emb = self.conv_emb(x_emb)
        # downsampling branch
        x = self.conv_0(x)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.DWT(x)
        if self.subband_grouping == "openstl":
            spatio_lo_coeffs = torch.cat([LLL, LLH], dim=2)
            spatio_hi_coeffs = torch.cat([LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        else:
            spatio_lo_coeffs = torch.cat([LLL, HLL], dim=2)
            spatio_hi_coeffs = torch.cat([LLH, LHL, LHH, HLH, HHL, HHH], dim=1)
        x = self.conv_1(spatio_lo_coeffs)
        return (x + x_emb), spatio_hi_coeffs


class Wavelet_3D_Reconstruction(nn.Module):
    def __init__(self, in_dim, out_dim, hi_dim):
        super().__init__()
        self.conv_0 = nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),)

        self.conv_hi =  nn.Sequential(nn.Conv3d(int(hi_dim * 6), int(out_dim * 6), kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=6),
            nn.BatchNorm3d(out_dim * 6),
            nn.GELU(),)

        self.IDWT = WaveletTransform3D(inverse=True)

    def forward(self, x, skip_hi=None):
        LLL, LLH = torch.chunk(self.conv_0(x), chunks=2, dim=2)
        LHL, LHH, HLL, HLH, HHL, HHH = torch.chunk(self.conv_hi(skip_hi), chunks=6, dim=1)
        x = self.IDWT([LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH])
        return x


class Learned_3D_Downsampling(nn.Module):
    """Shape-matched learned replacement for the wavelet analysis stage."""

    def __init__(self, in_dim, out_dim, emb_dim=None):
        super().__init__()
        emb_dim = in_dim if emb_dim is None else emb_dim
        self.feature_down = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
        )
        self.embedding_down = nn.Sequential(
            nn.Conv3d(emb_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
        )

    def forward(self, x, x_emb=None):
        if x_emb is None:
            raise ValueError("Learned downsampling requires the input embedding tensor")
        return self.feature_down(x) + self.embedding_down(x_emb), x


class Learned_3D_Upsampling(nn.Module):
    """Shape-matched learned replacement for the wavelet reconstruction stage."""

    def __init__(self, in_dim, out_dim, hi_dim):
        super().__init__()
        self.feature_up = nn.Sequential(
            nn.ConvTranspose3d(
                in_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2)
            ),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
        )
        self.skip_projection = nn.Conv3d(hi_dim, out_dim, kernel_size=1)

    def forward(self, x, skip_hi=None):
        if skip_hi is None:
            raise ValueError("Learned upsampling requires the encoder skip tensor")
        return self.feature_up(x) + self.skip_projection(skip_hi)


class DD_Net(nn.Module):
    """
    一个可配置的WaST模型，用于消融实验。
    可通过参数选择不同的“注意力机制”和“批量处理”方式。
    """
    def __init__(self, in_shape, encoder_dim, block_list, 
                 attention_mode='node', 
                 batching_mode='broadcast', 
                 drop_path_rate=0.1, mlp_ratio=4.,
                 wavelet_subband_grouping="openstl",
                 residual_layer_scale_mode="shared_openstl",
                 hypergraph_scales=("small", "large"),
                 hypergraph_scale_gates=None,
                 hypergraph_scale_fusion="linear",
                 scale_gate_hidden_channels=16,
                 hypergraph_input_stem_channels=0,
                 hypergraph_input_stem_kernel_size=3,
                 hyperedge_weighting="adaptive",
                 incidence_attention_heads=2,
                 incidence_attention_channels=8,
                 incidence_attention_logit_scale=0.1,
                 use_modulation_mask=True,
                 use_temporal_mixer=True,
                 modulation_mask_gate=None,
                 temporal_mixer_gate=None,
                 use_wavelet=True,
                 **kwargs):
        super().__init__()
        frame, in_dim, H, W = in_shape
        

        # 1. 注意力模块
        if attention_mode != 'node':
            raise ValueError(f"Unsupported DD-Net attention mode: {attention_mode}")
        if hyperedge_weighting == "adaptive":
            attention_module = NodeAttention
        elif hyperedge_weighting == "uniform":
            attention_module = UniformHypergraph
        elif hyperedge_weighting == "strict_uniform":
            attention_module = StrictUniformNodeAttention
        elif hyperedge_weighting == "corrected_adaptive":
            attention_module = CorrectedAdaptiveHypergraph
        elif hyperedge_weighting == "corrected_strict_uniform":
            attention_module = StrictUniformCorrectedHypergraph
        elif hyperedge_weighting == "incidence_attention":
            attention_module = partial(
                IncidenceAttentionHypergraph,
                num_heads=incidence_attention_heads,
                attention_channels=incidence_attention_channels,
                initial_logit_scale=incidence_attention_logit_scale,
            )
        elif hyperedge_weighting == "additive_incidence_attention":
            attention_module = partial(
                AdditiveIncidenceAttentionHypergraph,
                num_heads=incidence_attention_heads,
                attention_channels=incidence_attention_channels,
                initial_logit_scale=incidence_attention_logit_scale,
            )
        else:
            raise ValueError(
                "hyperedge_weighting must be 'adaptive', 'uniform', "
                "'strict_uniform', 'corrected_adaptive', or "
                "'corrected_strict_uniform', 'incidence_attention', or "
                "'additive_incidence_attention'"
            )

        # 2. 根据配置选择批量处理模块
        if batching_mode == 'broadcast':
            local_class = LocalHypergraph_BroadcastBatching
            stem_options = {}
            if hypergraph_input_stem_channels:
                from .feature_stem import SharedStemLocalHypergraph

                local_class = SharedStemLocalHypergraph
                stem_options = {
                    "stem_channels": int(hypergraph_input_stem_channels),
                    "stem_kernel_size": int(hypergraph_input_stem_kernel_size),
                }
            self.conv_in = local_class(
                in_channels=in_dim, out_channels=encoder_dim,
                spatial_size=(H, W), temporal_size=frame,
                attention_module=attention_module,
                scales=hypergraph_scales,
                scale_gates=hypergraph_scale_gates,
                scale_fusion=hypergraph_scale_fusion,
                scale_gate_hidden_channels=scale_gate_hidden_channels,
                **stem_options,
            )
        elif batching_mode == 'legacy_broadcast':
            if hypergraph_input_stem_channels:
                raise ValueError("The shared feature stem requires sample-isolated batching")
            if hypergraph_scale_fusion != "linear":
                raise ValueError("Scale-fusion candidates require isolated broadcast batching")
            self.conv_in = LocalHypergraph_LegacyBroadcastBatching(
                in_channels=in_dim, out_channels=encoder_dim,
                spatial_size=(H, W), temporal_size=frame,
                attention_module=attention_module,
                scales=hypergraph_scales,
                scale_gates=hypergraph_scale_gates,
            )
        else:
            raise ValueError(f"Unsupported DD-Net batching mode: {batching_mode}")
            
        # 3. 保持 WaST 的主干拓扑；translator block 内使用 DD-Net DCN Core
        dp_list = [x.item() for x in torch.linspace(0, drop_path_rate, sum(block_list))]
        indexes = list(accumulate(block_list))
        dp_list = [dp_list[start:end] for start, end in zip([0] + indexes, indexes)]


        block_options = {
            "residual_layer_scale_mode": residual_layer_scale_mode,
            "use_modulation_mask": use_modulation_mask,
            "use_temporal_mixer": use_temporal_mixer,
            "modulation_mask_gate": modulation_mask_gate,
            "temporal_mixer_gate": temporal_mixer_gate,
        }
        self.translator1 = TF_AwareBlocks(dim=encoder_dim * frame, num_blocks=block_list[0], drop_path=dp_list[0], mlp_ratio=mlp_ratio, large_kernel=51, small_kernel=5, **block_options)

        if use_wavelet:
            self.wavelet_embed1 = Wavelet_3D_Embedding(in_dim=encoder_dim, out_dim=encoder_dim * 2, emb_dim=in_dim, subband_grouping=wavelet_subband_grouping)
            self.wavelet_recon1 = Wavelet_3D_Reconstruction(in_dim=encoder_dim * 2, out_dim=encoder_dim, hi_dim=encoder_dim)
        else:
            self.wavelet_embed1 = Learned_3D_Downsampling(in_dim=encoder_dim, out_dim=encoder_dim * 2, emb_dim=in_dim)
            self.wavelet_recon1 = Learned_3D_Upsampling(in_dim=encoder_dim * 2, out_dim=encoder_dim, hi_dim=encoder_dim)

        self.bottleneck_translator = TF_AwareBlocks(dim=encoder_dim * 2 * frame, num_blocks=block_list[1], drop_path=dp_list[1], use_bottleneck=True, mlp_ratio=mlp_ratio, large_kernel=21, small_kernel=5, **block_options)

        self.translator2 = TF_AwareBlocks(dim=encoder_dim * frame, num_blocks=block_list[2], drop_path=dp_list[2], use_hid=True, mlp_ratio=mlp_ratio, large_kernel=51, small_kernel=5, **block_options)

        self.conv_out = nn.Sequential(
                    nn.BatchNorm3d(encoder_dim),
                    nn.GELU(),
                    nn.Conv3d(
                        encoder_dim,
                        in_dim,
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1))
        )

    def forward(self, x):
            # 1. 整理输入维度，与原始模型一致
            x = rearrange(x, 'b t c h w -> b c t h w')
            ori_img = x

            # 2. 通过我们可配置的超图入口层
            # conv_in 输出的是5D张量: [B, C_enc, T, H, W]
            x = self.conv_in(x)

            # 3. 遵循 WaST 主干的数据流，translator block 使用 DD-Net DCN Core
            # translator1 接收5D张量，并返回5D的 x 和4D的 tskip1
            x, tskip1 = self.translator1(x)
            
            # wavelet_embed1 接收5D张量
            x, skip1 = self.wavelet_embed1(x, x_emb=ori_img)

            # bottleneck_translator 接收5D张量
            x = self.bottleneck_translator(x)

            # wavelet_recon1 接收5D张量
            x = self.wavelet_recon1(x, skip1)

            # translator2 接收5D张量和4D的skip connection
            x = self.translator2(x, tskip1)
            
            # 最终输出层
            x = self.conv_out(x)
            
            # 恢复为标准的 [B, T, C, H, W] 格式
            x = rearrange(x, 'b c t h w -> b t c h w')
            return x
