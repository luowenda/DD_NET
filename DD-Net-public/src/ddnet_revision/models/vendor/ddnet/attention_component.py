"""Final attention-v1 encoder paired with the original WaST core."""

from itertools import accumulate

import torch

from ..wast.model import TF_AwareBlocks as WaSTBlocks
from .model import DD_Net


class AttentionV1HypergraphOnly(DD_Net):
    def __init__(self, *, in_shape, encoder_dim, block_list,
                 drop_path_rate=0.1, mlp_ratio=4.0, use_dcn_core=False, **kwargs):
        if use_dcn_core:
            raise ValueError("This component ablation must use the original WaST core")
        super().__init__(in_shape=in_shape, encoder_dim=encoder_dim,
                         block_list=block_list, drop_path_rate=drop_path_rate,
                         mlp_ratio=mlp_ratio, **kwargs)
        rates = torch.linspace(0, drop_path_rate, sum(block_list)).tolist()
        indexes = list(accumulate(block_list))
        groups = [rates[start:end] for start, end in zip([0] + indexes, indexes)]
        # Preserve the reference encoder, common weights and training RNG. Only
        # the three translator groups are replaced by the original WaST blocks.
        with torch.random.fork_rng(devices=[]):
            self.translator1 = WaSTBlocks(
                dim=encoder_dim * in_shape[0], num_blocks=block_list[0],
                drop_path=groups[0], mlp_ratio=mlp_ratio,
                large_kernel=51, small_kernel=5,
            )
            self.bottleneck_translator = WaSTBlocks(
                dim=encoder_dim * 2 * in_shape[0], num_blocks=block_list[1],
                drop_path=groups[1], mlp_ratio=mlp_ratio, use_bottleneck=True,
                large_kernel=21, small_kernel=5,
            )
            self.translator2 = WaSTBlocks(
                dim=encoder_dim * in_shape[0], num_blocks=block_list[2],
                drop_path=groups[2], mlp_ratio=mlp_ratio, use_hid=True,
                large_kernel=51, small_kernel=5,
            )
