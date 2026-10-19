"""Wavelet replacement without shifting the final attention-v1 backbone RNG."""

import torch

from .model import DD_Net, Learned_3D_Downsampling, Learned_3D_Upsampling


class AttentionV1WaveletReplacement(DD_Net):
    def __init__(self, *, in_shape, encoder_dim, block_list, use_wavelet=False, **kwargs):
        if use_wavelet:
            raise ValueError("This entry is only for the declared wavelet replacement")
        super().__init__(in_shape=in_shape, encoder_dim=encoder_dim,
                         block_list=block_list, use_wavelet=True, **kwargs)
        # Initialize the full reference first; replacement must not shift common
        # weights or the RNG used for training sample order and stochastic depth.
        with torch.random.fork_rng(devices=[]):
            self.wavelet_embed1 = Learned_3D_Downsampling(
                encoder_dim, encoder_dim * 2, emb_dim=in_shape[1]
            )
            self.wavelet_recon1 = Learned_3D_Upsampling(
                encoder_dim * 2, encoder_dim, hi_dim=encoder_dim
            )
