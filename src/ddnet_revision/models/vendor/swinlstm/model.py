import torch
from torch import nn

from .official import SwinLSTM


class SwinLSTM_D_Model(nn.Module):
    """Sequence wrapper around the authors' official SwinLSTM-D core."""

    def __init__(self, depths_downsample, depths_upsample, num_heads, configs):
        super().__init__()
        input_frames, channels, height, width = configs.in_shape
        if height != width:
            raise ValueError("The official SwinLSTM implementation requires square inputs")
        self.input_frames = input_frames
        self.depths_downsample = depths_downsample
        self.depths_upsample = depths_upsample
        self.model = SwinLSTM(
            img_size=height,
            patch_size=configs.patch_size,
            in_chans=channels,
            embed_dim=configs.embed_dim,
            depths_downsample=depths_downsample,
            depths_upsample=depths_upsample,
            num_heads=num_heads,
            window_size=configs.window_size,
        )
        self.loss = nn.MSELoss()

    def forward(self, frames_tensor, return_loss=True):
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        observed = frames[:, : self.input_frames]
        states_down = [None] * len(self.depths_downsample)
        states_up = [None] * len(self.depths_upsample)
        generated = []

        for frame in observed[:, :-1].unbind(dim=1):
            output, states_down, states_up = self.model(frame, states_down, states_up)
            generated.append(output)

        last_frame = observed[:, -1]
        for _ in range(frames.shape[1] - self.input_frames):
            output, states_down, states_up = self.model(
                last_frame, states_down, states_up
            )
            generated.append(output)
            last_frame = output

        prediction = torch.stack(generated, dim=1).permute(0, 1, 3, 4, 2).contiguous()
        value = self.loss(prediction, frames_tensor[:, 1:]) if return_loss else None
        return prediction, value
