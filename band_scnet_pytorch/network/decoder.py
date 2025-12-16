import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, List
from .fusion import CSAFusion

class SULayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_f: int,
        kernel_size_f: int = 3,
    ):
        super().__init__()
        self.stride_f = stride_f
        self.kernel_size_f = kernel_size_f

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size_f),
            stride=(1, stride_f),
            padding=(0, 0),
            bias=False,
        )

    def forward(self, x_band: torch.Tensor, origin_length: int) -> torch.Tensor:
        y = self.deconv(x_band)  # (B, C_out, T, F_up)
        F_up = y.size(-1)

        if F_up > origin_length:
            diff = F_up - origin_length
            left = diff // 2
            right = diff - left
            y = y[..., left:F_up - right]
        elif F_up < origin_length:
            diff = origin_length - F_up
            left = diff // 2
            right = diff - left

            y = F.pad(y, (left, right, 0, 0))

        return y

class SUBlock(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        in_channels: int,
        out_channels: int,
        band_strides: Sequence[int] = (1, 4, 16),
        band_kernels: Sequence[int] = (3, 3, 3),
    ):
        super().__init__()
        assert len(band_strides) == len(band_kernels) == 3

        self.fusion = CSAFusion(fusion_dim, in_channels)

        self.su_layers = nn.ModuleList(
            [
                SULayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride_f=stride,
                    kernel_size_f=k_f,
                )
                for stride, k_f in zip(band_strides, band_kernels)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        x_skip: torch.Tensor,
        sd_lengths: List[int],
        original_lengths: List[int],
    ) -> torch.Tensor:
        x = self.fusion(x, x_skip)  # (B, C_i, T, F_i)

        assert len(sd_lengths) == len(original_lengths) == len(self.su_layers) == 3
        splits = []
        cur = 0
        for L in sd_lengths:
            splits.append((cur, cur + L))
            cur += L

        bands_out = []
        for (start, end), su_layer, orig_len in zip(
            splits, self.su_layers, original_lengths
        ):
            x_band = x[..., start:end]  # (B, C_i, T, F_sd)
            y_band = su_layer(x_band, origin_length=orig_len)  # (B, C_{i-1}, T, F_orig)
            bands_out.append(y_band)

        y = torch.cat(bands_out, dim=3)  # (B, C_{i-1}, T, F_{i-1})
        return y

class Decoder(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 1024,
        dims: Sequence[int] = (4, 32, 64, 128),
        band_strides: Sequence[int] = (1, 4, 16),
        band_kernels: Sequence[int] = (3, 3, 3),
    ):
        super().__init__()
        ups = []
        for cin, cout in zip(dims[0:-1], dims[1:]):  # (C3->C2), (C2->C1), (C1->C0)
            ups.append(
                SUBlock(
                    fusion_dim=fusion_dim,
                    in_channels=cin,
                    out_channels=cout,
                    band_strides=band_strides,
                    band_kernels=band_kernels,
                )
            )
        self.blocks = nn.ModuleList(ups)
        self.dims = list(dims)

    def forward(
        self,
        e: torch.Tensor,
        skips: List[torch.Tensor],
        sd_lengths_list: List[List[int]],
        orig_lengths_list: List[List[int]],
    ) -> torch.Tensor:
        x = e

        for block, skip, sd_lengths, orig_lengths in zip(
            self.blocks,
            reversed(skips),
            reversed(sd_lengths_list),
            reversed(orig_lengths_list),
        ):
            x = block(x, skip, sd_lengths, orig_lengths)

        return x

