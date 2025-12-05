import torch
import torch.nn as nn
import numpy as np

from typing import Union, Tuple, Sequence
from encoder import ConvModule

class SULayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_f: int,
        kernel_size: Tuple[int, int] = (1, 3),
    ):
        super().__init__()

        kh, kw = kernel_size
        padding = (kh // 2, kw // 2)

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, stride_f),
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Sequence[int] = (16, 4, 1),
        split_ratios: Tuple[float, float] = (0.175, 0.392),
    ):
        super().__init__()
        self.split_ratios = split_ratios
        self.low_freq_block = nn.Sequential(
            SULayer(in_channels=in_channels, out_channels=in_channels, stride_f=strides[0]),
            nn.GELU(),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                num_modules=3,
                activation_cls=nn.ReLU,
            ),
        )

        self.mid_freq_block = nn.Sequential(
            SULayer(in_channels=in_channels, out_channels=in_channels, stride_f=strides[1]),
            nn.GELU(),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                num_modules=2,
                activation_cls=nn.ReLU,
            ),
        )

        self.high_freq_block = nn.Sequential(
            SULayer(in_channels=in_channels, out_channels=in_channels, stride_f=strides[2]),
            nn.GELU(),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                num_modules=1,
                activation_cls=nn.ReLU,
            ),
        )

        self.last_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

    def _split_freq(self, x: torch.Tensor):
        B, C, T, F = x.size()
        low_r, mid_r = self.split_ratios

        ls = int(F * low_r)
        ms = int(F * (low_r + mid_r))

        x_low = x[..., :ls]     # (B, C, H, 0:ls)
        x_mid = x[..., ls:ms]   # (B, C, H, ls:ms)
        x_high = x[..., ms:]    # (B, C, H, ms:)

        return x_low, x_mid, x_high

    def forward(self, x: torch.Tensor):
        x_low, x_mid, x_high = self._split_freq(x)

        l = self.low_freq_block(x_low)
        m = self.mid_freq_block(x_mid)
        h = self.high_freq_block(x_high)

        s = torch.cat([l, m, h], dim=3)   # (B, C_in, H, W')
        e = self.last_conv(s)             # (B, C_out, H, W')

        return e

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 2
    ):
        super().__init__()
        c0, c1, c2, c3 = 128, 64, 32, 2 * out_channels

        self.c_list = (c0, c1, c2, c3)

        self.su1 = SUBlock(in_channels=c0, out_channels=c1, kernel_size=3)
        self.su2 = SUBlock(in_channels=c1, out_channels=c2, kernel_size=3)
        self.su3 = SUBlock(in_channels=c2, out_channels=c3, kernel_size=3)

    def forward(self, x: torch.Tensor):
        E1 = self.su1(x)
        E2 = self.su2(E1)
        E3 = self.su3(E2)

        return E3
