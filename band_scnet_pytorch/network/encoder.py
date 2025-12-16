import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, Tuple, Sequence, List

class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        num_modules: int = 1,
        activation_cls=nn.ReLU,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            k = (kernel_size, kernel_size)
            padding = (kernel_size // 2, kernel_size // 2)
        else:
            k = kernel_size
            padding = tuple(kk // 2 for kk in kernel_size)

        layers = []
        for i in range(num_modules):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_channels,
                        kernel_size=k,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    activation_cls(),
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SDLayer(nn.Module):
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

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, stride_f),
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class SDBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        band_strides: Sequence[int] = (1, 4, 16),
        band_kernels: Sequence[int] = (3, 3, 3),
        band_ratios: Tuple[float, float, float] = (0.175, 0.392, 0.433),
        conv_depths: Sequence[int] = (3, 2, 1),
        conv_kernel: int = 3,
    ):
        super().__init__()
        assert len(band_strides) == len(band_kernels) == len(conv_depths) == 3
        assert abs(sum(band_ratios) - 1.0) < 1e-5, "band_ratios must sum to 1."

        self.band_strides = list(band_strides)
        self.band_kernels = list(band_kernels)
        self.band_ratios = band_ratios

        self.down_convs = nn.ModuleList()
        for stride, k in zip(band_strides, band_kernels):
            self.down_convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, k),
                    stride=(1, stride),
                    padding=(0, 0),
                    bias=False,
                )
            )

        self.conv_modules = nn.ModuleList(
            [
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel,
                    num_modules=depth,
                    activation_cls=nn.ReLU,
                )
                for depth in conv_depths
            ]
        )

        self.global_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel,
            stride=1,
            padding=conv_kernel // 2,
            bias=False,
        )

    def _split_freq(self, F: int) -> List[Tuple[int, int]]:
        r_low, r_mid, r_high = self.band_ratios
        assert abs(r_low + r_mid + r_high - 1.0) < 1e-5

        low_end = math.ceil(F * r_low)
        mid_end = math.ceil(F * (r_low + r_mid))
        splits = [
            (0, low_end),
            (low_end, mid_end),
            (mid_end, F),
        ]
        return splits

    def forward(self, x: torch.Tensor):
        B, C, T, F_in = x.shape
        splits = self._split_freq(F_in)

        bands_out = []
        sd_lengths = []
        original_lengths = []

        for (start, end), conv, stride, k, conv_mod in zip(
            splits, self.down_convs, self.band_strides, self.band_kernels, self.conv_modules
        ):
            band = x[..., start:end]  # (B, C_in, T, F_band)
            F_band = band.size(-1)
            original_lengths.append(F_band)
            current_len = F_band
            if stride == 1:
                total_padding = k - stride
            else:
                total_padding = (stride - current_len % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            band_padded = F.pad(band, (pad_left, pad_right, 0, 0))
            band_down = conv(band_padded)
            band_feat = conv_mod(band_down)
            band_feat = F.gelu(band_feat)

            bands_out.append(band_feat)
            sd_lengths.append(band_feat.size(-1))  # F_down

        # full-band concat
        full_band = torch.cat(bands_out, dim=3)  # (B, C_out, T, F_out)
        skip = full_band

        e = self.global_conv(full_band)         # (B, C_out, T, F_out)

        return e, skip, sd_lengths, original_lengths


class Encoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int] = (4, 32, 64, 128),
        band_strides: Sequence[int] = (1, 4, 16),
        band_kernels: Sequence[int] = (3, 3, 3),
        band_ratios: Tuple[float, float, float] = (0.175, 0.392, 0.433),
        conv_depths: Sequence[int] = (3, 2, 1),
        conv_kernel: int = 3,
    ):
        super().__init__()
        assert len(dims) >= 2

        blocks = []
        for cin, cout in zip(dims[:-1], dims[1:]):
            blocks.append(
                SDBlock(
                    in_channels=cin,
                    out_channels=cout,
                    band_strides=band_strides,
                    band_kernels=band_kernels,
                    band_ratios=band_ratios,
                    conv_depths=conv_depths,
                    conv_kernel=conv_kernel,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.dims = list(dims)

    def forward(self, x: torch.Tensor):
        skips = []
        sd_lengths_list = []
        orig_lengths_list = []

        for block in self.blocks:
            x, skip, sd_lengths, original_lengths = block(x)
            skips.append(skip)
            sd_lengths_list.append(sd_lengths)
            orig_lengths_list.append(original_lengths)

        e = x
        return e, skips, sd_lengths_list, orig_lengths_list

