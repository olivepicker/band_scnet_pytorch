import torch
import torch.nn as nn

from typing import Union, Tuple, Sequence

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
        kernel_size: Union[int, Tuple[int, int]],
        strides: Sequence[int] = (1, 4, 16),
        split_ratios: Tuple[float, float] = (0.175, 0.392),
    ):
        super().__init__()
        self.split_ratios = split_ratios
        self.low_freq_block = nn.Sequential(
            SDLayer(in_channels=in_channels, out_channels=out_channels, stride_f=strides[0]),
            nn.GELU(),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_modules=3,
                activation_cls=nn.ReLU,
            ),
        )

        self.mid_freq_block = nn.Sequential(
            SDLayer(in_channels=in_channels, out_channels=out_channels, stride_f=strides[1]),
            nn.GELU(),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_modules=2,
                activation_cls=nn.ReLU,
            ),
        )

        self.high_freq_block = nn.Sequential(
            SDLayer(in_channels=in_channels, out_channels=out_channels, stride_f=strides[2]),
            nn.GELU(),
            ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                num_modules=1,
                activation_cls=nn.ReLU,
            ),
        )

        self.last_conv = nn.Conv2d(
            in_channels=out_channels,
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

        return s, e

class Encoder(nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        win_length: int = 4096,
        in_channels: int = 2,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

        c0, c1, c2, c3 = 2 * in_channels, 32, 64, 128

        self.c_list = (c0, c1, c2, c3)

        self.sd1 = SDBlock(in_channels=c0, out_channels=c1, kernel_size=3)
        self.sd2 = SDBlock(in_channels=c1, out_channels=c2, kernel_size=3)
        self.sd3 = SDBlock(in_channels=c2, out_channels=c3, kernel_size=3)

    def stft_encode(self, wave: torch.Tensor) -> torch.Tensor:
        B, M, N = wave.shape
        device = wave.device

        window = self.window.to(device)
        wave_flat = wave.reshape(B * M, N)

        stft_c = torch.stft(
            wave_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        stft_ri = torch.view_as_real(stft_c)
        stft_ri = stft_ri.permute(0, 3, 2, 1)
        X = stft_ri.reshape(B, M * 2, stft_ri.size(2), stft_ri.size(3))

        return X

    def forward(self, wave: torch.Tensor):
        x = self.stft_encode(wave)
        print(x.size())
        s1, e1 = self.sd1(x)
        s2, e2 = self.sd2(e1)
        s3, e3 = self.sd3(e2)

        skips = [s1, s2, s3]
        return skips, e3
