import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Sequence

from network.decoder import Decoder
from network.encoder import Encoder
from network.separation import SeparationNet

class BandSCNet(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        enc_n_fft: int = 4096,
        enc_hop_length: int = 1024,
        enc_win_length: int = 4096,
        enc_in_channels: int = 2,
        fusion_dim: int = 1024,
        dec_out_channels: int = 2,
        sep_dim_squeeze: int = 64,
        sep_dim_ffn: int = 64,
    ):
        super().__init__()

        self.n_fft = enc_n_fft
        self.hop_length = enc_hop_length
        self.win_length = enc_win_length
        self.in_channels = enc_in_channels
        self.out_channels = dec_out_channels

        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

        c0 = 2 * enc_in_channels
        dims = (c0, dim_hidden//4, dim_hidden//2, dim_hidden)

        self.encoder = Encoder(dims=dims)
        self.decoder = Decoder(fusion_dim=fusion_dim, dims=dims)
        self.separation = SeparationNet(dim_hidden, sep_dim_squeeze, sep_dim_ffn)

    def stft_encode(self, wave: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, M, L = wave.shape
        device = wave.device

        padding = (self.hop_length - (L % self.hop_length)) % self.hop_length
        if padding > 0:
            wave = F.pad(wave, (0, padding))  # (B, M, L+padding)
        L_pad = wave.shape[-1]

        window = self.window.to(device)
        wave = wave.reshape(B * M, L_pad)

        stft_c = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )  # (B*M, F, T)

        stft_ri = torch.view_as_real(stft_c)        # (B*M, F, T, 2)
        stft_ri = stft_ri.permute(0, 3, 2, 1)      # (B*M, 2, T, F)
        x = stft_ri.reshape(B, M * 2, stft_ri.size(2), stft_ri.size(3))  # (B, 2M, T, F)

        return x, padding

    # def istft_decode(self, x: torch.Tensor, padding: int, length: int) -> torch.Tensor:
    #     B, C, T, Freq = x.shape
    #     M = self.in_channels
    #     assert C == 2 * M, f"expected 2*{M} channels, got {C}"

    #     x = x.reshape(B * M, 2, T, Freq)          # (B*M, 2, T, F)
    #     x = x.permute(0, 3, 2, 1).contiguous()           # (B*M, F, T, 2)
    #     x_complex = torch.view_as_complex(x)     # (B*M, F, T)

    #     window = self.window.to(x.device)
    #     wave = torch.istft(
    #         x_complex,
    #         n_fft=self.n_fft,
    #         hop_length=self.hop_length,
    #         win_length=self.win_length,
    #         window=window,
    #         length=length + padding,
    #     )  # (B*M, L_pad)

    #     wave = wave.view(B, M, -1)
    #     if padding > 0:
    #         wave = wave[..., :-padding]
    #     return wave

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, L = x.shape
        x, padding = self.stft_encode(x)
        e, skips, sd_lengths_list, orig_lengths_list = self.encoder(x)

        e = self.separation(e)
        y = self.decoder(e, skips, sd_lengths_list, orig_lengths_list)
        #y = self.istft_decode(y, padding=padding, length=L)
        return y
