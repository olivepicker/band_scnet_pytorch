import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Tuple, Sequence
from einops import rearrange
try:
    from .network.decoder import Decoder
    from .network.encoder import Encoder
    from .network.separation import SeparationNet
except ImportError:
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

        if enc_n_fft <= 0 or enc_hop_length <= 0 or enc_hop_length > enc_n_fft:
            raise ValueError("Require 0 < hop_length <= n_fft")
        if enc_win_length != enc_n_fft:
            raise ValueError(
                "center=False iSTFT requires full nonzero endpoint window; "
                "this causal reference implementation requires win_length == n_fft"
            )
        self.n_fft = enc_n_fft
        self.hop_length = enc_hop_length
        self.win_length = enc_win_length
        self.in_channels = enc_in_channels
        self.out_channels = dec_out_channels

        window = torch.ones(self.win_length)
        self.register_buffer("window", window, persistent=False)

        enc_dims = (2 * enc_in_channels, dim_hidden//4, dim_hidden//2, dim_hidden)
        dec_dims = (dim_hidden, dim_hidden//2, dim_hidden//4, 2 * dec_out_channels)

        self.encoder = Encoder(dims=enc_dims)
        self.decoder = Decoder(fusion_dim=fusion_dim, dims=dec_dims)
        self.separation = SeparationNet(dim_hidden, sep_dim_squeeze, sep_dim_ffn)

        frequencies = self.encoder.frequency_lengths(self.n_fft // 2 + 1)
        self.separation.crossband.fblm._build_freq_linear(frequencies[-1])
        for block, freq in zip(self.decoder.blocks, reversed(frequencies)):
            down_freq = freq // 2 + 1
            block.fusion.gate._build_freq_linear(down_freq)

    def stft_encode(self, wave: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, S, C, L = wave.shape
        device = wave.device

        num_frames = max(1, (max(L - self.n_fft, 0) + self.hop_length - 1)
                         // self.hop_length + 1)
        target_length = (num_frames - 1) * self.hop_length + self.n_fft
        padding = target_length - L
        if padding > 0:
            wave = F.pad(wave, (0, padding))
        L_pad = wave.shape[-1]

        wave = wave.reshape(B * S * C, L_pad)

        stft_c = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            #window=self.window,
            window=self.window,
            center=False,
            normalized=True,
            return_complex=True,
        ) 

        x = torch.view_as_real(stft_c)
        x = rearrange(x, '(b s c) fr t cp -> b s c fr t cp', b=B, s=S)
        
        return x, padding

    def istft_decode(self, x, padding=0):
        B, S, C, Fr, T, Cp = x.shape
        
        total_len = self.hop_length * (T - 1) + self.n_fft
        x = rearrange(x, 'b s c fr t cp -> (b s c) fr t cp')
        x_complex = torch.view_as_complex(x)    # (B*M, F, T)
        wave = torch.istft(
            x_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            #window=self.window,
            window=self.window,
            normalized=True,
            center=False,
            length=total_len,
        )

        wave = rearrange(wave, '(b s c) n -> b s c n', s = S, c = C)

        if padding > 0:
            wave = wave[..., :-padding]
            
        return wave

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        x, pad_x = self.stft_encode(x) # B S C Fr T Cp
        B, S, C, Fr, T, Cp = x.size()

        y_orig = None
        if y is not None:
            y_orig = y.clone()
            y, pad_y = self.stft_encode(y) # B S C Fr T Cp
         
        x = rearrange(x, 'b s c fr t cp -> b (s c cp) t fr')
        e, skips, sd_lengths_list, orig_lengths_list = self.encoder(x)

        e = self.separation(e)
        x_hat = self.decoder(e, skips, sd_lengths_list, orig_lengths_list)
        x_hat = rearrange(x_hat, 'b (s c cp) t fr -> b s c fr t cp', s=self.out_channels//C, c=C, cp=Cp)
        x_recon = self.istft_decode(x_hat, pad_x)

        return x_hat, y, x_recon, y_orig