import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Union, Tuple, Sequence

from network.decoder import Decoder
from network.encoder import Encoder
from network.fusion import CSAFusion
from network.separation import SeparationNet

class BandSCNet(nn.Module):
    def __init__(
        self,
        dim_hidden,
        enc_sample_rate: int = 44100,
        enc_n_fft: int = 4096,
        enc_hop_length: int = 1024,
        enc_win_length: int = 4096,
        enc_in_channels: int = 2,
        dec_out_channels: int = 4,
        sep_dim_squeeze: int = 64, 
        sep_dim_ffn: int = 64, 
        sep_num_freqs: int = 56
    ):
        super().__init__()
        self.encoder = Encoder(
    
        )
        # self.fusion = CSAFusion()
        self.separation = SeparationNet(
            dim_hidden, sep_dim_squeeze, sep_dim_ffn, sep_num_freqs
        )
        self.decoder = Decoder(
            out_channels=dec_out_channels
        )
    
    def forward(self, x):
        skips, x = self.encoder(x)
        # self.fusion(x)
        x = self.separation(x)
        x = self.decoder(x)

        return x