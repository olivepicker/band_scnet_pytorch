import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Union, Tuple, Sequence

from band_scnet_pytorch.decoder import Decoder
from band_scnet_pytorch.encoder import Encoder
from band_scnet_pytorch.fusion import CSAFusion
from band_scnet_pytorch.separation import CrossBandBlock, NarrowBandBlock

class BandSCNet(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = BandSCNetEncoder(

        )

        self.fusion = CSAFusion(

        )

        self.separation = 

        self.decoder = BandSCNetDecoder(

        )
    
    def forward(self, x):
        pass