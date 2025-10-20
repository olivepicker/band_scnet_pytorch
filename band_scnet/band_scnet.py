import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange

# Encoder / Decoder
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_modules=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(num_modules):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=kernel_size),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
        
    def forward(self, x):
        for layers in self.layers:
            x = layers(x)
        
        return x

class SDLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=kernel_size, stride = stride)
    def forward(self, x):
        
        return self.conv(x)

class SDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=[]):
        super().__init__()
        self.low_freq_block = nn.Sequential(
            SDLayer(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size, stride=(1, strides[0])),
            nn.GELU(),
            ConvModule(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size, num_modules=3)
        )
        self.mid_freq_block = nn.Sequential(
            SDLayer(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size, stride=(1, strides[1])),
            nn.GELU(),
            ConvModule(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size, num_modules=2)
        )
        self.high_freq_block = nn.Sequential(
            SDLayer(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size, stride=(1, strides[2])),
            nn.GELU(),
            ConvModule(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size, num_modules=1)
        )

        self.last_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=kernel_size)

    def forward(self, x):
        B, C, H, W = x.size()

        ls = int(W * 0.175)
        ms = int(W * (0.175 + 0.392))

        x_low = x[..., :ls]     # (B, C, H, 0:ls)
        x_mid = x[..., ls:ms]  # (B, C, H, ls:ms)
        x_high = x[..., ms:]   # (B, C, H, ms:)

        l = self.low_freq_block(x_low)
        m = self.mid_freq_block(x_mid)
        h = self.high_freq_block(x_high)
    
        s = torch.concat([l, m, h], dim=3)
        e = self.last_conv(s)

        return s, e

# Separation Network
class FConv(nn.Module):
    def __init__(self, in_channels, kernel_size=5, groups=8):
        super().__init__()
        self.blk = nn.Sequential(
            Rearrange('b c h f -> b h f c'),
            nn.LayerNorm(in_channels),
            Rearrange('b h f c -> b c (h f)'),
            nn.Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size,groups=groups, padding='same'),
            nn.PReLU(in_channels)
        )
    def forward(self, x):
        
        return self.blk(x)
    
class FullBandLinearModule(nn.Module):
    def __init__(self, dim_hidden, dim_squeeze, num_freqs=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim_hidden)
        self.squeeze = nn.Sequential(
            nn.Linear(dim_hidden, dim_squeeze),
            nn.SiLU()
        )

        self.full = nn.Linear(num_freqs, num_freqs)
        self.unsqueeze = nn.Sequential(
            nn.Linear(dim_squeeze, dim_hidden),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.squeeze(x)
        x = rearrange(x, 'b c f -> b f c')
        x = self.full(x)
        x = rearrange(x, 'b f c -> b c f')
        x = self.unsqueeze(x)

        return x

class CrossBandBlock(nn.Module):
    def __init__(self, dim_hidden, dim_squeeze, num_freqs):
        super().__init__()
        self.fconv0 = FConv(dim_hidden)
        self.fblm = FullBandLinearModule(dim_hidden, dim_squeeze, num_freqs)
        self.fconv1 = FConv(dim_hidden)

    def forward(self, x):
        B, C, H, F = x.size()
        f0 = self.fconv0(x)
        f0 = rearrange(f0, 'b c (h f) -> (b h) f c', b=B, h=H)

        fblm = self.fblm(f0) + f0
        fblm = rearrange(fblm, '(b h) f c -> b c h f', b=B)
        
        f1 = self.fconv1(fblm)  
        f1 = rearrange(f1, 'b c (h f) -> b c h f', h=H) + fblm

        return f1
        
class MHSA(nn.Module):
    def __init__(self, dim_hidden, num_heads=4, drop_rate=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim_hidden)
        self.mhsa = nn.MultiheadAttention(embed_dim = dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, attn_mask=None):
        x = self.norm(x)
        x, attn = self.mhsa(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x)

        return x, attn

class TConvFFN(nn.Module):
    def __init__(self, dim_hidden, dim_ffn, kernel_size=3, groups=8, drop_rate=0.):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim_hidden)
        self.blk0 = nn.Sequential(
            nn.Linear(in_features=dim_hidden, out_features=dim_ffn),
            nn.SiLU()
        )
        self.blk1 = nn.Sequential(
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, groups=groups, kernel_size=kernel_size, padding='same'),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, groups=groups, kernel_size=kernel_size, padding='same'),
            nn.GroupNorm(num_groups=8, num_channels=dim_ffn),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, groups=groups, kernel_size=kernel_size, padding='same'),
            nn.SiLU(),
        )
        self.blk2 = nn.Sequential(
            nn.Linear(in_features = dim_ffn, out_features = dim_hidden),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        x = self.norm0(x)
        x = self.blk0(x)
        x = rearrange(x, 'b f c -> b c f')
        x = self.blk1(x)
        x = rearrange(x, 'b c f -> b f c')
        x = self.blk2(x)
        x = self.dropout(x)

        return x

class NarrowBandBlock(nn.Module):
    def __init__(self, dim_hidden, dim_ffn, num_heads=4):
        super().__init__()
        self.mhsa = MHSA(dim_hidden=dim_hidden, num_heads=num_heads)
        self.ffn = TConvFFN(dim_hidden=dim_hidden, dim_ffn=dim_ffn)
    
    def forward(self, x):
        B = x.size(0)
        x = rearrange(x, 'b c h f -> (b h) f c')
        x, attn = self.mhsa(x)
        x = self.ffn(x)
        x = rearrange(x, '(b h) f c -> b c h f', b=B)
        return x        
