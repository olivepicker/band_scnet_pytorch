import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

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
        self.fconv0 = FConv(dim_hidden, kernel_size=3)
        self.fblm = FullBandLinearModule(dim_hidden, dim_squeeze, num_freqs)
        self.fconv1 = FConv(dim_hidden, kernel_size=3)

    def forward(self, x):
        B, C, H, F = x.size()
        f0 = self.fconv0(x) + x
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
    def __init__(self, dim_hidden, dim_ffn, kernel_size=5, groups=8, drop_rate=0.):
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
            nn.GroupNorm(num_groups=groups, num_channels=dim_ffn),
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
        
        x_mhsa, attn = self.mhsa(x)
        x_mhsa = x + x_mhsa
        
        x_ffn = self.ffn(x_mhsa)
        x_ffn = x_mhsa + x_ffn
        x = rearrange(x_ffn, '(b h) f c -> b c h f', b=B)

        return x
    
#class SeparationNet(nn.Module):