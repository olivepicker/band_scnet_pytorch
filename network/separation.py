import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

class FConv(nn.Module):
    def __init__(self, in_channels, kernel_size=5, groups=8):
        super().__init__()
        self.blk = nn.Sequential(
            Rearrange('b c t f -> (b t) f c'),
            nn.LayerNorm(in_channels),
            Rearrange('bt f c -> bt c f'),
            nn.Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size=kernel_size,groups=groups, padding='same'),
            nn.PReLU(in_channels),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.blk(x)
        x = rearrange(x, '(b t) c f -> b c t f', b=B)
        
        return x
    
class FullBandLinearModule(nn.Module):
    def __init__(self, dim_hidden, dim_squeeze, num_freqs=56):
        super().__init__()
        self.dim_squeeze = dim_squeeze
        self.norm = nn.LayerNorm(dim_hidden)

        self.freq_linears = nn.ModuleList([
            nn.Linear(num_freqs, num_freqs) for _ in range(dim_squeeze)
        ])

        self.squeeze = nn.Sequential(
            nn.Linear(dim_hidden, dim_squeeze),
            nn.SiLU()
        )
        self.unsqueeze = nn.Sequential(
            nn.Linear(dim_squeeze, dim_hidden),
            nn.SiLU()
        )

    def forward(self, x): # x: b c t f
        B, C, T, F = x.size()
        x = rearrange(x, 'b c t f -> (b t) f c')
        x = self.norm(x)

        x = self.squeeze(x)
        x = rearrange(x, 'bt f c -> bt c f')

        outs = []
        for i, layer in enumerate(self.freq_linears):
            yi = layer(x[:,i,:]).unsqueeze(1)
            outs.append(yi)

        x = torch.cat(outs, dim=1)

        x = rearrange(x, 'bt c f -> bt f c')
        x = self.unsqueeze(x)
        x = rearrange(x, '(b t) f c -> b c t f', t = T)

        return x

class CrossBandBlock(nn.Module):
    def __init__(self, dim_hidden, dim_squeeze, num_freqs):
        super().__init__()
        self.fconv0 = FConv(dim_hidden, kernel_size=3)
        self.fblm = FullBandLinearModule(dim_hidden, dim_squeeze, num_freqs)
        self.fconv1 = FConv(dim_hidden, kernel_size=3)

    def forward(self, x):
        B, C, T, F = x.size()
        x_f0 = x + self.fconv0(x)
        x_fblm = x_f0 + self.fblm(x_f0)
        x_f1 = x_fblm + self.fconv1(x_fblm)

        return x_f1

class MHSA(nn.Module):
    def __init__(self, dim_hidden, num_heads=4, drop_rate=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim_hidden)
        self.mhsa = nn.MultiheadAttention(embed_dim = dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, attn_mask=None):
        B = x.size(0)
        x = rearrange(x, 'b c t f -> (b t) f c')
        x = self.norm(x)
        x, attn = self.mhsa(x, x, x, attn_mask=attn_mask)
        x = self.dropout(x)
        x = rearrange(x, '(b t) f c -> b c t f', b=B)

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
        B = x.size(0)

        x = rearrange(x, 'b c t f -> (b t) f c')
        x = self.norm0(x)
        x = self.blk0(x)
        x = rearrange(x, 'bt f c -> bt c f')
        x = self.blk1(x)
        x = rearrange(x, 'bt c f -> bt f c')
        x = self.blk2(x)
        x = self.dropout(x)
        x = rearrange(x, '(b t) f c -> b c t f', b=B)

        return x

class NarrowBandBlock(nn.Module):
    def __init__(self, dim_hidden, dim_ffn, num_heads=4):
        super().__init__()
        self.mhsa = MHSA(dim_hidden=dim_hidden, num_heads=num_heads)
        self.ffn = TConvFFN(dim_hidden=dim_hidden, dim_ffn=dim_ffn)
    
    def forward(self, x):
        B = x.size(0)

        x_mhsa, attn = self.mhsa(x)
        x_mhsa = x + x_mhsa
        
        x_ffn = self.ffn(x_mhsa)
        x_ffn = x_mhsa + x_ffn

        return x
    
class SeparationNet(nn.Module):
    def __init__(
        self, 
        dim_hidden, 
        dim_squeeze, 
        dim_ffn, 
        num_freqs
    ):
        super().__init__()
        self.crossband = CrossBandBlock(
            dim_hidden=dim_hidden,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs
        )
        self.narrowband = NarrowBandBlock(
            dim_hidden=dim_hidden,
            dim_ffn=dim_ffn
        )
    
    def forward(self, x):
        x = self.crossband(x)
        x = self.narrowband(x)

        return x