import torch
import torch.nn as nn
import math

from einops import rearrange
from einops.layers.torch import Rearrange

def _rearrange_ln(channels):
    return nn.Sequential(
        Rearrange('b c t f -> b t f c'),
        nn.LayerNorm(channels),
        Rearrange('b t f c -> b c t f'),
    )

class CSAFusionT(nn.Module):
    def __init__(
        self, 
        dim,
        gate_channels,
        num_heads=4,
    ):
        super().__init__()
        self.cmhsa = CMHSA(dim, num_heads=num_heads)
        self.gate = LinearGate(gate_channels)
        self.glu = GLU(dim, dim)
        
    def forward(self, x):
        x_cmhsa = self.cmhsa(x)
        x_gate = self.gate(x)
        x = x_cmhsa * x_gate
        x = self.glu(x)

        return x

class LinearGate(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size_f=2,
        stride_f=3,
    ):
        super().__init__()
        padding_f = kernel_size_f // 2 
        self.down = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size_f),
            stride=(1, stride_f),
            padding=(0, padding_f),
            bias=False,
        )
        # F축 upsample: ConvTranspose2d
        self.up = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size_f),
            stride=(1, stride_f),
            padding=(0, padding_f),
            output_padding=(0, stride_f - 1),
            bias=False,
        )
        
        self.norm = _rearrange_ln(channels)
        self.freq_linear = None

    def _build_freq_linear(self, f_down: int):
        self.freq_linear = nn.Linear(f_down, f_down)

    def forward(self, x):
        B, C, T, F = x.shape

        y = self.down(x) 
        B, C, T, Fd = y.shape

        y = self.norm(y)
        y_flat = rearrange(y, 'b c t f -> (b c t) f')

        if self.freq_linear is None:
            self._build_freq_linear(Fd)

        y_gate = self.freq_linear(y_flat)
        y_gate = torch.sigmoid(y_gate)
        y_gate = rearrange(y_gate, '(b c t) f -> b c t f', b=B, c=C, t=T)
        y_gate = self.up(y_gate)

        return y_gate

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        in_channels = 128,
        num_freqs = 54,
        num_heads = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.num_freqs = num_freqs
        self.num_heads = num_heads

        self.Cd = math.ceil(self.dim / self.num_freqs)
        qk_out_dim = self.Cd * num_heads
        v_out_dim = self.in_channels // num_heads * num_heads
        self.q_proj = nn.Sequential(
            nn.Conv2d(in_channels, qk_out_dim, kernel_size=1, bias=False),
            nn.PReLU(qk_out_dim),
            _rearrange_ln(qk_out_dim),

        )

        self.k_proj = nn.Sequential(
            nn.Conv2d(in_channels, qk_out_dim, kernel_size=1, bias=False),
            nn.PReLU(qk_out_dim),
            _rearrange_ln(qk_out_dim),
        )

        self.v_proj = nn.Sequential(
            nn.Conv2d(in_channels, v_out_dim, kernel_size=1, bias=False),
            nn.PReLU(v_out_dim),
            _rearrange_ln(v_out_dim),
        )

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, T, F = x.size()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.q_proj(x)   # (B, H*Cd, T, F)
        k = self.k_proj(x)   # (B, H*Cd, T, F)
        v = self.v_proj(x)   # (B, H*C_per_head, T, F) ~= (B, Ci, T, F)

        H = self.num_heads
        Cd = self.Cd
        C_per_head = self.in_channels // self.num_heads

        q = rearrange(q, 'b (h cd) t f -> b h t (cd f)', h=H, cd=Cd)
        k = rearrange(k, 'b (h cd) t f -> b h t (cd f)', h=H, cd=Cd)
        v = rearrange(v, 'b (h c) t f -> b h t (c f)', h=H, c=C_per_head)

        print(q.size(), k.size(), v.size())
        # shapes:
        #   q: (B, H, T, Dq)   Dq = Cd*F
        #   k: (B, H, T, Dq)
        #   v: (B, H, T, Dv)   Dv = (Ci/H)*F

        Dq = q.size(-1)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dq)  # (B, H, T, T)
        attn_weights = attn_scores.softmax(dim=-1)                          # (B, H, T, T)

        out = torch.matmul(attn_weights, v)  # (B, H, T, Dv)

        out = rearrange(
            out,
            'b h t (c f) -> b (h c) t f',
            h=H,
            c=C_per_head,
            f=F,
        )  # (B, Ci, T, F)
        return out, q, k, v


class CMHSA(nn.Module):
    def __init__(
        self,
        dim,
        in_channels = 128,
        num_freqs = 54,
        num_heads = 4,
    ):
        super().__init__()
        self.attn = Attention(dim, in_channels, num_freqs, num_heads)
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.PReLU(),
            _rearrange_ln(dim)
        )
        
    def forward(self, x):
        attn, q, k, v = self.attn(x)
        x = self.head(attn)
        
        return x

class GLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2*out_channels, kernel_size=1)
        self.act = nn.Sigmoid()
        self.pwconv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_conv = self.conv(x)
        x_a, x_b = x_conv.chunk(2, dim=1)
        x = x_a * self.act(x_b)
        
        return self.pwconv(x)