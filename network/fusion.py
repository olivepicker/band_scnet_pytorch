import torch
import torch.nn as nn

class CSAFusion(nn.Module):
    def __init__(
        self, 
        dim,
        num_heads=4,
        gate_kernel_size=2,
        gate_stride_size=2,
    ):
        super().__init__()
        self.cmhsa = CMHSA(dim, num_heads=num_heads)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.Conv2d(dim, dim)
        )
        self.glu = GLU(dim)
        
    def forward(self, x):
        x_cmhsa = self.cmhsa(x)
        x_gate = self.gate(x)
        x = torch.cat([x_cmhsa, x_gate], dim=-1)
        x = self.glu(x)

        return x
    
class CMHSA(nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        num_heads = 4,
    ):
        super().__init__()
        self.to_q = nn.Linear(
            nn.Conv2d(dim, dim),
            nn.PReLU(),
            nn.LayerNorm(dim)
        )
        self.to_k = nn.Linear(
            nn.Conv2d(dim, dim),
            nn.PReLU(),
            nn.LayerNorm(dim)
        )
        self.to_v = nn.Linear(
            nn.Conv2d(dim, dim),
            nn.PReLU(),
            nn.LayerNorm(dim)
        )

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Conv2d(),
            nn.PReLU(),
            nn.LayerNorm()
        )
        
    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(k), self.to_v(x)
        attn, _ = self.attn(q, k, v)
        x = self.head(attn)
        
        return x

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d()
        self.act = nn.Sigmoid()
        self.pwconv = nn.Conv2d(kernel_size=1)

    def forward(self, x):
        x_conv = self.conv(x)
        x_act = self.act(x_conv)
        x = torch.cat([x_conv, x_act], dim=-1)
        
        return self.pwconv(x)