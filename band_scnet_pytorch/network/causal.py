
import torch
from torch import nn
from torch.nn import functional as F

class CausalConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs.pop("padding", None)
        super().__init__(*args, padding=0, **kwargs)
        kh, kw = self.kernel_size
        self.pad_time_left = self.dilation[0] * (kh - 1)
        self.pad_freq_left = kw // 2
        self.pad_freq_right = (kw - 1) - self.pad_freq_left
        if self.stride[0] != 1:
            raise ValueError("CausalConv2d requires temporal stride 1")

    def forward(self, x):
        x = F.pad(x, (self.pad_freq_left, self.pad_freq_right,
                      self.pad_time_left, 0))
        return super().forward(x)

class FrameChannelNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        y = F.layer_norm(x.permute(0, 2, 3, 1),
                         (self.channels,), self.weight, self.bias, self.eps)
        return y.permute(0, 3, 1, 2).contiguous()
