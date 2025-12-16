import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.eps = eps

    def forward(self, inputs, targets):
        mse_real = self.mse_loss(inputs[..., 0], targets[..., 0])
        mse_imag = self.mse_loss(inputs[..., 1], targets[..., 1])

        rmse_loss = torch.sqrt(mse_real + mse_imag + self.eps)

        return rmse_loss