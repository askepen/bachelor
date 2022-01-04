import torch
from torch import nn


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(
            torch.log(1 + pred - pred.min()),
            torch.log(1 + actual - actual.min())
        )
