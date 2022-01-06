import torch
from torch import nn


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        """Assumes inputs are non-negative"""
        return self.mse(torch.log(1 + pred), torch.log(1 + actual))


class ComplexMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.msle = MSLELoss()

    def forward(self, pred, actual):
        """Assumes inputs are complex tensors, viewed as real"""
        pred = torch.abs(torch.view_as_complex(pred))
        actual = torch.abs(torch.view_as_complex(actual))
        return self.msle(pred, actual)


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.msle = MSLELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.msle(pred, actual))
