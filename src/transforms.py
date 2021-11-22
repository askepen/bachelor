import torch
import torch.nn.functional as F
from torch.nn import Module

class STFT(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        waveform, sample_rate = x
        n_fft = round(sample_rate / (2 ** 5))
        # n_fft = sample_rate // (2 ** 5)
        return torch.stft(waveform, return_complex=True, n_fft=n_fft)
        # return torch.stft(waveform, return_complex=False, n_fft=n_fft)

class OnlyWavefrom(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        """x should be a Tensor[waveform, sample_rate]"""
        waveform, _sample_rate = x
        return waveform

# class PadToSize(Module):    
#     def __init__(self, shape) -> None:
#         self.shape = shape
#         super().__init__()
    
#     def forward(self, x):
#         """ """
#         return F.pad(x, (self.shape , 0))
