import torch
import torch.nn.functional as F
from torch.nn import Module
from matplotlib import pyplot as plt
from torchaudio import transforms as T
import torchaudio
from math import exp, ceil
from baseline import BaselineAudioRegressor


class STFT(Module):
    def __init__(self, n_fft, return_sample_rate=False) -> None:
        self.n_fft = n_fft
        self.return_sample_rate = return_sample_rate
        super().__init__()

    def forward(self, x):
        waveform, sample_rate = x
        n_fft = self.n_fft or sample_rate // (2 ** 5)
        x = torch.stft(waveform, return_complex=True, n_fft=n_fft)
        if self.return_sample_rate:
            return x, sample_rate
        else:
            return x


class BSpline(Module):
    def __init__(self, sr_thresh) -> None:
        self.bspline = BaselineAudioRegressor()
        self.sr_thresh = sr_thresh
        super().__init__()

    def forward(self, x: torch.Tensor):
        waveform, sr = x
        if sr < self.sr_thresh:
            waveform = self.bspline.predict(waveform)
        return waveform, sr


class MelSpectrogram(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        x, sample_rate = x
        # n_fft = sample_rate // (2 ** 6)
        n_fft = 2048
        x = torchaudio.transforms.MelSpectrogram(
            sample_rate,
            n_fft,
            n_mels=128,
        )(x)
        return x


class DropSampleRate(Module):
    """Given a tuple (x, sample_rate) it returns only x"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x, _sample_rate = x
        return x


class EnsureChannel(Module):
    """Given a tuple (x, sample_rate) it returns only x"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        return x


class PadToSize(Module):
    def __init__(self, shape, has_sample_rate=False) -> None:
        self.shape = shape
        self.has_sample_rate = has_sample_rate
        super().__init__()

    def forward(self, x):
        if self.has_sample_rate:
            x, sample_rate = x

        y_diff = self.shape[-1] - x.shape[-1]
        x_diff = self.shape[-2] - x.shape[-2]

        x = x.squeeze()
        x = F.pad(x, pad=[0, y_diff, 0, x_diff], value=0.0)

        if self.has_sample_rate:
            return x, sample_rate
        else:
            return x


class RepeatToSize(Module):
    def __init__(self, shape, has_sample_rate=False) -> None:
        self.shape = shape
        super().__init__()

    def forward(self, x):
        x = x.squeeze()

        phase, magnitude = torch.angle(x), torch.abs(x)

        ratio = ceil(self.shape[0] / x.shape[0])
        magnitude = torch.cat([magnitude*exp(-i) for i in range(ratio)])

        y_diff = self.shape[-1] - magnitude.shape[-1]
        x_diff = self.shape[-2] - magnitude.shape[-2]
        magnitude = F.pad(magnitude, pad=[0, y_diff, 0, x_diff], value=0.0)

        y_diff = self.shape[-1] - phase.shape[-1]
        x_diff = self.shape[-2] - phase.shape[-2]
        phase = F.pad(phase, pad=[0, y_diff, 0, x_diff], value=0.0)

        x = torch.polar(magnitude, phase)

        return x


class RandomSubsample(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, waveform):
        waveform, sample_rate = waveform
        length = waveform.shape[-1]
        offset = torch.randint(int(length * 0.2), int(length * 0.8), (1,))
        waveform = waveform[:, offset:]
        return waveform, sample_rate


class Trim(Module):
    def __init__(self, left=0.0, right=0.0) -> None:
        super().__init__()
        self.left = left
        self.right = 1.0 - right

    def forward(self, waveform):
        waveform, sample_rate = waveform
        length = waveform.shape[-1]
        idx_left, idx_right = int(length * self.left), int(length * self.right)
        waveform = waveform[:, idx_left:idx_right]
        return waveform, sample_rate


class ViewAsReal(Module):
    """Represents complex type as [real, img]."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.view_as_real(x)


class NormailizeAndLog(Module):
    """Decibels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = torch.linalg.norm(x, dim=-1)
        x = torch.log(x)
        return x


class OnlyReal(Module):
    """Drops the imaginary part of complex tensor."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x = torch.view_as_complex(x)
        return x.real
