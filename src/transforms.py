import torch
import torch.nn.functional as F
from torch.nn import Module
import seaborn as sns
from matplotlib import pyplot as plt
import audio_utils


class STFT(Module):
    def __init__(self, return_sample_rate=False) -> None:
        self.return_sample_rate = return_sample_rate
        super().__init__()

    def forward(self, x):
        waveform, sample_rate = x
        n_fft = sample_rate // (2 ** 5)
        x = torch.stft(waveform, return_complex=True, n_fft=n_fft)
        if self.return_sample_rate:
            return x, sample_rate
        else:
            return x


class DropSampleRate(Module):
    """Given a tuple (x, sample_rate) it returns only x"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x, _sample_rate = x
        return x


class PadToSize(Module):
    def __init__(self, shape, has_sample_rate=False) -> None:
        self.shape = shape
        self.has_sample_rate = has_sample_rate
        super().__init__()

    def forward(self, x):
        if self.has_sample_rate:
            x, sample_rate = x
        shape_diff = [0, self.shape - x.shape[-1]]
        x = x.squeeze()
        x = F.pad(x, pad=shape_diff, value=0)
        if self.has_sample_rate:
            return x, sample_rate
        else:
            return x


class DisplayTensor(Module):
    """Plots a spectrogram. Input must be tuple of (stft, sample_rate)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        spec, sample_rate = x
        spec = torch.view_as_real(spec)
        audio_utils.plot_specgram(
            spec, sample_rate, n_fft=sample_rate // (2 ** 5)
        )
        plt.show()
        return x


class PlayWaveform(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        waveform, sample_rate = x
        audio_utils.play_audio(waveform, sample_rate)
        return x


class PrintShape(Module):
    """Prints the size of a tensor. Does not actuaclly transform the input."""

    def __init__(self, annotation="PrintShape()", has_sample_rate=False) -> None:
        self.annotation = annotation
        self.has_sample_rate = has_sample_rate
        super().__init__()

    def forward(self, x):
        tensor = x[0] if self.has_sample_rate else x
        print(f"{self.annotation}: {tensor.shape}")
        return x


class ViewAsReal(Module):
    """Represents complex type as [real, img]."""

    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, x):
        return torch.view_as_real(x)


class OnlyReal(Module):
    """Drops the imaginary part of complex tensor."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x = torch.view_as_complex(x)
        return x.real
