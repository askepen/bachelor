import json

import IPython.display
import numpy as np
import seaborn as sns
import torch
from torch.functional import block_diag, norm
from torchaudio import transforms as T
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import torchaudio
import PIL


def Audio(audio: np.ndarray, rate: int, button_text: str = "Play"):
    """
    Use instead of IPython.display.Audio as a workaround for VS Code.

    ### Parameters
    - `audio` is an array with shape `(channels, samples)` or just `(samples,)` for mono.
    - `rate` is the sample rate.
    """

    if np.ndim(audio) == 1:
        channels = [audio.tolist()]
    else:
        channels = audio.tolist()

    return IPython.display.HTML(
        f""" 
            <script>
                if (!window.audioContext) {{
                    window.audioContext = new AudioContext();
                    window.playAudio = function(audioChannels, rate) {{
                        const buffer = audioContext.createBuffer(
                            audioChannels.length, 
                            audioChannels[0].length, 
                            rate
                        );
                        for (let [channel, data] of audioChannels.entries()) {{
                            buffer.copyToChannel(Float32Array.from(data), channel);
                        }}
                        const source = audioContext.createBufferSource();
                        source.buffer = buffer;
                        source.connect(audioContext.destination);
                        source.start();
                    }}
                }}
            </script>
            <button onclick="playAudio({json.dumps(channels)}, {rate})">{button_text}</button>
        """
    )


def plot_specgram(
    spec_tensor: torch.tensor,
    sample_rate: int,
    n_fft=2 ** 8,
    title="Spectrogram",
    n_yticks=12,
    ylim_freq=None,
    save_path=None,
    return_fig=False,
):
    """
    Plots a spectrogram given a tensor with shape [1, B, N, 2].
    The resulting plot will have shape [B, N].

    - B: Number of bins,
    - N: Number of samples

    # Returns:
    If return_fig=True a pyplot figure object will be returned. 
    It is the users responsibility to close the figure object, by 
    calling (`plt.close(fig)`) once it's no longer needed.
    """
    # spec_tensor = spec_tensor.detach().cpu()
    spec = np.abs(spec_tensor.squeeze().numpy()[:, :, 0]) + np.abs(
        spec_tensor.squeeze().numpy()[:, :, 1]
    )
    fig, ax = plt.subplots(1, 1, dpi=72, figsize=(12.5, 6))
    sns.heatmap(spec, norm=LogNorm(), ax=ax, cmap="gist_heat")

    # Set y-ticks to frequencies in Hz. Computed using the
    # implementation of librosa.fft_frequencies:
    # https://librosa.org/doc/latest/_modules/librosa/core/convert.html#fft_frequencies
    freqs = np.linspace(0, float(sample_rate) / 2,
                        1 + n_fft // 2, dtype=np.int)
    ytick_idx = np.linspace(0, len(freqs) - 1, n_yticks, dtype=np.int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(freqs[ytick_idx])

    # Set upper y-limit given a frequency in Hz
    if ylim_freq is None:
        ylim = None
    elif ylim_freq > freqs[-1]:
        ylim = ylim_freq / (freqs[-1] / len(freqs))
    else:
        ylim = (np.abs(freqs - ylim_freq)).argmin()
    ax.set_ylim(ylim)

    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel("Frequency [Hz]")

    if save_path is not None:
        plt.savefig(save_path, dpi=72)

    if return_fig:
        return fig
    else:
        plt.show(block=False)

    plt.close(fig)


def plot_specgram_from_waveform(
    waveform: torch.tensor, sample_rate: int, title: str = "Spectrogram"
):
    """Computes and plots a spctgrogram given a waveform"""
    spec_tensor = torch.stft(waveform, return_complex=False, n_fft=2 ** 8)
    plot_specgram(spec_tensor, sample_rate, title=title)


def play_audio(waveform, sample_rate, button_text="Play"):
    """Displays a play button, that plays the given waveform when pressed"""
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate, button_text=button_text))
    elif num_channels == 2:
        display(
            Audio((waveform[0], waveform[1]),
                  rate=sample_rate, button_text=button_text)
        )
    else:
        raise ValueError(
            "Waveform with more than 2 channels are not supported.")


def save_audio(waveform, sample_rate, path):
    """Saves the given waveform to a path"""
    # Sox needs a 2d tensor in order to save an audio clip
    waveform = waveform.unsqueeze(-2)
    torchaudio.save(path, waveform, sample_rate, format="wav")


def play_audio_from_spec(spec, sample_rate):
    """Computes a waveform from a spectrogram and displays a button to play it"""
    waveform_recovered = torch.istft(spec, n_fft=2 ** 8)
    play_audio(waveform_recovered, sample_rate)


def plot_waveform(
    waveform: torch.tensor, sample_rate: int, title="Waveform", xlim=None, ylim=None
):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1, figsize=(100, 6))
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
