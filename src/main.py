import torch

import audio_utils
import train
from dataset import CompressedAudioDataset


def main():
    # dataset_examples()
    train.train()


def dataset_examples():
    """
    Display a spectrogram and play the audio of a sample
    in the dataset. Both the compressed and uncompressed
    audio will be presented.
    """
    dataset = CompressedAudioDataset(train=True)
    gsm, wav = dataset[123]
    plot_spec_and_play(gsm[0], gsm[1])
    plot_spec_and_play(wav[0], wav[1])


def plot_spec_and_play(waveform, sample_rate, title="Spectrogram"):
    """
    Plot a spectrogram and and display play button for given waveform
    """
    n_fft = sample_rate // (2 ** 5)
    spec = torch.stft(waveform, return_complex=False, n_fft=n_fft)
    audio_utils.plot_specgram(
        spec, sample_rate, n_fft=n_fft,
        ylim_freq=None, n_yticks=13, title=title
    )
    audio_utils.play_audio(waveform, sample_rate, button_text=f"Play {title}")


if __name__ == "__main__":
    main()
