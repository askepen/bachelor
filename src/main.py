from typing import Tuple
from numpy.core.fromnumeric import transpose
from torch.utils.data import DataLoader
from dataset import CompressedAudioDataset
from audio_utils import *
from baseline import BaselineAudioRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from train import train

import torchaudio.transforms as T


def main():
    # dataset_examples()
    # train_baseline()
    train()


def dataset_examples():
    dataset = CompressedAudioDataset(train=True)
    gsm, wav = dataset[420]
    plot_spec_and_play(gsm[0], gsm[1])
    plot_spec_and_play(wav[0], wav[1])


def plot_spec_and_play(waveform, sample_rate, title="Spectrogram"):
    n_fft = sample_rate // (2 ** 5)
    spec = torch.stft(waveform, return_complex=False, n_fft=n_fft)
    plot_specgram(spec, sample_rate, n_fft=n_fft,
                  ylim_freq=None, n_yticks=13, title=title)
    play_audio(waveform, sample_rate, button_text=f"Play {title}")
    # waveform_recovered = torch.istft(spec, n_fft=n_fft)
    # play_audio(waveform_recovered, sample_rate)


def train_baseline():
    transform = torch.nn.Sequential()
    dataset = CompressedAudioDataset(
        data_dir="../data", test=True, transform=transform)

    model = BaselineAudioRegressor()
    # i = 0
    # for (x, x_sr), (y, y_sr) in dataset:
    #     pred = model.predict(x)

    #     print(f"{i = }")
    #     plot_spec_and_play(x, x_sr, title=f"Low res {i}")
    #     plot_spec_and_play(pred, y_sr, title=f"Predicted {i}")
    #     plot_spec_and_play(y, y_sr, title=f"Actual {i}")

    #     i += 1
    #     if i > 3:
    #         break
    (X, y), _ = dataset.numpy()
    #results = cross_val_score(model, X, y, cv=4)
    # print(results)


if __name__ == "__main__":
    main()
