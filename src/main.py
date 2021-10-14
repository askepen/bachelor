from typing import Tuple
from numpy.core.fromnumeric import transpose
from torch.utils.data import DataLoader
from dataset import CompressedAudioDataset
from audio_utils import *
from baseline import BaselineAudioRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics 
import torchaudio.transforms as T

def dataset_examples():
    dataset = CompressedAudioDataset(test=False)
    gsm, wav = dataset[120]

    plot_spec_and_play(gsm[0], gsm[1])
    plot_spec_and_play(wav[0], wav[1])


def plot_spec_and_play(waveform, sample_rate):
    n_fft=sample_rate//(2**5)

    spec = torch.stft(waveform, return_complex=False, n_fft=n_fft)
    waveform_recovered = torch.istft(spec, n_fft=n_fft)

    plot_specgram(spec, sample_rate, n_fft=n_fft, ylim_freq=None, n_yticks=13)
    play_audio(waveform, sample_rate)
    play_audio(waveform_recovered, sample_rate)


def train_baseline():
    transform = torch.nn.Sequential()
    dataset = CompressedAudioDataset(data_dir="../data", test=True, transform=transform)
    
    X, y = dataset.numpy()
    X = np.transpose(X)
    y = np.transpose(y)

    print(X.shape)
    print(y.shape)
    model = BaselineAudioRegressor()
    results = cross_val_score(model, X, y, cv=4)
    print(results)

def main():
    dataset_examples()
    #train_baseline()

if __name__ == "__main__":
    main()
