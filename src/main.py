from torch.utils import data
from dataset import CompressedAudioDataset
from audio_utils import *

def main():
    dataset = CompressedAudioDataset(test=True)
    speech, sample_rate = dataset[5]
    play_audio(speech, sample_rate)

if __name__ == "__main__":
    main()

