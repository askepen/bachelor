from torch.utils import data
from dataset import CompressedAudioDataset
from audio_utils import *


def main():
    dataset = CompressedAudioDataset(test=False)

    gsm, wav = dataset[5]
    
    play_audio(gsm[0], gsm[1])
    play_audio(wav[0], wav[1])

if __name__ == "__main__":
    main()
