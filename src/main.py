from dataset import CompressedAudioDataset
from gsm_compression import simulate_phone_recording

def main():
    dataset = CompressedAudioDataset(test=True)
    simulate_phone_recording(dataset, 5)

if __name__ == "__main__":
    main()
