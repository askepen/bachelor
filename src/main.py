from dataset import CompressedAudioDataset
from gsm_compression import simulate_phone_recording

if __name__ == "__main__":
    dataset = CompressedAudioDataset()
    simulate_phone_recording(dataset)
