import torch
import torchaudio
from dataset import CompressedAudioDataset
from audio_utils import *
import math

def _get_sample(dataset: CompressedAudioDataset, resample=None, index=0):
    effects = [
        ["remix", "1"]
    ]
    if resample:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(dataset[index], effects=effects)   

def simulate_phone_recording(dataset, index=0):
    sample_rate = 48000
    speech, _ = _get_sample(dataset, resample=sample_rate, index=index)

    plot_specgram(speech, sample_rate, title="Original")
    play_audio(speech, sample_rate)

    # Apply RIR
    # rir, _ = get_rir_sample(resample=sample_rate, processed=True)
    # speech_ = torch.nn.functional.pad(speech, (rir.shape[1]-1, 0))
    # speech = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

    # plot_specgram(speech, sample_rate, title="RIR Applied")
    # play_audio(speech, sample_rate)

    # Add background noise
    # Because the noise is recorded in the actual environment, we consider that
    # the noise contains the acoustic feature of the environment. Therefore, we add
    # the noise after RIR application.
    # noise, _ = get_noise_sample(resample=sample_rate)
    # noise = noise[:, :speech.shape[1]]

    # snr_db = 8
    # scale = math.exp(snr_db / 10) * noise.norm(p=2) / speech.norm(p=2)
    # speech = (scale * speech + noise) / 2

    # plot_specgram(speech, sample_rate, title="BG noise added")
    # play_audio(speech, sample_rate)

    # Apply filtering and change sample rate
    speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        speech,
        sample_rate,
        effects=[
            ["lowpass", "4000"],
            ["compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
            ["rate", "8000"],
        ],
    )

    plot_specgram(speech, sample_rate, title="Phöne")
    play_audio(speech, sample_rate)
