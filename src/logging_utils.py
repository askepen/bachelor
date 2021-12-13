import torch
import wandb
import audio_utils


def get_wandb_image(x, sr, name):
    x, sr = x.detach()[0], sr.detach()[0]
    sr = sr.item()
    n_fft = sr // (2 ** 5)
    img = audio_utils.plot_specgram(
        x,
        sr,
        n_fft=n_fft,
        ylim_freq=None,
        n_yticks=13,
        title=name,
        return_pil=True,
    )
    return wandb.Image(img)


def log_image(y, pred, sr):
    images = [
        get_wandb_image(y, sr, "y"),
        get_wandb_image(pred, sr, "Prediction"),
    ]
    wandb.log({f"images": images})


def get_wandb_audio(x, sr):
    x, sr = x.detach()[0], sr.detach()[0].item()
    n_fft = sr // (2 ** 5)
    x = torch.view_as_complex(x.contiguous())
    waveform = torch.istft(x, n_fft)
    return wandb.Audio(waveform.numpy(), sr)


def log_audio(y, pred, sr):
    audios = [get_wandb_audio(y, sr), get_wandb_audio(pred, sr)]
    wandb.log({f"audio": audios})
