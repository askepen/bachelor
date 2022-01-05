import torch
import torchaudio
import wandb
import audio_utils
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from torchaudio import transforms as T


def get_wandb_image(x, sr, name):
    sr = sr.item()
    n_fft = sr // (2 ** 5)
    # n_fft = 2048
    fig = audio_utils.plot_specgram(
        x,
        sr,
        n_fft=n_fft,
        ylim_freq=None,
        n_yticks=13,
        title=name,
        return_fig=True,
    )
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def get_wandb_audio(x, sr):
    sr = sr.item()
    n_fft = sr // (2 ** 5)
    # x = torchaudio.transforms.InverseMelScale(
    #     x.shape[0],
    #     sample_rate=sr,
    #     n_mels=128,
    # )(x)
    x = torch.view_as_complex(x.contiguous())
    waveform = x
    waveform = torch.istft(x, n_fft).detach().cpu()
    return wandb.Audio(waveform.numpy(), sr)


class ImagePredictionLogger(Callback):
    def __init__(self, n_samples, log_every_n_steps=10):
        super().__init__()
        self.n_samples = n_samples
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps != 0:
            return

        (x_batch, x_sr), (y_batch, y_sr) = batch
        x_batch = x_batch.detach().cpu()[:self.n_samples]
        y_batch = y_batch.detach().cpu()[:self.n_samples]
        y_sr = y_sr.detach().cpu()
        x_sr = x_sr.detach().cpu()

        pred_batch = pl_module(x_batch)
        pred_batch = pred_batch.detach().cpu()

        x_imgs, imgs, y_audio, pred_audio = zip(*[
            [
                get_wandb_image(x, x_sr, "Input"),
                get_wandb_image((y, pred), y_sr, "Actual / Prediction"),
                get_wandb_audio(y, y_sr),
                get_wandb_audio(pred, y_sr),
            ] for x, y, pred, x_sr, y_sr in zip(x_batch, y_batch, pred_batch, x_sr, y_sr)
        ])

        trainer.logger.experiment.log({
            "audio targets": y_audio,
            "image input": x_imgs,
            "image predictions": imgs,
            "audio predictions": pred_audio,
        }, commit=False)
