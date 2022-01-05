import torch
import torchaudio
import wandb
import audio_utils
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from torchaudio import transforms as T


def get_wandb_image(x, sr, name, n_fft=None):
    sr = sr.item()
    n_fft = n_fft or sr // (2 ** 5)
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


def get_wandb_audio(x, sr, n_fft=None):
    sr = sr.item()
    n_fft = n_fft or sr // (2 ** 5)
    x = torch.view_as_complex(x.contiguous())

    waveform = torch.istft(x, n_fft).detach().cpu()
    return wandb.Audio(waveform.numpy(), sr)


class ImagePredictionLogger(Callback):
    def __init__(self, n_samples, log_every_n_steps=10, n_fft=None):
        super().__init__()
        self.n_samples = n_samples
        self.log_every_n_steps = log_every_n_steps
        self.n_fft = n_fft

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
                get_wandb_image(x, y_sr, "Input"),
                get_wandb_image((y, pred), y_sr, "Actual / Prediction"),
                get_wandb_audio(y, y_sr, self.n_fft),
                get_wandb_audio(pred, y_sr, self.n_fft),
            ] for x, y, pred, x_sr, y_sr in zip(x_batch, y_batch, pred_batch, x_sr, y_sr)
        ])

        trainer.logger.experiment.log({
            "audio targets": y_audio,
            "image input": x_imgs,
            "image predictions": imgs,
            "audio predictions": pred_audio,
        }, commit=False)
