from typing_extensions import OrderedDict
import pytorch_lightning as pl
import torch
from torch import nn
import loss
from torchvision.transforms import CenterCrop


class LitFullyConnected(pl.LightningModule):
    def __init__(
        self,
        stft_height,
        lr,
        momentum,
        optim,
        n_fft,
        **kwargs,
    ):
        super().__init__()
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.n_fft = n_fft
        self.first_layers = nn.ModuleList([
            self.layer(stft_height*1, stft_height*2),
            self.layer(stft_height*2, stft_height*2),
            self.layer(stft_height*2, stft_height*2),
        ])
        self.last_layers = nn.ModuleList([
            self.layer(stft_height*4, stft_height*2),
            self.layer(stft_height*4, stft_height*2),
            self.layer(stft_height*4, stft_height*1, activation=False),
        ])
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = loss.MagnitudeMSELoss()
        self.save_hyperparameters()

    def layer(self, in_layers, out_layers, activation=True):
        if activation:
            return nn.Sequential(nn.Linear(in_layers, out_layers), nn.LeakyReLU())
        else:
            return nn.Sequential(nn.Linear(in_layers, out_layers))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--momentum", type=float, default=0.0,
            help="Only for SGD optimizer"
        )
        parser.add_argument("--optim", type=str, default="adam")
        return parent_parser

    def crop_width_height(self, x, shape_to_match):
        return CenterCrop(shape_to_match[-2:])(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device)

        # x = torch.stft(x, self.n_fft, return_complex=True)
        x = torch.view_as_complex(x)
        # out_size = x.shape[-2:]
        phase = torch.angle(x)
        x = torch.abs(x)

        def forward(x):
            x = x.squeeze(-1)
            skip = []
            for layer in self.first_layers:
                x = layer(x)
                skip.append(x.clone())

            for layer, x_skip in zip(self.last_layers, skip[::-1]):
                x = torch.cat([x, x_skip], dim=-1)
                x = layer(x)

            x = x.unsqueeze(-1)
            return x

        x = torch.cat([forward(chunk) for chunk in torch.split(x, 1, -1)], -1)

        x = torch.polar(x, phase)
        # x = torch.istft(x, self.n_fft)
        # x = self.crop_width_height(x, out_size)
        x = torch.view_as_real(x)
        return x

    def _step(self, batch, batch_idx, step_name):
        """Generic code to run for each step in train/val/test"""
        (x, _), (y, _) = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> None:
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self.optim == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise Exception(f"Optimizer '{self.optim}' not recognized")
