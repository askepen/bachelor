from typing_extensions import OrderedDict
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.transforms import CenterCrop


class LitFullyConnected(pl.LightningModule):
    def __init__(
        self,
        stft_width,
        stft_height,
        lr,
        momentum,
        num_blocks,
        optim,
        kernel_size,
        **kwargs,
    ):
        super().__init__()
        self.optim = optim
        self.lr = lr
        self.momentum = momentum
        self.real_layers = self.linear_layers(stft_height)
        self.imag_layers = self.linear_layers(stft_height)
        self.loss_fn = nn.MSELoss()
        self.save_hyperparameters()

    def linear_layers(self, stft_height):
        return nn.Sequential(
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
            nn.Linear(stft_height, stft_height), nn.LeakyReLU(),
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--momentum", type=float, default=0.0,
            help="Only for SGD optimizer"
        )
        parser.add_argument("--num_blocks", type=int, default=3)
        parser.add_argument("--optim", type=str, default="adam")
        parser.add_argument("--kernel_size", type=int, default=3)
        return parent_parser

    def crop_width_height(self, x, shape_to_match):
        return CenterCrop(shape_to_match[-2:])(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device)
        x = torch.view_as_complex(x)

        x_real, x_imag = [
            torch.cat([
                layer(chunk.squeeze(-1)).unsqueeze(-1)
                for chunk in torch.split(axes, 1, -1)
            ], -1)
            for axes, layer in [
                (x.real, self.real_layers),
                (x.imag, self.imag_layers),
            ]
        ]

        x = torch.complex(x_real, x_imag)
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
        """Returns loss from single batch"""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx) -> None:
        """Logs validation loss"""
        return self._step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        """Logs test loss"""
        return self._step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self.optim == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            raise Exception(f"Optimizer '{self.optim}' not recognized")
