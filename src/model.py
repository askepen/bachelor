import pytorch_lightning as pl
import torch
from torch import nn


class LitModel(pl.LightningModule):
    def __init__(self, in_features: int, out_features: int, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.loss_fn = nn.MSELoss(reduction="sum")
        mid_features = 128
        self.layers = nn.Sequential(
            # nn.Linear(in_features, in_features),
            # nn.Linear(in_features, out_features),
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
            nn.Conv2d(mid_features, mid_features, kernel_size=3, padding=1),
            nn.Conv2d(mid_features, mid_features, kernel_size=3, padding=1),
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
        )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def _step(self, batch, batch_idx, step_name):
        """Generic code to run for each step in train/val/test"""
        (x, x_sr), (y, y_sr) = batch
        pred = self(x)  # Run model

        # print(f"{step_name} pred:\t{pred.shape}")
        # print(f"{step_name} y:   \t{y.shape}")

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
