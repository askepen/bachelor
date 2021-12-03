import numpy as np
import torch
from scipy import interpolate
from dataset import CompressedAudioDataset
from tqdm import tqdm


class BaselineAudioRegressor():
    """ Cubic B-sline regression model """

    def __init__(self):
        # interpolate.CubicSpline()
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        r = 6  # ratio between sample rates (i think)

        x = x.flatten()
        y_len = len(x)*r

        timesteps_x = np.arange(y_len, step=r)
        timesteps_y = np.arange(y_len)

        f = interpolate.splrep(timesteps_x, x)
        pred = interpolate.splev(timesteps_y, f)

        return torch.Tensor(pred).view(1, -1)


def train_baseline():
    transform = torch.nn.Sequential()
    dataset = CompressedAudioDataset(
        data_dir="../data", train=True, transform=transform)

    model = BaselineAudioRegressor()
    losses = []
    for (x, _), (y, _) in tqdm(dataset):
        x, y = x[0], y[0]  # Drop sr
        pred = model.predict(x)

        # Crop to smallest size
        min_len = min(len(pred), len(y))
        pred, y = pred[:, min_len], y[:, min_len]

        loss = torch.nn.MSELoss(reduction="sum")(pred, y)
        losses.append(loss)

    train_loss = np.mean(losses)
    print(f"{train_loss = }")


if __name__ == "__main__":
    train_baseline()
