import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate

class BaselineAudioRegressor():
    """ Cubic B-sline regression model """
    def __init__(self):
        # interpolate.CubicSpline()
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor: 
        r = 6 # ratio between sample rates (i think)

        x = x.flatten()
        y_len = len(x)*r

        timesteps_x = np.arange(y_len, step=r)
        timesteps_y = np.arange(y_len)

        f = interpolate.splrep(timesteps_x, x)
        pred = interpolate.splev(timesteps_y, f)

        return torch.Tensor(pred).view(1,-1)
    