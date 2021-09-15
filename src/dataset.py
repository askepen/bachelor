from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlretrieve
import os

from tqdm import tqdm
from torch.utils.data import Dataset

# Path to where the datasets will be stored
DATA_DIR = "../data"

# We are using the "Noisy speech database for training speech enhancement 
# algorithms and TTS models". Created by Cassia Valentini-Botinhao and 
# published by the University of Edinburgh. 
#
# More info here: https://datashare.ed.ac.uk/handle/10283/2791
CLEAN_TRAINSET_56SPK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_56spk_wav.zip?sequence=3&isAllowed=y"
CLEAN_TRAINSET_28SPK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y"
CLEAN_TESTSET_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y"

class CompressedAudioDataset(Dataset):
    def __init__(self, test=False, force_download=False) -> None:
        super().__init__()
        if test:
            self.data_path = self._get_dataset(CLEAN_TESTSET_URL, "test", force_download)
        else:
            self.data_path = self._get_dataset(CLEAN_TRAINSET_56SPK_URL, "train", force_download)
    
    def _get_dataset(self, url, path, force_download=False):
        data_path = os.path.join( DATA_DIR, path)

        # Updates a tqdm progressbar with current download status
        def reporthook(progress_bar: tqdm):
            last_b = [0]
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    progress_bar.total = tsize
                progress_bar.update((b - last_b[0]) * bsize)
                last_b[0] = b
                return 
            return update_to 
        
        if force_download and os.path.exists(data_path):
            os.remove(data_path)

        # Download and extract the dataset if we don't already have it
        if not(os.path.exists(data_path)) or force_download:
            zip_path = os.path.join(DATA_DIR, f"{path}.zip")
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=f"Downloading {path} dataset") as progress_bar:
                zip_path, _ = urlretrieve(url, reporthook=reporthook(progress_bar))
            zipfile = ZipFile(zip_path)
            zipfile.extractall(data_path)
            os.remove(zip_path)
        
        return data_path
    
    def __len__(self):
        raise NotImplemented

    def __getitem__(self, index):
        dir = os.path.join(self.data_path, "clean_testset_wav")
        paths = os.listdir(dir)
        return os.path.join(dir, paths[index])
    