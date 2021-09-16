import os
from urllib.request import urlretrieve
from zipfile import ZipFile
from joblib import Parallel, delayed
import torch
from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

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
    def __init__(self, test=False, force_download=False, force_generate=False) -> None:
        super().__init__()

        if test:
            self.data_url = CLEAN_TESTSET_URL
            self.data_path = os.path.join(DATA_DIR, "test")
        else:
            self.data_url = CLEAN_TRAINSET_56SPK_URL
            self.data_path = os.path.join(DATA_DIR, "train")

        self.data_path_wav = os.path.join(self.data_path, "wav")
        self.data_path_gsm = os.path.join(self.data_path, "gsm")

        self._get_wav_samples(force_download)
        self._get_gsm_samples(force_generate)

    def _get_wav_samples(self, force_download=False):

        if force_download and os.path.exists(self.data_path_wav):
            os.remove(self.data_path_wav)

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

        # Download and extract the dataset if we don't already have it
        if not (os.path.exists(self.data_path_wav)):
            with tqdm(
                unit="B", unit_scale=True, miniters=1, desc="Downloading dataset"
            ) as progress_bar:
                zip_path, _ = urlretrieve(
                    self.data_url, reporthook=reporthook(progress_bar)
                )
            zipfile = ZipFile(zip_path)
            zipfile.extractall(self.data_path)
            os.remove(zip_path)

            # Rename extracted folder to "wav". Please note that this will rename
            # the first folder it finds regardsless of what was extracted.
            folders = [
                dir for dir in os.listdir(self.data_path) if dir not in ["wav", "gsm"]
            ]
            os.rename(folders[0], "wav")

    def _get_gsm_samples(self, force_generate=False):
        if force_generate and os.path.exists(self.data_path_gsm):
            os.remove(self.data_path_gsm)

        # Generate gsm samples if we do not have them
        if not (os.path.exists(self.data_path_gsm)) or force_generate:
            os.mkdir(self.data_path_gsm)
            _ = Parallel(n_jobs=-1)(map(
                delayed(self._generate_gsm_sample),
                tqdm(os.listdir(self.data_path_wav), desc="Generating GSM samples"),
            ))

    def _generate_gsm_sample(self, file_name):
        speech, sample_rate = torchaudio.sox_effects.apply_effects_file(
            os.path.join(self.data_path_wav, file_name),
            effects=[
                ["remix", "1"],
                # ["lowpass", "4000"],
                [
                    "compand",
                    "0.02,0.05",  # attack, decay
                    "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",  # [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
                    "-8",
                    "-7",  # [gain [initial-volume-dB [delay]]]
                    "0.05",
                ],
                ["rate", "8000"],
            ],
        )

        # Save to folder with GSM codec
        gsm_path = os.path.join(self.data_path_gsm, file_name)
        torchaudio.save(gsm_path, speech, sample_rate, format="gsm")

    def __len__(self):
        dir = os.path.join(self.data_path_wav)
        return len(os.listdir(dir))

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        wav_paths = os.listdir(self.data_path_wav)
        file_name = wav_paths[index]
        gsm_path = os.path.join(self.data_path_gsm, file_name)
        return torchaudio.load(gsm_path, format="gsm")
