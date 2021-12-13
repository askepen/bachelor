import os
from typing import Tuple, List
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np

import torch
import torchaudio
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm


# Links to the base dataset
CLEAN_TRAINSET_56SPK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_56spk_wav.zip?sequence=3&isAllowed=y"
CLEAN_TRAINSET_28SPK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y"
CLEAN_TESTSET_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y"


class CompressedAudioDataset(Dataset):
    """
    A dataset consisting of pairs of compressed and uncompressed speech samples.
    The compression used is GSM compression in order to mimic a phone line a
    closely as possible.

    For the base dataset, we are using the "Noisy speech database for training
    speech enhancement algorithms and TTS models" created by Cassia Valentini-Botinhao
    and published by the University of Edinburgh.

    More info about the base dataset here: https://datashare.ed.ac.uk/handle/10283/2791
    """

    def __init__(
        self,
        data_dir: str = "../data",
        train: bool = True,
        force_download: bool = False,
        force_generate: bool = False,
        transform: torch.nn.Module = None,
    ) -> None:
        super().__init__()

        self.transform = transform

        if train:
            self.data_url = CLEAN_TRAINSET_56SPK_URL
            self.data_path = os.path.join(data_dir, "train")
        else:
            self.data_url = CLEAN_TESTSET_URL
            self.data_path = os.path.join(data_dir, "test")

        self.data_path_wav = os.path.join(self.data_path, "wav")
        self.data_path_gsm = os.path.join(self.data_path, "gsm")

        self._get_wav_samples(force_download)
        self._get_gsm_samples(force_generate)

    def _get_wav_samples(self, force_download=False):
        """
        Downloads and extracts the base dataset into a fodler named `wav` in the data directory.
        ### Parameters
        - `force_download`: If `True` it will download everything from source and extract it
                            into the `wav` folder, remove everything previously the `wav` folder.
        """

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

    def _get_gsm_samples(self, force_generate=False, n_jobs=-1):
        """
        Makes a compressed version of every file in the `wav`-folder and puts
        the in a new folder named `gsm` in the data directory.
        ### Parameters
        - `force_generate`: If `True` it will generate new compressed versions
                            of every file in the `wav` folder, removing any
                            previous files in the `gsm`-folder before.
        - `n_jobs`: Amount of processes to use when compressing audio files
        """
        if force_generate and os.path.exists(self.data_path_gsm):
            os.remove(self.data_path_gsm)

        # Generate gsm samples if we do not have them
        if not (os.path.exists(self.data_path_gsm)) or force_generate:
            os.mkdir(self.data_path_gsm)
            _ = Parallel(n_jobs=n_jobs)(
                map(
                    delayed(self._generate_gsm_sample),
                    tqdm(os.listdir(self.data_path_wav), desc="Generating GSM samples"),
                )
            )

    def _generate_gsm_sample(self, file_name):
        """
        Apply GSM compression to a file from the `wav`-folder and save it to the `gsm`-folder
        ### Parameters
        - `file_name`: name of the file in the `wav`-folder to apply compression to
        """
        speech, sample_rate = torchaudio.sox_effects.apply_effects_file(
            os.path.join(self.data_path_wav, file_name),
            effects=[
                ["remix", "1"],
                # ["lowpass", "4000"],
                [
                    "compand",
                    "0.02,0.05",  # attack, decay
                    # [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2}
                    "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
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

    def _wav_filenames(self):
        return os.listdir(self.data_path_wav)

    def _filename_pairs(self) -> List[Tuple[str, str]]:
        """
        Returns a list of tuples `(gsm_path, wav_path)` containing pairs of paths
        to compressed and uncompressed matching audio samples
        """
        return [
            (
                os.path.join(self.data_path_gsm, file_name),
                os.path.join(self.data_path_wav, file_name),
            )
            for file_name in self._wav_filenames()
        ]

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, int]]:
        """
        ### Returns
        `((gsm_tensor, gsm_sr), (wav_tensor, wav_sr))`:
        A tuple of the compressed speech audio + sample rate and the
        corresponding target audio + sample rate
        """
        gsm_path, wav_path = self._filename_pairs()[index]

        gsm_tensor, gsm_sr = torchaudio.load(gsm_path, format="gsm")
        wav_tensor, wav_sr = torchaudio.load(wav_path, format="wav")

        if self.transform is not None:
            # gsm_tensor = self.transform.forward(gsm_tensor)
            # wav_tensor = self.transform.forward(wav_tensor)
            gsm_tensor = self.transform.forward((gsm_tensor, gsm_sr))
            wav_tensor = self.transform.forward((wav_tensor, wav_sr))

        return (gsm_tensor, gsm_sr), (wav_tensor, wav_sr)

    def numpy(self) -> np.ndarray:
        return np.transpose(
            [
                ((gsm[0].numpy(), gsm[1]), (wav[0].numpy(), wav[1]))
                for (gsm, wav) in tqdm(self, desc="Loading dataset into memory")
            ]
        )
