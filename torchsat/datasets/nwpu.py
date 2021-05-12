import os

import torch
from torchvision.utils import download_file_from_google_drive, check_integrity
from torchvision.vision import VisionDataset


class VHR10(VisionDataset):
    """`NWPU VHR-10 <https://doi.org/10.1016/j.isprsjprs.2014.10.002>`_ Dataset."""

    base_folder = "vhr10"
    meta = {
        "file_id": "1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE",
        "filename": "NWPU VHR-10 dataset.rar",
        "md5": "d30a7ff99d92123ebb0b3a14d9102081",
    }

    def __init__(self, root: str, download: bool = False) -> None:
        """Initialize a new VHR-10 dataset instance.

        Parameters:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
        """
        super().__init__(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5 matches, else False
        """
        return check_integrity(
            os.path.join(self.root, self.base_folder, self.meta["filename"])
        )

    def download(self) -> None:
        """Download the dataset from Google Drive and extract it."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_file_from_google_drive(
            self.meta["file_id"],
            os.path.join(self.root, self.base_folder),
            self.meta["filename"],
            self.meta["md5"],
        )

        # Must be installed to extract RAR file
        import rarfile

        with rarfile.RarFile(
            os.path.join(self.root, self.base_folder, self.meta["filename"])
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))
