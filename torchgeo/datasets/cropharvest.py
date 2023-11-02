# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CropHarvest datasets."""

import glob
import os
from typing import Callable, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from matplotlib.figure import Figure
import h5py
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_and_extract_archive


class CropHarvest(NonGeoDataset):
    """CropHarvest dataset.

    The CropHarvest <https://github.com/nasaharvest/cropharvest> datataset is a
    classification dataset.

    Dataset features:

    * _ pixel single timeseries with croptype labels 
    * 18 bands per image over 12 months

    Dataset format:

    * images are 12x18 ndarrays with 18 bands over 12 months

    Dataset properties:

    1. is_crop - whether or not a single pixel contains cropland
    2. classification_label - optional field identifying a specific croptype
    3. dataset - source datset for the imagery
    4. lat
    5. lon

    

    If you use this dataset in your research, please cite the following paper:

    * https://openreview.net/forum?id=JtjzUXPEaCu
    """

    #*https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/bands.py
    all_bands = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9","B11", "B12", "temperature_2m", "total_precipitation", "elevation", "slope", "NDVI"]
    rgb_bands = ["B11", "B8", "B2"]

    file_dict = {
        "features": {
            "url": "https://zenodo.org/records/7257688/files/features.tar.gz?download=1",
            "filename": "features.tar.gz",
            "extracted_filename":os.path.join("features", "arrays")
        },
        "labels": {
            "url": "https://zenodo.org/records/7257688/files/labels.geojson?download=1",
            "filename": "labels.geojson",
            "extracted_filename":"labels.geojson"
        },
    }

    directory = "CropHarvest"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CropHarvest dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "val"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """

        self.root = root
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.files = self._load_features(self.root, self.directory)
        self.labels = self._load_labels(self.root, self.directory)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        data = self._load_array(files["chip"])
        
        label = self._load_label(files["index"], files["dataset"])
        sample = {"data": data, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_features(
        self, root: str, directory: str
    ) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            directory: sub directory CropHarvest

        Returns:
            list of dicts containing path for each of hd5 single pixel time series and its key for associated data 
        """
        files = []
        chips = glob.glob(os.path.join(root, directory, self.file_dict["features"]["extracted_filename"], "*.h5"))
        chips = sorted(os.path.basename(chip) for chip in chips)
        for chip in chips:
            chip_path = os.path.join(root, directory, self.file_dict["features"]["extracted_filename"], chip)
            index = int(chip.split("_")[0])
            dataset = chip.split("_")[1][:-3]
            files.append(dict(chip=chip_path, index=index, dataset=dataset))
        return files
    
    def _load_labels(
        self, root: str, directory: str
    ) -> pd.DataFrame:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset
            directory: sub directory CropHarvest

        Returns:
            pandas dataframe  
        """
        with open(os.path.join(root, directory, self.file_dict["labels"]["extracted_filename"]),encoding="utf8") as f:
            data = json.load(f)

            pd.json_normalize(data["features"])
            df = pd.json_normalize(data["features"])

            return df

    def _load_array(self, path: str) -> Tensor:
        """Load an individual single pixel timeseries.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with h5py.File(filename, 'r') as f:
            array = f.get('array')[()]
            tensor = torch.from_numpy(array).float()
            return tensor

    def _load_label(self, index: int, dataset: str) -> str:
        """Load the croptype label for a single pixel timeseries.

        Args:
            index: sample index in labels.geojson
            dataset: dataset name to query labels.geojson
        Returns:
            the croptype label, "Some" if no croptype is defined, or "None" if there are no crops
        """
        row = self.labels[(self.labels["properties.index"] == index) & (self.labels["properties.dataset"] == dataset)].to_dict(orient='records')[0]

        label = "None"      
        if row["properties.classification_label"]:
            label = row["properties.classification_label"]
        elif row["properties.is_crop"] == 1:
            label = "Some"

        return label
    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories are found, else False
        """
        for fileinfo in self.file_dict.values():
            filename = fileinfo["extracted_filename"]
            filepath = os.path.join(self.root, self.directory, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        for fileinfo in self.file_dict.values():
            download_and_extract_archive(
                fileinfo["url"],
                self.root,
                filename=os.path.join(self.directory,fileinfo["filename"]),
                md5=self.md5 if self.checksum else None,
            )

    def plot(
        self,
        sample: dict[str, Tensor],
        subtitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset using bands for Agriculture RGB composite.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional subtitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots()
        bands = [self.all_bands.index(band) for band in self.rgb_bands]
        rgb = sample["data"].numpy()[:,bands]
        normalized = rgb / np.max(rgb,axis=1, keepdims=True)
        axs.imshow(normalized[None, ...])
        axs.set_title(f'Croptype: {sample["label"]}')

        if subtitle is not None:
            plt.subtitle(subtitle)

        return fig