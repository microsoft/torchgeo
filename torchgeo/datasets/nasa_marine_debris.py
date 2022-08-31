# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Marine Debris dataset."""

import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

from .geo import NonGeoDataset
from .utils import download_radiant_mlhub_dataset, extract_archive


class NASAMarineDebris(NonGeoDataset):
    """NASA Marine Debris dataset.

    The `NASA Marine Debris <https://mlhub.earth/data/nasa_marine_debris>`__
    dataset is a dataset for detection of floating marine debris in satellite imagery.

    Dataset features:

    * 707 patches with 3 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 1 object class: marine_debris
    * images taken by Planet Labs PlanetScope satellites
    * imagery taken from 2016-2019 from coasts of Greece, Honduras, and Ghana

    Dataset format:

    * images are three-channel geotiffs in uint8 format
    * labels are numpy files (.npy) containing bounding box (xyxy) coordinates
    * additional: images in jpg format and labels in geojson format

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/rdnt.9r6ekg

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.2
    """

    dataset_id = "nasa_marine_debris"
    directories = ["nasa_marine_debris_source", "nasa_marine_debris_labels"]
    filenames = ["nasa_marine_debris_source.tar.gz", "nasa_marine_debris_labels.tar.gz"]
    md5s = ["fe8698d1e68b3f24f0b86b04419a797d", "d8084f5a72778349e07ac90ec1e1d990"]
    class_label = "marine_debris"

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize a new NASA Marine Debris Dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            verbose: if True, print messages when new tiles are loaded
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.api_key = api_key
        self.checksum = checksum
        self.verbose = verbose
        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        image = self._load_image(self.files[index]["image"])
        boxes = self._load_target(self.files[index]["target"])
        sample = {"image": image, "boxes": boxes}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array = f.read()
        tensor = torch.from_numpy(array)
        return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target bounding boxes for a single image.

        Args:
            path: path to the labels

        Returns:
            the target boxes
        """
        array = np.load(path)
        # boxes contain unecessary value of 1 after xyxy coords
        array = array[:, :4]
        tensor = torch.from_numpy(array)
        return tensor

    def _load_files(self) -> List[Dict[str, str]]:
        """Load a image and label files.

        Returns:
            list of dicts containing image and label files
        """
        image_root = os.path.join(self.root, self.directories[0])
        target_root = os.path.join(self.root, self.directories[1])
        image_folders = sorted(
            f for f in os.listdir(image_root) if not f.endswith("json")
        )

        files = []
        for folder in image_folders:
            files.append(
                {
                    "image": os.path.join(image_root, folder, "image_geotiff.tif"),
                    "target": os.path.join(
                        target_root,
                        folder.replace("source", "labels"),
                        "pixel_bounds.npy",
                    ),
                }
            )
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        exists = [
            os.path.exists(os.path.join(self.root, directory))
            for directory in self.directories
        ]
        if all(exists):
            return

        # Check if zip file already exists (if so then extract)
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # TODO: need a checksum check in here post downloading
        # Download and extract the dataset
        download_radiant_mlhub_dataset(self.dataset_id, self.root, self.api_key)
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            extract_archive(filepath)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 1

        image = draw_bounding_boxes(image=sample["image"], boxes=sample["boxes"])
        image = image.permute((1, 2, 0)).numpy()

        if "prediction_boxes" in sample:
            ncols += 1
            preds = draw_bounding_boxes(
                image=sample["image"], boxes=sample["prediction_boxes"]
            )
            preds = preds.permute((1, 2, 0)).numpy()

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols < 2:
            axs.imshow(image)
            axs.axis("off")
            if show_titles:
                axs.set_title("Ground Truth")
        else:
            axs[0].imshow(image)
            axs[0].axis("off")
            axs[1].imshow(preds)
            axs[1].axis("off")

            if show_titles:
                axs[0].set_title("Ground Truth")
                axs[1].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
