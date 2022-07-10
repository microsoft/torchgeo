# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""2022 IEEE GRSS Data Fusion Contest (DFC2022) dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib import colors
from rasterio.enums import Resampling
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive, percentile_normalization


class DFC2022(NonGeoDataset):
    """DFC2022 dataset.

    The `DFC2022 <https://www.grss-ieee.org/community/technical-committees/2022-ieee-grss-data-fusion-contest/>`__
    dataset is used as a benchmark dataset for the 2022 IEEE GRSS Data Fusion Contest
    and extends the MiniFrance dataset for semi-supervised semantic segmentation.
    The dataset consists of a train set containing labeled and unlabeled imagery and an
    unlabeled validation set. The dataset can be downloaded from the
    `IEEEDataPort DFC2022 website <https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022/>`_.

    Dataset features:

    * RGB aerial images at 0.5 m per pixel spatial resolution (~2,000x2,0000 px)
    * DEMs at 1 m per pixel spatial resolution (~1,000x1,0000 px)
    * Masks at 0.5 m per pixel spatial resolution (~2,000x2,0000 px)
    * 16 land use/land cover categories
    * Images collected from the
      `IGN BD ORTHO database <https://geoservices.ign.fr/documentation/donnees/ortho/bdortho/>`_
    * DEMs collected from the
      `IGN RGE ALTI database <https://geoservices.ign.fr/documentation/donnees/alti/rgealti/>`_
    * Labels collected from the
      `UrbanAtlas 2012 database <https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012/view/>`_
    * Data collected from 19 regions in France

    Dataset format:

    * images are three-channel geotiffs
    * DEMS are single-channel geotiffs
    * masks are single-channel geotiffs with the pixel values represent the class

    Dataset classes:

    0. No information
    1. Urban fabric
    2. Industrial, commercial, public, military, private and transport units
    3. Mine, dump and construction sites
    4. Artificial non-agricultural vegetated areas
    5. Arable land (annual crops)
    6. Permanent crops
    7. Pastures
    8. Complex and mixed cultivation patterns
    9. Orchards at the fringe of urban classes
    10. Forests
    11. Herbaceous vegetation associations
    12. Open spaces with little or no vegetation
    13. Wetlands
    14. Water
    15. Clouds and Shadows

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1007/s10994-020-05943-y

    .. versionadded:: 0.3
    """  # noqa: E501

    classes = [
        "No information",
        "Urban fabric",
        "Industrial, commercial, public, military, private and transport units",
        "Mine, dump and construction sites",
        "Artificial non-agricultural vegetated areas",
        "Arable land (annual crops)",
        "Permanent crops",
        "Pastures",
        "Complex and mixed cultivation patterns",
        "Orchards at the fringe of urban classes",
        "Forests",
        "Herbaceous vegetation associations",
        "Open spaces with little or no vegetation",
        "Wetlands",
        "Water",
        "Clouds and Shadows",
    ]
    colormap = [
        "#231F20",
        "#DB5F57",
        "#DB9757",
        "#DBD057",
        "#ADDB57",
        "#75DB57",
        "#7BC47B",
        "#58B158",
        "#D4F6D4",
        "#B0E2B0",
        "#008000",
        "#58B0A7",
        "#995D13",
        "#579BDB",
        "#0062FF",
        "#231F20",
    ]
    metadata = {
        "train": {
            "filename": "labeled_train.zip",
            "md5": "2e87d6a218e466dd0566797d7298c7a9",
            "directory": "labeled_train",
        },
        "train-unlabeled": {
            "filename": "unlabeled_train.zip",
            "md5": "1016d724bc494b8c50760ae56bb0585e",
            "directory": "unlabeled_train",
        },
        "val": {
            "filename": "val.zip",
            "md5": "6ddd9c0f89d8e74b94ea352d4002073f",
            "directory": "val",
        },
    }

    image_root = "BDORTHO"
    dem_root = "RGEALTI"
    target_root = "UrbanAtlas"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new DFC2022 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` is invalid
        """
        assert split in self.metadata
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._verify()

        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image = self._load_image(files["image"])
        dem = self._load_image(files["dem"], shape=image.shape[1:])
        image = torch.cat(tensors=[image, dem], dim=0)

        sample = {"image": image}

        if self.split == "train":
            mask = self._load_target(files["target"])
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image/dem/mask
        """
        directory = os.path.join(self.root, self.metadata[self.split]["directory"])
        images = glob.glob(
            os.path.join(directory, "**", self.image_root, "*.tif"), recursive=True
        )

        files = []
        for image in sorted(images):
            dem = image.replace(self.image_root, self.dem_root)
            dem = f"{os.path.splitext(dem)[0]}_RGEALTI.tif"

            if self.split == "train":
                target = image.replace(self.image_root, self.target_root)
                target = f"{os.path.splitext(target)[0]}_UA2012.tif"
                files.append(dict(image=image, dem=dem, target=target))
            else:
                files.append(dict(image=image, dem=dem))

        return files

    def _load_image(self, path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image
            shape: the (h, w) to resample the image to

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_dtype="int32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        # Check if the files already exist
        exists = []
        for split_info in self.metadata.values():
            exists.append(
                os.path.exists(os.path.join(self.root, split_info["directory"]))
            )

        if all(exists):
            return

        # Check if .zip files already exists (if so then extract)
        exists = []
        for split_info in self.metadata.values():
            filepath = os.path.join(self.root, split_info["filename"])
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, split_info["md5"]):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        raise RuntimeError(
            "Dataset not found in `root` directory, either specify a different"
            + " `root` directory or manually download the dataset to this directory."
        )

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
        ncols = 2
        image = sample["image"][:3]
        image = image.to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(self.colormap)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(mask, cmap=cmap, interpolation=None)
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(pred, cmap=cmap, interpolation=None)
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(pred, cmap=cmap, interpolation=None)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")

            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
