# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Cloud Cover Detection Challenge dataset."""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_radiant_mlhub_dataset, extract_archive


# TODO: read geospatial information from stac.json files
class CloudCoverDetection(NonGeoDataset):
    """Cloud Cover Detection Challenge dataset.

    This training dataset was generated as part of a
    `crowdsourcing competition
    <https://www.drivendata.org/competitions/83/cloud-cover/>`_ on DrivenData.org, and
    later on was validated using a team of expert annotators. See
    `this website <https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1>`__
    for dataset details.

    The dataset consists of Sentinel-2 satellite imagery and corresponding cloudy
    labels stored as GeoTiffs. There are 22,728 chips in the training data,
    collected between 2018 and 2020.

    Each chip has:

    * 4 multi-spectral bands from Sentinel-2 L2A product. The four bands are
      [B02, B03, B04, B08] (refer to Sentinel-2 documentation for more
      information about the bands).
    * Label raster for the corresponding source tile representing a binary
      classification for if the pixel is a cloud or not.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.HFQ6M7

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.4
    """

    dataset_id = "ref_cloud_cover_detection_challenge_v1"

    image_meta = {
        "train": {
            "filename": "ref_cloud_cover_detection_challenge_v1_train_source.tar.gz",
            "md5": "32cfe38e313bcedc09dca3f0f9575eea",
        },
        "test": {
            "filename": "ref_cloud_cover_detection_challenge_v1_test_source.tar.gz",
            "md5": "6c67edae18716598d47298f24992db6c",
        },
    }

    target_meta = {
        "train": {
            "filename": "ref_cloud_cover_detection_challenge_v1_train_labels.tar.gz",
            "md5": "695dfb1034924c10fbb17f9293815671",
        },
        "test": {
            "filename": "ref_cloud_cover_detection_challenge_v1_test_labels.tar.gz",
            "md5": "ec2b42bb43e9a03a01ae096f9e09db9c",
        },
    }

    collection_names = {
        "train": [
            "ref_cloud_cover_detection_challenge_v1_train_source",
            "ref_cloud_cover_detection_challenge_v1_train_labels",
        ],
        "test": [
            "ref_cloud_cover_detection_challenge_v1_test_source",
            "ref_cloud_cover_detection_challenge_v1_test_labels",
        ],
    }

    band_names = ["B02", "B03", "B04", "B08"]

    RGB_BANDS = ["B04", "B03", "B02"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = band_names,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initiatlize a new Cloud Cover Detection Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        self._validate_bands(bands)
        self.bands = bands

        if download:
            self._download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.chip_paths = self._load_collections()

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            length of dataset in integer
        """
        return len(self.chip_paths)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Returns a sample from dataset.

        Args:
            Index: index to return

        Returns:
            data and label at given index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {"image": image, "mask": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load all source images for a chip.

        Args:
            index: position of the indexed chip

        Returns:
            a tensor of stacked source image data
        """
        source_asset_paths = self.chip_paths[index]["source"]
        images = []
        for path in source_asset_paths:
            with rasterio.open(path) as image_data:
                image_array = image_data.read(1).astype(np.int32)
                images.append(image_array)
        image_stack: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        image_tensor = torch.from_numpy(image_stack)
        return image_tensor

    def _load_target(self, index: int) -> Tensor:
        """Load label image for a chip.

        Args:
            index: position of the indexed chip

        Returns:
            a tensor of the label image data
        """
        label_asset_path = self.chip_paths[index]["target"][0]
        with rasterio.open(label_asset_path) as target_data:
            target_img = target_data.read(1).astype(np.int32)

        target_array: "np.typing.NDArray[np.int_]" = np.array(target_img)
        target_tensor = torch.from_numpy(target_array)
        return target_tensor

    @staticmethod
    def _read_json_data(object_path: str) -> Any:
        """Loads a JSON file.

        Args:
            object_path: string path to the JSON file

        Returns:
            json_data: JSON object / dictionary

        """
        with open(object_path) as read_contents:
            json_data = json.load(read_contents)
        return json_data

    def _load_items(self, item_json: str) -> Dict[str, List[str]]:
        """Loads the label item and corresponding source items.

        Args:
            item_json: a string path to the item JSON file on disk

        Returns:
            a dictionary with paths to the source and target TIF filenames
        """
        item_meta = {}

        label_data = self._read_json_data(item_json)
        label_asset_path = os.path.join(
            os.path.split(item_json)[0], label_data["assets"]["labels"]["href"]
        )
        item_meta["target"] = [label_asset_path]

        source_item_hrefs = []
        for link in label_data["links"]:
            if link["rel"] == "source":
                source_item_hrefs.append(
                    os.path.join(self.root, link["href"].replace("../../", ""))
                )

        source_item_hrefs = sorted(source_item_hrefs)
        source_item_paths = []

        for item_href in source_item_hrefs:
            source_item_path = os.path.split(item_href)[0]
            source_data = self._read_json_data(item_href)
            source_item_assets = []
            for asset_key, asset_value in source_data["assets"].items():
                if asset_key in self.bands:
                    source_item_assets.append(
                        os.path.join(source_item_path, asset_value["href"])
                    )
            source_item_assets = sorted(source_item_assets)
            for source_item_asset in source_item_assets:
                source_item_paths.append(source_item_asset)

        item_meta["source"] = source_item_paths
        return item_meta

    def _load_collections(self) -> List[Dict[str, Any]]:
        """Loads the paths to source and label assets for each collection.

        Returns:
            a dictionary with lists of filepaths to all assets for each chip/item

        Raises:
            RuntimeError if collection.json is not found in the uncompressed dataset
        """
        indexed_chips = []
        label_collection: List[str] = []
        for c in self.collection_names[self.split]:
            if "label" in c:
                label_collection.append(c)
        label_collection_path = os.path.join(self.root, label_collection[0])
        label_collection_json = os.path.join(label_collection_path, "collection.json")

        label_collection_item_hrefs = []
        for link in self._read_json_data(label_collection_json)["links"]:
            if link["rel"] == "item":
                label_collection_item_hrefs.append(link["href"])

        label_collection_item_hrefs = sorted(label_collection_item_hrefs)

        for label_href in label_collection_item_hrefs:
            label_json = os.path.join(label_collection_path, label_href)
            indexed_item = self._load_items(label_json)
            indexed_chips.append(indexed_item)

        return indexed_chips

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided tuple of bands to load

        Raises:
            ValueError: if an invalid band name is provided
        """
        for band in bands:
            if band not in self.band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        images: bool = check_integrity(
            os.path.join(self.root, self.image_meta[self.split]["filename"]),
            self.image_meta[self.split]["md5"] if self.checksum else None,
        )

        targets: bool = check_integrity(
            os.path.join(self.root, self.target_meta[self.split]["filename"]),
            self.target_meta[self.split]["md5"] if self.checksum else None,
        )

        return images and targets

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_radiant_mlhub_dataset(self.dataset_id, self.root, api_key)

        image_archive_path = os.path.join(
            self.root, self.image_meta[self.split]["filename"]
        )
        target_archive_path = os.path.join(
            self.root, self.target_meta[self.split]["filename"]
        )
        for fn in [image_archive_path, target_archive_path]:
            extract_archive(fn, self.root)

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
            time_step: time step at which to access image, beginning with 0
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if dataset does not contain an RGB band
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        if "prediction" in sample:
            prediction = sample["prediction"]
            n_cols = 3
        else:
            n_cols = 2

        image, mask = sample["image"] / 3000, sample["mask"]

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[1].imshow(mask)
        axs[1].axis("off")

        if "prediction" in sample:
            axs[2].imshow(prediction)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
