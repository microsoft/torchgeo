# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Cloud Cover Detection Challenge dataset."""

import os, glob, json
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
from PIL import Image
from torch import Tensor

from .geo import VisionDataset
from .utils import check_integrity, download_radiant_mlhub_dataset, extract_archive


# TODO: read geospatial information from stac.json files
class CloudCoverDetection(VisionDataset):
    """Cloud Cover Detection Challenge dataset.

    This training dataset was generated as part of a 
    [crowdsourcing competition](https://www.drivendata.org/competitions/83/cloud-cover/) 
    on DrivenData.org, and later on was validated using a team of expert annotators. See 
    `this website<https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1>` for dataset details.

    The dataset consists of Sentinel-2 satellite imagery and corresponding cloudy labels stored as GeoTiffs. 
    There are 22,728 chips in the training data, collected between 2018 and 2020. 

    Each chip has:

    * 4 multi-spectral bands from Sentinel-2 L2A product. The four bands are [B02, B03, B04, B08] (refer to 
    Sentinel-2 documentation for more information about the bands).
    * Label raster for the corresponding source tile representing a binary classifcation 
    for if the pixel is is a cloud or not.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.HFQ6M7

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    datset_id = "ref_cloud_cover_detection_challenge_v1"

    image_meta = {
        "train": {
            "filename": "ref_cloud_cover_detection_challenge_v1_train_source.tar.gz",
            "md5": "32cfe38e313bcedc09dca3f0f9575eea"
        },
        "test": {
            "filename": "ref_cloud_cover_detection_challenge_v1_test_source.tar.gz",
            "md5": "6c67edae18716598d47298f24992db6c"
        }
    }

    target_meta = {
        "train": {
            "filename": "ref_cloud_cover_detection_challenge_v1_train_labels.tar.gz",
            "md5": "695dfb1034924c10fbb17f9293815671"
        },
        "test": {
            "filename": "ref_cloud_cover_detection_challenge_v1_test_labels.tar.gz",
            "md5": "ec2b42bb43e9a03a01ae096f9e09db9c"
        }
    }

    collection_names = {
        "train": [
            "ref_cloud_cover_detection_challenge_v1_train_source",
            "ref_cloud_cover_detection_challenge_v1_train_labels"
        ],
        "test": [
            "ref_cloud_cover_detection_challenge_v1_test_source",
            "ref_cloud_cover_detection_challenge_v1_test_labels"
        ]
    }

    band_names = [
        "B02",
        "B03",
        "B04",
        "B08"
    ]

    RGB_BANDS = ["B04", "B03", "B02"]

    def __init__(
        self,
        root: str = "data",
        chip_size: int = 512,
        split: str = "train",
        bands: Tuple[str, ...] = band_names,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initiatlize a new Cloud Cover Detection Dataset instance.

        Args:
            root: root directory where dataset can be found
            chip_size: size of chips
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
        self.chip_size = chip_size
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.checksum = checksum
        self.collection_glob = self._get_collection_glob()

        if download:
            self._download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.chip_paths = self._load_collections()

    def __len__(self):
        """Return the number of items in the dataset

        Returns:
            length of dataset in integer
        """
        return len(self.chip_paths)

    def __getitem__(self, index):
        """Returns a sample from dataset

        Args:
            Index: index to return

        Returns:
            data and label at given index
        """

        image = self._load_image(index)
        label = self._load_target(index)
        sample = Dict[str, Tensor] = {
            "image": image,
            "mask": label
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load all source images for a chip

        Args:
            index: position of the indexed chip

        Returns:
            a tensor of stacked source image data
        """

        source_asset_paths = self.chip_paths[index]['source']
        images = []
        for path in source_asset_paths:
            with rasterio.open(path) as image_data:
                image_array = image_data.read(
                    indexes=1,
                    # out_shape=self.chip_size,
                    out_type="int32",
                    # resampling=Resampling.bilinear
                )
                images.append(image_array)
        image_stack: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        image_tensor = torch.from_numpy(image_stack)
        return image_tensor

    def _load_target(self, index: int) -> Tensor:
        """Load label image for a chip

        Args:
            index: position of the indexed chip

        Returns:
            a tensor of the label image data
        """

        label_asset_path = self.chip_paths[index]['target']
        with Image.open(label_asset_path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            labels = torch.from_numpy(array)

    def _get_collection_glob(self):
        """Loads collection paths for the train/test split

        Returns:
            a list of paths to each collection.json file from the dataset
        """
        
        collection_glob = []
        for collection in self.collection_names[self.split]:
            collection_glob += glob.glob(f'{self.root}/{collection}/**/collection.json', recursive=True) 
        return collection_glob

    @staticmethod
    def _validate_stac_object(object_path: str) -> bool:
        """Validates a STAC object JSON filepath exists
        
        Returns:
            True if the object is a valid JSON file

        Raises:
            RuntimeError: if JSON or TIF file is not a valid object on disk
        """

        if not os.path.exists(object_path):
            raise RuntimeError(
                f"{object_path} not a valid STAC object path in the Catalog. "
                + "Make sure all files uncompressed completely."
            )
        return True

    @staticmethod
    def _read_json_data(object_path: str) -> Dict[str, any]:
        with open(object_path, 'r') as read_contents:
            json_data = json.load(read_contents)
        return json_data

    def _load_items(self, item_json: str) -> Dict[str, any]:
        """Loads the label item and corresponding source items 
        for a given spatially aligned chip.
        
        Args:
            item_json: a string path to the item JSON file on disk

        Returns:
            a dictionary with paths to the source and target TIF filenames
        """
        
        item_meta = {}
        
        if self._validate_stac_object(item_json):
            label_data = self._read_json_data(item_json)
            label_asset_path = os.path.join(os.path.split(item_json)[0], label_data['assets']['labels']['href'])
            item_meta['target'] = label_asset_path
            
            if self._validate_stac_object(label_asset_path):
                source_item_hrefs = [os.path.join(self.root, l['href'].replace('../../','')) for l in label_data['links'] if l['rel'] == 'source']
                source_item_hrefs = sorted(source_item_hrefs)
                source_item_paths = []
                
                for item_href in source_item_hrefs:
                    source_item_path = os.path.split(item_href)[0]
                    if self._validate_stac_object(item_href):
                        source_data = self._read_json_data(item_href)
                        source_item_assets = [os.path.join(source_item_path, asset_value['href']) for asset_key, asset_value in source_data['assets'].items() if asset_key in bands]
                        source_item_assets = sorted(source_item_assets)
                        for source_item_asset in source_item_assets:
                            if self._validate_stac_object(source_item_asset):
                                source_item_paths.append(source_item_asset)
                                
            item_meta['source'] = source_item_paths
        return item_meta

    def _load_collections(self) -> List[Dict[str, any]]:
        """Loads the paths to source and label assets for each collection

        Returns:
            a dictionary with lists of filepaths to all assets for each chip/item

        Raises:
            RuntimeError if collection.json is not found in the uncompressed dataset
        """

        indexed_chips = []
        label_collection = [c for c in self.collection_names[self.split] if c.__contains__('label')][0]
        label_collection_path = os.path.join(self.root, label_collection)
        label_collection_json = os.path.join(label_collection_path, 'collection.json')
        
        if self._validate_stac_object(label_collection_json):
            if not label_collection_json in self.collection_glob:
                raise RuntimeError(
                    f"{label_collection_json} not found in dataset directory"
                )

            label_collection_item_hrefs = [l['href'] for l in self._read_json_data(label_collection_json)['links'] if l['rel'] == 'item']
            label_collection_item_hrefs = sorted(label_collection_item_hrefs)
            
            for label_href in label_collection_item_hrefs:
                label_json = os.path.join(label_collection_path, label_href)
                indexed_item = self._load_items(label_json)
                indexed_chips.append(indexed_item)
                
        return indexed_chips

    def _validate_bands(self, bands: Tuple[str, ...]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided tuple of bands to load

        Raises:
            AssertionError: if ``bands`` is not a tuple
            ValueError: if an invalid band name is provided
        """
        assert isinstance(bands, tuple), "The list of bands must be a tuple"
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

        image_archive_path = os.path.join(self.root, self.image_meta[self.split]["filename"])
        target_archive_path = os.path.join(self.root, self.target_meta[self.split]["filename"])
        for fn in [image_archive_path, target_archive_path]:
            extract_archive(fn, self.root)