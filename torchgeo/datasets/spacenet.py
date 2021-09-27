# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datasets."""

import abc
import glob
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import fiona
import numpy as np
import rasterio as rio
import torch
from affine import Affine
from rasterio.features import rasterize
from torch import Tensor

from torchgeo.datasets.geo import VisionDataset
from torchgeo.datasets.utils import (
    check_integrity,
    download_radiant_mlhub_collection,
    extract_archive,
)


class SpaceNet(VisionDataset, abc.ABC):
    """Abstract base class for the SpaceNet datasets.

    The `SpaceNet <https://spacenet.ai/datasets/>`_ datasets are a set of
    datasets that all together contain >11M building footprints and ~20,000 km
    of road labels mapped over high-resolution satellite imagery obtained from
    Worldview-2 and Worldview-3 sensors.
    """

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """Dataset ID."""

    @property
    @abc.abstractmethod
    def imagery(self) -> Dict[str, str]:
        """Mapping of image identifier and filename."""

    @property
    @abc.abstractmethod
    def label_glob(self) -> str:
        """Label filename."""

    @property
    @abc.abstractmethod
    def collection_md5_dict(self) -> Dict[str, str]:
        """Mapping of collection id and md5 checksum."""

    def __init__(
        self,
        root: str,
        image: str,
        collections: List[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection
            collections: collection selection
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        self.root = root
        self.image = image  # For testing

        if not collections:
            for collection in collections:
                assert collection in self.collection_md5_dict

        self.collections = collections or list(self.collection_md5_dict.keys())
        self.filename = self.imagery[image]
        self.transforms = transforms
        self.checksum = checksum

        to_be_downloaded = self._check_integrity()

        if to_be_downloaded:
            if not download:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it."
                )
            else:
                self._download(to_be_downloaded, api_key)

        self.files = self._load_files(root)

    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        for collection in self.collections:
            images = glob.glob(os.path.join(root, collection, "*", self.filename))
            images = sorted(images)
            for imgpath in images:
                lbl_path = os.path.join(
                    os.path.dirname(imgpath) + "-labels", self.label_glob
                )
                files.append({"image_path": imgpath, "label_path": lbl_path})
        return files

    def _load_image(self, path: str) -> Tuple[Tensor, Affine]:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with rio.open(filename) as img:
            array = img.read().astype(np.float32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor, img.transform

    def _load_mask(self, path: str, tfm: Affine, shape: Tuple[int, int]) -> Tensor:
        """Rasterizes the dataset's labels (in geojson format).

        Args:
            path (str): path to the label
            tfm (Affine): transform of corresponding image
            shape (List[int, int]): shape of corresponding image

        Returns:
            Tensor: label tensor
        """
        with fiona.open(path) as src:
            labels = [feature["geometry"] for feature in src]

        if not labels:
            mask_data = np.zeros(shape=shape)
        else:
            mask_data = rasterize(
                labels,
                out_shape=shape,
                fill=0,  # nodata value
                transform=tfm,
                all_touched=False,
                dtype=np.uint8,
            )

        mask: Tensor = torch.from_numpy(mask_data).long()  # type: ignore[attr-defined]

        return mask

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        img, tfm = self._load_image(files["image_path"])
        h, w = img.shape[1:]
        mask = self._load_mask(files["label_path"], tfm, (h, w))

        sample = {"image": img, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> List[str]:
        """Checks the integrity of the dataset structure.

        Returns:
            List of collections to be downloaded
        """
        # Check if collections exist
        missing_collections = []
        for collection in self.collections:
            stacpath = os.path.join(self.root, collection, "collection.json")

            if not os.path.exists(stacpath):
                missing_collections.append(collection)

        if not missing_collections:
            return []

        to_be_downloaded = []
        for collection in missing_collections:
            archive_path = os.path.join(self.root, collection + ".tar.gz")
            if os.path.exists(archive_path):
                print(f"Found {collection} archive")
                if (
                    self.checksum
                    and check_integrity(
                        archive_path, self.collection_md5_dict[collection]
                    )
                    or not self.checksum
                ):
                    print("Extracting...")
                    extract_archive(archive_path)
                else:
                    print(f"Collection {collection} is corrupted")
                    to_be_downloaded.append(collection)
            else:
                print(f"{collection} not found")
                to_be_downloaded.append(collection)

        return to_be_downloaded

    def _download(self, collections: List[str], api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            collections: Collections to be downloaded
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        for collection in collections:
            download_radiant_mlhub_collection(collection, self.root, api_key)
            archive_path = os.path.join(self.root, collection + ".tar.gz")
            if (
                not self.checksum
                or not check_integrity(
                    archive_path, self.collection_md5_dict[collection]
                )
            ) and self.checksum:
                raise RuntimeError(f"Collection {collection} corrupted")

            print("Extracting...")
            extract_archive(archive_path)


class SpaceNet1(SpaceNet):
    """SpaceNet 1: Building Detection v1 Dataset.

    `SpaceNet 1 <https://spacenet.ai/spacenet-buildings-dataset-v1/>`_
    is a dataset of building footprints over the city of Rio de Janeiro.

    * *Dataset features*:

        * No. of images: 6940 (8 Band) + 6940 (RGB)
        * No. of polygons: 382,534 building labels
        * Area Coverage: 2544 sq km
        * GSD: 1 m (8 band),  50 cm (rgb)
        * Chip size: 102 x 110 (8 band), 407 x 439 (rgb)

    .. note::
       Chip size of both imagery can have 1 pixel difference

    * *Dataset format*:

        * Imagery - Worldview-2 GeoTIFFs
            * 8Band.tif (Multispectral)
            * RGB.tif (Pansharpened RGB)
        * Labels - GeoJSON
            * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    """

    dataset_id = "spacenet1"
    imagery = {"rgb": "RGB.tif", "8band": "8Band.tif"}
    label_glob = "labels.geojson"
    collection_md5_dict = {"sn1_AOI_1_RIO": "e6ea35331636fa0c036c04b3d1cbf226"}

    def __init__(
        self,
        root: str,
        image: str = "rgb",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 1 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be "rgb" or "8band"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        collections = ["sn1_AOI_1_RIO"]
        assert image in ["rgb", "8band"]
        super().__init__(
            root, image, collections, transforms, download, api_key, checksum
        )


class SpaceNet2(SpaceNet):
    r"""SpaceNet 2: Building Detection v2 Dataset.

    `SpaceNet 2 <https://spacenet.ai/spacenet-buildings-dataset-v2/>`_
    is a dataset of building footprints over the cities of Las Vegas,
    Paris, Shanghai and Khartoum.

    - *Collection features*


    +------------+---------------------+------------+------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Buildings|
    +============+=====================+============+============+
    | Las Vegas  |    216              |   3850     |  151,367   |
    +------------+---------------------+------------+------------+
    | Paris      |    1030             |   1148     |  23,816    |
    +------------+---------------------+------------+------------+
    | Shanghai   |    1000             |   4582     |  92,015    |
    +------------+---------------------+------------+------------+
    | Khartoum   |    765              |   1012     |  35,503    |
    +------------+---------------------+------------+------------+

    - *Imagery features*

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - MS
            - PS-MS
            - PS-RGB
        *   - GSD (m)
            - 0.31
            - 1.24
            - 0.30
            - 0.30
        *   - Chip size (px)
            - 650 x 650
            - 162 x 162
            - 650 x 650
            - 650 x 650

    .. note::
       Chip size of MS images can have 1 pixel difference


    - *Dataset format*

        * Imagery - Worldview-3 GeoTIFFs
            * PAN.tif (Panchromatic)
            * MS.tif (Multispectral)
            * PS-MS (Pansharpened Multispectral)
            * PS-RGB (Pansharpened RGB)
        * Labels - GeoJSON
            * label.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    """

    dataset_id = "spacenet2"
    collection_md5_dict = {
        "sn2_AOI_2_Vegas": "cdc5df70920adca870a9fd0dfc4cca26",
        "sn2_AOI_3_Paris": "8299186b7bbfb9a256d515bad1b7f146",
        "sn2_AOI_4_Shanghai": "4e3e80f2f437faca10ca2e6e6df0ef99",
        "sn2_AOI_5_Khartoum": "8070ff9050f94cd9f0efe9417205d7c3",
    }

    imagery = {
        "MS": "MS.tif",
        "PAN": "PAN.tif",
        "PS-MS": "PS-MS.tif",
        "PS-RGB": "PS-RGB.tif",
    }
    label_glob = "label.geojson"

    def __init__(
        self,
        root: str,
        image: str = "PS-RGB",
        collections: List[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 2 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["MS", "PAN", "PS-MS", "PS-RGB"]
            collections: collection selection which must be a subset of:
                         [sn2_AOI_2_Vegas, sn2_AOI_3_Paris, sn2_AOI_4_Shanghai,
                         sn2_AOI_5_Khartoum]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """
        assert image in ["MS", "PAN", "PS-MS", "PS-RGB"]
        super().__init__(
            root, image, collections, transforms, download, api_key, checksum
        )

    # TODO: Remove this once radiantearth/radiant-mlhub#65 is fixed
    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        pat = re.compile("img1" + re.escape(os.sep))
        for collection in self.collections:
            images = glob.glob(os.path.join(root, collection, "*", self.filename))
            images = sorted(images)
            for imgpath in images:
                if collection == "sn2_AOI_2_Vegas" and pat.search(imgpath):
                    lbl_path = os.path.join(
                        os.path.dirname(os.path.dirname(imgpath)),
                        "_common",
                        "labels.geojson",
                    )
                else:
                    lbl_path = os.path.join(
                        os.path.dirname(imgpath) + "-labels", self.label_glob
                    )
                files.append({"image_path": imgpath, "label_path": lbl_path})
        return files
