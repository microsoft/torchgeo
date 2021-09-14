"""Spacenet1 dataset."""

import glob
import os
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
    download_radiant_mlhub,
    extract_archive,
)


class Spacenet1(VisionDataset):
    """Spacenet 1: Building Detection v1 Dataset.

    `Spacenet 1 <https://spacenet.ai/spacenet-buildings-dataset-v1/>`_
    is a dataset of building footprints over the city of Rio de Janeiro.

    Dataset features:

    * No. of images - 6940 (8 Band) + 6940 (RGB)
    * No. of polygons - 382,534 building labels
    * Area Coverage - 2544 sq km

    Dataset format:

    * Imagery - Raw 8 band Worldview-3 (GeoTIFF) & Pansharpened RGB image (GeoTIFF)
    * Labels - GeoJSON

    If you are using data from SpaceNet in a paper, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    """

    dataset_id = "spacenet1"
    md5 = "e6ea35331636fa0c036c04b3d1cbf226"
    filename_glob = "RGB.tif"  # To prevent reading .tif.aux.xml
    raw_8band_glob = "8Band.tif"
    label_glob = "labels.geojson"
    foldername = "sn1_AOI_1_RIO"

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialise a new Spacenet 1 Dataset instance

        Args:
            root: root directory where dataset can be found
            crs (Optional[CRS], optional): [description]. Defaults to None.
            res (float, optional): [description]. Defaults to None.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing
        """

        self.root = root
        self.transforms = transforms
        self.checksum = checksum

        if not self._check_integrity():
            if download:
                self._download(api_key)
            else:
                raise RuntimeError(
                    "Dataset not found. You can use download=True to download it."
                )

        self.files = self._load_files(os.path.join(root, self.foldername))

    def _load_files(self, root: str) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each triple of rgb,
            8band and label
        """
        files = []
        images = glob.glob(os.path.join(root, "*", self.filename_glob))
        images = sorted(images)
        for imgpath in images:
            rawpath = imgpath.replace("RGB.tif", "8Band.tif")
            lbl_path = os.path.join(
                os.path.dirname(rawpath) + "-labels", "labels.geojson"
            )
            files.append({"rgb": imgpath, "8band": rawpath, "label": lbl_path})
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

    def _load_label(self, path: str, tfm: Affine, shape: Tuple[int, int]) -> Tensor:
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
        rgb, tfm = self._load_image(files["rgb"])
        raw, _ = self._load_image(files["8band"])
        h, w = rgb.shape[1:]
        label = self._load_label(files["label"], tfm, (h, w))

        sample = {"rgb": rgb, "8band": raw, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> bool:
        """Checks the integrity of the dataset structure.
        Returns:
            True if the dataset directories are found, else False
        """
        stacpath = os.path.join(self.root, self.foldername, "collection.json")

        if os.path.exists(stacpath):
            return True

        # If dataset folder does not exist, check for uncorrupted archive
        archive_path = os.path.join(self.root, self.foldername + ".tar.gz")
        if not os.path.exists(archive_path):
            return False
        print("Archive found")
        if self.checksum and not check_integrity(archive_path, self.md5):
            print("Dataset corrupted")
            return False
        print("Extracting...")
        extract_archive(archive_path)
        return True

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """

        if self._check_integrity():
            print("Files already downloaded")
            return

        download_radiant_mlhub(self.dataset_id, self.root, api_key)
        archive_path = os.path.join(self.root, self.foldername + ".tar.gz")
        if (
            self.checksum
            and check_integrity(archive_path, self.md5)
            or not self.checksum
        ):
            extract_archive(archive_path)
        else:
            raise RuntimeError("Dataset corrupted")
