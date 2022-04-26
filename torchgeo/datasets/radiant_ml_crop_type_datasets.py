# Copyright (c) Microsoft Corporation. All rights reserved.>
# Licensed under the MIT License.

"""Radiant MLHub Crop Type Datasets."""

import abc
import glob
import json
import os
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from torch import Tensor

from .geo import GeoDataset
from .utils import (
    BoundingBox,
    check_integrity,
    disambiguate_timestamp,
    download_radiant_mlhub_dataset,
    extract_archive,
    percentile_normalization,
)


class CropTypeDatasetRadiantML(GeoDataset, abc.ABC):
    """Abstract Base Class for Radiant ML Crop Type datasets.

    These are Crop Type datasets from `Radiant MLHub
    <https://mlhub.earth/datasets?tags=crop+type>`_
    for semantic segmentation that follow the same format.

    .. versionadded:: 0.3
    """

    msi_bands = (
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    )

    RGB_BANDS = ["B04", "B03", "B02"]

    date_format = "%Y-%m-%dT%H:%M:%SZ"

    @property
    @abc.abstractmethod
    def extra_band_names(self) -> Tuple[str, ...]:
        """Additional band names contained in dataset besides Sentinel-2 bands."""

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """Radiant ML Dataset ID."""

    @property
    @abc.abstractmethod
    def image_meta(self) -> Dict[str, Dict[str, str]]:
        """Meta Image Information.

        Dict that contains .tar.gz 'filename', extracted 'directory'
        name and 'md5'.
        """

    @property
    @abc.abstractmethod
    def target_meta(self) -> Dict[str, Dict[str, str]]:
        """Meta Target Information.

        Dict that contains .tar.gz 'filename', extracted 'directory'
        name and 'md5'.
        """

    @property
    @abc.abstractmethod
    def crop_labels(self) -> List[str]:
        """List that contains crop labels."""

    @property
    @abc.abstractmethod
    def directory_regex(self) -> str:
        """Regex to match filename directories."""

    @property
    @abc.abstractmethod
    def crop_label_key(self) -> str:
        """Key in labels.geojson files that specifies crop type."""

    @property
    @abc.abstractmethod
    def data_splits(self) -> List[str]:
        """Specify available splits of datasets."""

    def __init__(
        self,
        root: str = "data",
        bands: Tuple[str, ...] = msi_bands,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
        cache: bool = True,
    ) -> None:
        """Initialize a new Crop Type Radianet ML Hub Dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            split: select pre-defined train or test split if available
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.all_band_names = self.msi_bands + self.extra_band_names
        self._validate_bands(bands)
        self.bands = bands

        assert split in self.data_splits, (
            "The specfied split is not available " "for {self.__class__.__name__}"
        )
        self.split = split

        self.root = root
        self.checksum = checksum
        self.cache = cache
        self.api_key = api_key
        self.download = download
        self.class2idx = {c: i for i, c in enumerate(self.crop_labels)}
        self.idx2class = {i: c for i, c in enumerate(self.crop_labels)}

        self._verify()

        super().__init__(transforms)

        # fill index base on stac.json files
        i = 0
        pathname = os.path.join(
            root, self.image_meta[self.split]["directory"], "**", "stac.json"
        )
        directory_regex = re.compile(self.directory_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            # labels and source have an id in common
            match = re.search(directory_regex, os.path.dirname(filepath))
            label_path = os.path.join(
                self.root,
                self.target_meta[self.split]["directory"],
                self.target_meta[self.split]["directory"] + "_" + match.group("id"),
                "labels.geojson",
            )
            label_path = label_path.replace("_source_", "_labels_")

            if i == 0:
                # find crs in first B01.tif
                src = rasterio.open(os.path.join(os.path.dirname(filepath), "B01.tif"))

                if crs is None:
                    crs = src.crs
                if res is None:
                    res = 10

            # bboxes in source imagery stac.json are in epsg '4326'
            transformer = Transformer.from_crs(
                CRS.from_epsg(4326), crs.to_epsg(), always_xy=True
            )

            with open(filepath) as stac_file:
                data = json.load(stac_file)

            minx, miny, maxx, maxy = data["bbox"]

            (minx, maxx), (miny, maxy) = transformer.transform(
                [minx, maxx], [miny, maxy]
            )

            date = data["properties"]["datetime"]
            mint, maxt = disambiguate_timestamp(date, self.date_format)

            coords = (minx, maxx, miny, maxy, mint, maxt)
            # insert dict here with filepath and label
            self.index.insert(
                i,
                coords,
                {"img_dir": os.path.dirname(filepath), "label_path": label_path},
            )
            i += 1

        if i == 0:
            raise FileNotFoundError("No stac.json files found!")

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if extracted files already exists
        exists = []
        for directory in [
            self.image_meta[self.split]["directory"],
            self.target_meta[self.split]["directory"],
        ]:
            if os.path.exists(os.path.join(self.root, directory)):
                exists.append(True)
            else:
                exists.append(False)
        if all(exists):
            return

        # check if compressed files exists
        exists = []
        for filename, md5 in zip(
            [
                self.image_meta[self.split]["filename"],
                self.target_meta[self.split]["filename"],
            ],
            [self.image_meta[self.split]["md5"], self.target_meta[self.split]["md5"]],
        ):
            filepath = os.path.join(self.root, filename)
            if os.path.isfile(filepath):
                if self.checksum and not check_integrity(filepath, md5):
                    raise RuntimeError("Dataset found, but corrupted.")
                exists.append(True)
                extract_archive(filepath, self.root)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_radiant_mlhub_dataset(self.dataset_id, self.root, self.api_key)

    def _extract(self) -> None:
        """Extract the dataset."""
        image_archive_path = os.path.join(
            self.root, self.image_meta[self.split]["filename"]
        )
        target_archive_path = os.path.join(
            self.root, self.target_meta[self.split]["filename"]
        )
        for filename in [image_archive_path, target_archive_path]:
            extract_archive(filename, self.root)

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
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask pair and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index TxCxHxW

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # load imagery
        data = self._load_separate_images(filepaths, query)
        # load mask
        mask = self._load_mask(filepaths, query)

        sample = {"image": data, "mask": mask, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_separate_images(
        self, filepaths: Sequence[str], query: BoundingBox
    ) -> Tensor:
        """Load seperate band images.

        Args:
            filepaths: one por more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            images at this query index
        """
        # load imagery
        img_list: List[Tensor] = []
        for filepath in filepaths:
            band_filepaths = []
            for band in getattr(self, "bands", self.all_band_names):
                tif_name: str = band + ".tif"
                band_filename = os.path.join(filepath["img_dir"], tif_name)
                band_filepaths.append(band_filename)
            img_list.append(self._concatenate_bands(band_filepaths, query))

        # stack along time dimension
        data = torch.stack(img_list, dim=0)

        return data

    def _concatenate_bands(
        self, filepaths: Sequence[str], query: BoundingBox
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        img_list: List[Tensor] = []
        for src in vrt_fhs:
            out_width = int(round((query.maxx - query.minx) / self.res))
            out_height = int(round((query.maxy - query.miny) / self.res))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(
                out_shape=out_shape, window=from_bounds(*bounds, src.transform)
            ).astype(np.int32)

            img_list.append(torch.from_numpy(dest))

        tensor = torch.cat(img_list, dim=0)
        return tensor

    @lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src

    def _load_mask(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load the mask.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            labels at this query index
        """
        # only images are patched into tiles, masks are large, so remove duplicates
        label_paths = list({f["label_path"] for f in filepaths})
        per_label_shapes: Dict[str, List[str]] = {}
        for filepath in label_paths:
            with fiona.open(filepath) as src:
                # We need to know the bounding box of the query in the source CRS
                (minx, maxx), (miny, maxy) = fiona.transform.transform(
                    self.crs.to_dict(),
                    src.crs,
                    [query.minx, query.maxx],
                    [query.miny, query.maxy],
                )

                # Filter geometries to those that intersect with the bounding box
                for feature in src.filter(bbox=(minx, miny, maxx, maxy)):
                    # Warp geometries to requested CRS
                    shape = fiona.transform.transform_geom(
                        src.crs, self.crs.to_dict(), feature["geometry"]
                    )
                    # collect shapes per class label to be able to draw separate mask
                    # per class
                    label = feature["properties"][self.crop_label_key]
                    if label in per_label_shapes:
                        per_label_shapes[label].append(shape)
                    else:
                        per_label_shapes[label] = [shape]

        # Rasterize geometries
        width = int(round((query.maxx - query.minx) / self.res))
        height = int(round((query.maxy - query.miny) / self.res))
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if per_label_shapes:
            # create a mask tensor per class that is being found and then concatenate
            # to single segmentation mask with labels
            mask_list: List[Tensor] = []
            for label, shapes in per_label_shapes.items():
                label_mask = rasterio.features.rasterize(
                    shapes, out_shape=(int(height), int(width)), transform=transform
                )
                # assign correct segmentation label
                label_mask *= self.class2idx[label]
                mask_list.append(label_mask)

            mask_stack: "np.typing.NDArray[np.int_]" = np.stack(mask_list, axis=0)
            # assumes non-overlapping labels
            masks = torch.from_numpy(np.sum(mask_stack, axis=0, dtype=np.uint8))

        else:
            # If no features are found in this query, return an empty mask
            # with the default fill value and dtype used by rasterize
            masks = torch.from_numpy(
                np.zeros((int(height), int(width)), dtype=np.uint8)
            )

        return masks

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        time_step: int = 0,
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

        image, mask = sample["image"].numpy(), sample["mask"].numpy()

        assert time_step <= image.shape[0] - 1, (
            "The specified time step"
            " does not exist, image only contains {} time"
            " instances."
        ).format(image.shape[0])

        image = image[time_step, rgb_indices, :, :]

        image = percentile_normalization(image)

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))

        axs[0].imshow(image.transpose(1, 2, 0))
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


class CropTypeTanzaniaGAFCO(CropTypeDatasetRadiantML):
    """Great African Food Company Crop Type Tanzania dataset.

    The `Great African Food Company (GAFCO) Crop Type Tanzania dataset
    <https://mlhub.earth/data/ref_african_crops_tanzania_01>`_ contains both
    time-series imagery and masks for crop type segmentation. Here, the mask
    is the same for all time-series images.

    Dataset Features:

    * Sentinel-2 imagery at 10m resolution
    * 4 labelled regions
    * time-series images between January and December of 2018
    * 6 different crop types

    Dataset Format:

    * images are 13 channel geotiffs, 12 Sentinel 2 bands plus pixel wise
      cloud probability layer
    * stac.json files
    * annotations in geojson files

    Dataset Classes:

    0. No Data
    1. Bush Bean
    2. Dry Bean
    3. Safflower
    4. Sunflower
    5. White Sorghum
    6. Yellow Maize

    If you use this dataset in your research, please use the following citation:

    Great African Food Company (2019) "Great African Food Company Tanzania
    Ground Reference Crop Type Dataset", Version 1.0, Radiant MLHub.
    [Date Accessed] https://doi.org/10.34911/RDNT.5VX40R

    .. note::
       This dataset requires the following additional library to be installed:
       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.3
    """

    dataset_id = "ref_african_crops_tanzania_01"
    image_meta = {
        "train": {
            "filename": "ref_african_crops_tanzania_01_source.tar.gz",
            "directory": "ref_african_crops_tanzania_01_source",
            "md5": "2d94a7f56771701b8ff6598ffdf1f814",
        }
    }
    target_meta = {
        "train": {
            "filename": "ref_african_crops_tanzania_01_labels.tar.gz",
            "directory": "ref_african_crops_tanzania_01_labels",
            "md5": "a8ab4f957f7eee1e8bf1bada4a0a4f54",
        }
    }

    data_splits = ["train"]

    directory_regex = r"""
    _(?P<id>[0-9]{2})
    _(?P<year>[0-9]{4})
    (?P<month>[0-9]{2})
    (?P<day>[0-9]{2})$"""

    extra_band_names = ("CLD",)

    crop_labels = [
        "No Data",
        "Bush Bean",
        "Dry Bean",
        "Safflower",
        "Sunflower",
        "White Sorghum",
        "Yellow Maize",
    ]

    crop_label_key = "Crop"


class CropTypeKenyaPlantVillage(CropTypeDatasetRadiantML):
    """Plant Village Crop Type Kenya dataset.

    The `Plant Village Crop Type Kenya dataset
    <https://mlhub.earth/data/ref_african_crops_kenya_01>`_ contains both
    time-series imagery and masks for crop type segmentation. Here, the mask
    is the same for all time-series images.

    Dataset Features:

    * Sentinel-2 imagery at 10m resolution
    * 3 labelled regions
    * time-series images between April 2018 and March 2020
    * 14 different crop types
    * between 83 and 85 timestamps per labelled region

    Dataset Format:

    * images are 13 channel geotiffs, 12 Sentinel 2 bands plus pixel wise
      cloud probability layer
    * stac.json files
    * annotations in geojson files

    Dataset Classes:

    0. No Data
    1. Banana
    2. Bean
    3. Cabbage
    4. Cassava
    5. Cowpea
    6. Fallowland
    7. Groundnut
    8. Maize
    9. Millet
    10. Sorghum
    11. Soybean
    12. Sugarcane
    13. Sweetpotato
    14. Tomato

    If you use this dataset in your research, please use the following citation:

    PlantVillage (2019) "PlantVillage Kenya Ground Reference Crop Type Dataset",
    Version 1.0, Radiant MLHub. [Date Accessed] https://doi.org/10.34911/RDNT.U41J87

    .. note::
       This dataset requires the following additional library to be installed:
       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.3
    """

    dataset_id = "ref_african_crops_kenya_01"
    image_meta = {
        "train": {
            "filename": "ref_african_crops_kenya_01_source.tar.gz",
            "directory": "ref_african_crops_kenya_01_source",
            "md5": "1428a00cc984c591909c945ec8d30c1b",
        }
    }
    target_meta = {
        "train": {
            "filename": "ref_african_crops_kenya_01_labels.tar.gz",
            "directory": "ref_african_crops_kenya_01_labels",
            "md5": "19e31536982d97f2f62efcea4d860279",
        }
    }

    directory_regex = r"""
    _(?P<id>[0-9]{2})
    _(?P<year>[0-9]{4})
    (?P<month>[0-9]{2})
    (?P<day>[0-9]{2})$"""

    data_splits = ["train"]

    extra_band_names = ("CLD",)

    crop_labels = [
        "No Data",
        "Banana",
        "Bean",
        "Cabbage",
        "Cassava",
        "Cowpea",
        "Fallowland",
        "Groundnut",
        "Maize",
        "Millet",
        "Sorghum",
        "Soybean",
        "Sugarcane",
        "Sweetpotato",
        "Tomato",
    ]

    crop_label_key = "Crop1"


class CropTypeUgandaDalbergDataInsight(CropTypeDatasetRadiantML):
    """Dalberg Data Insight Crop Type Uganda dataset.

    The `Dalberg Data Insight Crop Type Uganda dataset
    <https://mlhub.earth/data/ref_african_crops_uganda_01>`_ contains both
    time-series imagery and masks for crop type segmentation. Here, the mask
    is the same for all time-series images.

    Dataset Features:

    * Sentinel-2 imagery at 10m resolution
    * 53 different labelled regions
    * time-series images between January 2017 and December 2017
    * 2 different crop types
    * between 83 and 85 timestamps per labelled region

    Dataset Format:

    * images are 13 channel geotiffs, 12 Sentinel 2 bands plus pixel wise
      cloud probability layer
    * stac.json files
    * annotations in geojson files

    Dataset Classes:

    0. No Data
    1. Maize
    2. Sorghum

    If you use this dataset in your research, please use the following citation:

    Bocquet, C., & Dalberg Data Insights. (2019) "Dalberg Data Insights Uganda
    Crop Classification", Version 1.0, Radiant MLHub.
    [Date Accessed] https://doi.org/10.34911/RDNT.EII04X

    .. note::
       This dataset requires the following additional library to be installed:
       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.3
    """

    dataset_id = "ref_african_crops_uganda_01"
    image_meta = {
        "train": {
            "filename": "ref_african_crops_uganda_01_source.tar.gz",
            "directory": "ref_african_crops_uganda_01_source",
            "md5": "fd5718698a3167c6e9b65184a5c6b281",
        }
    }
    target_meta = {
        "train": {
            "filename": "ref_african_crops_uganda_01_labels.tar.gz",
            "directory": "ref_african_crops_uganda_01_labels",
            "md5": "3565c040b026372e21d75e163e13e781",
        }
    }

    directory_regex = r"""
    _(?P<id>[0-9]{2})
    _(?P<year>[0-9]{4})
    (?P<month>[0-9]{2})
    (?P<day>[0-9]{2})$"""

    data_splits = ["train"]

    extra_band_names = ("CLD",)

    # labels are lower case here
    crop_labels = ["no data", "maize", "sorghum"]

    crop_label_key = "crop1"


class CropTypeSouthAfricaCompetition(CropTypeDatasetRadiantML):
    """South Africa Crop Type Competition.

    The `South Africa Crop Type Competition dataset
    <https://mlhub.earth/data/ref_south_africa_crops_competition_v1>`_ contains both
    time-series imagery and masks for crop type segmentation. Here, the mask
    is the same for all time-series images, and this version implements the
    Sentinel-2 imagery version.

    Dataset Features:

    * Sentinel-2 imagery at 10m resolution
    * 2650 labelled regions
    * time-series images between 2017-01-05 and 2017-12-31
    * 2 different crop types
    * between 37 and 76 timestamps per labelled region

    Dataset Format:

    * images are 13 channel geotiffs, 12 Sentinel 2 bands plus cloud mask
    * stac.json files
    * annotations in geotiff

    Dataset Classes:

    0. No Data
    1. Lucerne/Medics
    2. Planted pastures (perennial)
    3. Fallow
    4. Wine grapes
    5. Weeds
    6. Small grain grazing
    7. Wheat
    8. Canola
    9. Rooibos

    If you use this dataset in your research, please use the following citation:

    Western Cape Department of Agriculture, Radiant Earth Foundation (2021)
    "Crop Type Classification Dataset for Western Cape, South Africa", Version 1.0,
    Radiant MLHub, [Date Accessed] https://doi.org/10.34911/rdnt.j0co8q

    .. note::
       This dataset requires the following additional library to be installed:
       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.3
    """

    dataset_id = "ref_south_africa_crops_competition_v1"
    image_meta = {
        "train": {
            "filename": "ref_south_africa_crops_competition_v1_train_source_s2.tar.gz",
            "directory": "ref_south_africa_crops_competition_v1_train_source_s2",
            "md5": "cf9cc7d35bba52ad41494fcb2c920eac",
        },
        "test": {
            "filename": "ref_south_africa_crops_competition_v1_test_source_s2.tar.gz",
            "directory": "ref_south_africa_crops_competition_v1_test_source_s2",
            "md5": "b413c829243667f0806390c9947a7bca",
        },
    }
    target_meta = {
        "train": {
            "filename": "ref_south_africa_crops_competition_v1_train_labels.tar.gz",
            "directory": "ref_south_africa_crops_competition_v1_train_labels",
            "md5": "263af3170e3d217144eab171b3271894",
        },
        "test": {
            "filename": "ref_south_africa_crops_competition_v1_test_labels.tar.gz",
            "directory": "ref_south_africa_crops_competition_v1_test_labels",
            "md5": "74b3aba76ba2dec0b83cdcaad8f0dfdb",
        },
    }

    data_splits = ["train", "test"]

    extra_band_names = ("CLM",)

    crop_labels = [
        "No Data",
        "Lucerne/Medics",
        "Planted pastures (perennial)",
        "Fallow",
        "Wine grapes",
        "Weeds",
        "Small grain grazing",
        "Wheat",
        "Canola",
        "Rooibos",
    ]

    # dataset follows a different label format
    crop_label_key = "None"

    directory_regex = r"""
    _(?P<id>[0-9]{4})
    _(?P<year>[0-9]{4})
    _(?P<month>[0-9]{2})
    _(?P<day>[0-9]{2})$"""

    date_format = "%Y-%m-%dT%H:%M:%S+0000"

    # overload method because label for this data is in different format
    def _load_mask(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load the mask.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            labels at this query index
        """
        # load imagery
        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        label_list: List[Tensor] = []

        for filepath in filepaths:
            label_filename = os.path.join(
                os.path.dirname(filepath["label_path"]), "labels.tif"
            )

            src = self._load_warp_file(label_filename)

            out_width = int(round((query.maxx - query.minx) / self.res))
            out_height = int(round((query.maxy - query.miny) / self.res))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(
                out_shape=out_shape, window=from_bounds(*bounds, src.transform)
            ).astype(np.int32)
            label_list.append(torch.from_numpy(dest))

        # stack along time dimension
        labels = torch.stack(label_list, dim=0).squeeze()

        return labels
