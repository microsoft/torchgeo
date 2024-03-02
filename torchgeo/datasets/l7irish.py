# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L7 Irish dataset."""

import glob
import os
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive


class L7Irish(RasterDataset):
    """L7 Irish dataset.

    The `L7 Irish <https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data>`__
    dataset is based on Landsat 7 Enhanced Thematic Mapper Plus (ETM+) Level-1G scenes.
    Manually generated cloud masks are used to train and validate cloud cover assessment
    algorithms, which in turn are intended to compute the percentage of cloud cover in
    each scene.

    Dataset features:

    * Images divided between 9 unique biomes
    * 206 scenes from Landsat 7 ETM+ sensor
    * Imagery from global tiles between June 2000--December 2001
    * 9 Level-1 spectral bands with 30 m per pixel resolution

    Dataset format:

    * Images are composed of single multiband geotiffs
    * Labels are multiclass, stored in single geotiffs
    * Level-1 metadata (MTL.txt file)
    * Landsat 7 ETM+ bands: (B10, B20, B30, B40, B50, B61, B62, B70, B80)

    Dataset classes:

    0. Fill
    1. Cloud Shadow
    2. Clear
    3. Thin Cloud
    4. Cloud

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/F7XD0ZWC
    * https://doi.org/10.1109/TGRS.2011.2164087
    * https://www.sciencebase.gov/catalog/item/573ccf18e4b0dae0d5e4b109

    .. versionadded:: 0.5
    """  # noqa: E501

    url = "https://hf.co/datasets/torchgeo/l7irish/resolve/6807e0b22eca7f9a8a3903ea673b31a115837464/{}.tar.gz"  # noqa: E501

    md5s = {
        "austral": "0a34770b992a62abeb88819feb192436",
        "boreal": "b7cfdd689a3c2fd2a8d572e1c10ed082",
        "mid_latitude_north": "c40abe5ad2487f8ab021cfb954982faa",
        "mid_latitude_south": "37abab7f6ebe3d6cf6a3332144145427",
        "polar_north": "49d9e616bd715057db9acb1c4d234d45",
        "polar_south": "c1503db1cf46d5c37b579190f989e7ec",
        "subtropical_north": "a6010de4c50167260de35beead9d6a65",
        "subtropical_south": "c37d439df2f05bd7cfe87cf6ff61a690",
        "tropical": "d7931419c70f3520a17361d96f1a4810",
    }

    classes = ["Fill", "Cloud Shadow", "Clear", "Thin Cloud", "Cloud"]

    # https://landsat.usgs.gov/cloud-validation/cca_irish_2015/L7_Irish_Cloud_Validation_Masks.xml
    filename_glob = "L71*.TIF"
    filename_regex = r"""
        ^L71
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        _(?P=wrs_row)
        (?P<date>\d{8})
        \.TIF$
    """
    date_format = "%Y%m%d"

    separate_files = False
    rgb_bands = ["B30", "B20", "B10"]
    all_bands = ["B10", "B20", "B30", "B40", "B50", "B61", "B62", "B70", "B80"]

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = CRS.from_epsg(3857),
        res: Optional[float] = None,
        bands: Sequence[str] = all_bands,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new L7Irish instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to EPSG:3857)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            paths, crs=crs, res=res, bands=bands, transforms=transforms, cache=cache
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        if self.files:
            return

        # Check if the tar.gz files have already been downloaded
        assert isinstance(self.paths, str)
        pathname = os.path.join(self.paths, "*.tar.gz")
        if glob.glob(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `paths={self.paths!r}` and `download=False`, "
                "either specify a different `paths` or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for biome, md5 in self.md5s.items():
            download_url(
                self.url.format(biome), self.paths, md5=md5 if self.checksum else None
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str)
        pathname = os.path.join(self.paths, "*.tar.gz")
        for tarfile in glob.iglob(pathname):
            extract_archive(tarfile)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        image = self._merge_files(filepaths, query, self.band_indexes)

        mask_filepaths = []
        for filepath in filepaths:
            path, row = os.path.basename(os.path.dirname(filepath)).split("_")[:2]
            mask_filepath = filepath.replace(
                os.path.basename(filepath), f"L7_{path}_{row}_newmask2015.TIF"
            )
            mask_filepaths.append(mask_filepath)

        mask = self._merge_files(mask_filepaths, query)
        mask_mapping = {64: 1, 128: 2, 192: 3, 255: 4}

        for k, v in mask_mapping.items():
            mask[mask == k] = v

        sample = {
            "crs": self.crs,
            "bbox": query,
            "image": image.float(),
            "mask": mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].numpy().astype("uint8").squeeze()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy().astype("uint8").squeeze()
            num_panels += 1

        kwargs = {"cmap": "gray", "vmin": 0, "vmax": 4, "interpolation": "none"}
        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, **kwargs)
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, **kwargs)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
