# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L7 Irish dataset."""

import glob
import os
import re
from collections.abc import Sequence
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
import torch
from rasterio.crs import CRS
from torch import Tensor

from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive


class L7Irish(RasterDataset):
    """L7 Irish dataset.

    The `L7 Irish <https://landsat.usgs.gov/landsat-7-cloud-cover-assessment-validation-data>`__ dataset
    is based on Landsat 7 Enhanced Thematic Mapper Plus (ETM+) Level-1G scenes.
    Manually generated cloud masks are used to train and validate cloud cover assessment
    algorithms, which in turn are intended to compute the percentage of cloud cover in each
    scene.

    Dataset features:

    * 206 scenes from Landsat-7 ETM+ tiles
    * Imagery from global tiles between June 2000--December 2001
    * 9 Level-1 spectral bands with 15 and 30 m per pixel resolution

    Dataset format:

    * Images are composed of multiple single channel geotiffs
    * Labels are multiclass, stored in a single geotiffs file per image
    * Level-1 metadata (MTL.txt file)
    * Landsat-7 ETM+ bands: (B10, B20, B30, B40, B50, B61, B62, B70, B80)

    Dataset classes (5):

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

    url = "https://huggingface.co/datasets/torchgeo/l7irish/resolve/main/{}.tar.gz"  # noqa: E501

    md5s = {
        "austral": "9c2629884c1e7251e24953e1e5f880de",
        "boreal": "0a9f50998c0fb47c0cc226faf479f883",
        "mid_latitude_north": "0860e218403d949f4b38e4f9f70e0087",
        "mid_latitude_south": "c66bbeaa6dbf0ba2cd26b9eea89eb3a4",
        "polar_north": "18a6b9b4684ae91bfdcc7b78ea1f42ee",
        "polar_south": "a12e4d7fddaa377259328190f10a1c17",
        "subtropical_north": "ebdfaee37ffc5ba1bd4763f7f72df97f",
        "subtropical_south": "3670c9490753efe3d36927329bb87e2f",
        "tropical": "f60c93d8609c72ac86e858105b6272f2",
    }

    classes = ["Fill", "Cloud Shadow", "Clear", "Thin Cloud", "Cloud"]

    # https://landsat.usgs.gov/cloud-validation/cca_irish_2015/L7_Irish_Cloud_Validation_Masks.xml
    filename_glob = "L7*_B10.TIF"
    filename_regex = r"""
        ^L7[12]
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        _(?P=wrs_row)
        (?P<date>\d{8})
        _(?P<band>B\d{2})
        \.TIF$
    """
    date_format = "%Y%m%d"

    separate_files = True
    rgb_bands = ["B30", "B20", "B10"]
    all_bands = ["B10", "B20", "B30", "B40", "B50", "B61", "B62", "B70", "B80"]

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Sequence[str] = all_bands,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new L7Irish instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
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
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(
            root, crs=crs, res=res, bands=bands, transforms=transforms, cache=cache
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, "**", self.filename_glob)
        for fname in glob.iglob(pathname, recursive=True):
            return

        # Check if the tar files have already been downloaded
        pathname = os.path.join(self.root, "*.tar.gz")
        if glob.glob(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for biome, md5 in self.md5s.items():
            download_url(
                self.url.format(biome), self.root, md5=md5 if self.checksum else None
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, "*.tar.gz")
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

        image_list: list[Tensor] = []
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for band in self.all_bands:
            band_filepaths = []
            for filepath in filepaths:
                filename = os.path.basename(filepath)
                directory = os.path.dirname(filepath)
                match = re.match(filename_regex, filename)
                if match:
                    if "date" in match.groupdict():
                        start = match.start("band")
                        end = match.end("band")
                        filename = filename[:start] + band + filename[end:]
                        if band in ["B62", "B70", "B80"]:
                            filename = filename.replace("L71", "L72")
                filepath = os.path.join(directory, filename)
                band_filepaths.append(filepath)
            image_list.append(self._merge_files(band_filepaths, query))
        image = torch.cat(image_list)

        mask_filepaths = []
        for filepath in filepaths:
            path, row = os.path.basename(os.path.dirname(filepath)).split("_")[:2]
            mask_filepath = filepath.replace(
                os.path.basename(filepath), f"L7_{path}_{row}_newmask2015.TIF"
            )
            mask_filepaths.append(mask_filepath)

        mask = self._merge_files(mask_filepaths, query)
        mask_mapping = {64: 1, 128: 2, 191: 3, 255: 4}

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
    ) -> plt.Figure:
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

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=4, cmap="gray")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=4, cmap="gray")
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
