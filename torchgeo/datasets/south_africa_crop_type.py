# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""South Africa Crop Type Competition Dataset."""

import os
import re
from typing import Any, Callable, Optional, cast

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .geo import RasterDataset
from .utils import BoundingBox, RGBBandsMissingError


class SouthAfricaCropType(RasterDataset):
    """South Africa Crop Type Challenge dataset.

    The `South Africa Crop Type Challenge
    <https://beta.source.coop/repositories/radiantearth/south-africa-crops-competition/description/>`__
    dataset includes satellite imagery from Sentinel-1 and Sentinel-2 and labels for
    crop type that were collected by aerial
    and vehicle survey from May 2017 to March 2018. Data was collected by the
    provided by the Western Cape Department of Agriculture and is
    is available via the Radiant Earth Foundation.
    Each chip is matched with a label.
    Each pixel in the label contains an integer field number and crop type class.

    Dataset format:

    * images are 2-band Sentinel 1 and 12-bands Sentinel-2 data with a cloud mask
    * masks are tiff image with unique values representing the class and field id.

    Dataset classes:

    0: No Data
    1: Lucerne/Medics
    2: Planted pastures (perennial)
    3: Fallow
    4: Wine grapes
    5: Weeds
    6: Small grain grazing
    7: Wheat
    8: Canola
    9: Rooibos

    If you use this dataset in your research, please cite the following dataset:
    Western Cape Department of Agriculture, Radiant Earth Foundation (2021)
    "Crop Type Classification Dataset for Western Cape, South Africa",
    Version 1.0, Radiant MLHub, https://doi.org/10.34911/rdnt.j0co8q

    .. versionadded:: 0.6
    """

    filename_regex = r"""
        ^(?P<field_id>[0-9]*)_(?P<year>[0-9]{4})_
        (?P<month>[0-9]{2})_(?P<day>[0-9]{2})_
        (?P<band>(B[0-9A-Z]{2} | VH | VV))_10m\.tif"""

    rgb_bands = ["B04", "B03", "B02"]
    S1_bands = ["VH", "VV"]
    S2_bands = [
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
    ]
    all_bands: list[str] = S1_bands + S2_bands
    cmap = {
        0: (0, 0, 0, 255),
        1: (255, 211, 0, 255),
        2: (255, 37, 37, 255),
        3: (0, 168, 226, 255),
        4: (255, 158, 9, 255),
        5: (37, 111, 0, 255),
        6: (255, 255, 0, 255),
        8: (111, 166, 0, 255),
        9: (0, 175, 73, 255),
    }

    def __init__(
        self,
        root: str = "",
        crs: CRS = CRS.from_epsg(32634),
        classes: list[int] = list(cmap.keys()),
        bands: list[str] = all_bands,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new South Africa dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: coordinate reference system to be used
            classes: crop type classes to be included
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert (
            set(classes) <= self.cmap.keys()
        ), f"Only the following classes are valid: {list(self.cmap.keys())}."
        assert 0 in classes, "Classes must include the background class: 0"

        self.root = root
        self.classes = classes
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)

        super().__init__(paths=root, crs=crs, bands=bands, transforms=transforms)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, label, and field ids at that index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data_list: list[Tensor] = []
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        sample_dates_s1 = {}
        sample_dates_s2 = {}
        for band in self.bands:
            sample_dates = sample_dates_s1 if band in self.S1_bands else sample_dates_s2
            fields_added = []
            band_filepaths = []
            for filepath in filepaths:
                filename = os.path.basename(filepath)
                match = re.match(filename_regex, filename)
                if match and match.groupdict()["band"] == band:
                    field_id = match.groupdict()["field_id"]
                    month = match.groupdict()["month"]
                    day = match.groupdict()["day"]
                    if field_id not in fields_added and month == "07":
                        if field_id not in sample_dates:
                            sample_dates[field_id] = day
                        if day == sample_dates[field_id]:
                            band_filepaths.append(filepath)
                            fields_added.append(field_id)
            data_list.append(self._merge_files(band_filepaths, query))
        image = torch.cat(data_list)

        mask_filepaths = []
        for root, dirs, files in os.walk(os.path.join(self.root, "train", "labels")):
            for file in files:
                if not file.endswith("_field_ids.tif") and file.endswith(".tif"):
                    file_path = os.path.join(root, file)
                    mask_filepaths.append(file_path)

        mask = self._merge_files(mask_filepaths, query)

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

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].squeeze()
        ncols = 2

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred = sample["prediction"].squeeze()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_prediction:
            axs[2].imshow(pred)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
