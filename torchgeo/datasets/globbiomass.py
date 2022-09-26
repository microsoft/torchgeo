# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""GlobBiomass dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import torch
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import BoundingBox, check_integrity, extract_archive


class GlobBiomass(RasterDataset):
    """GlobBiomass dataset.

    The `GlobBiomass dataset <https://doi.pangaea.de/10.1594/PANGAEA.894711>`_
    consists of global pixel wise aboveground biomass (AGB) and growth stock
    volume (GSV) maps.

    Dataset features:

    * estimates of AGB and GSV around the world at ~100m per pixel resolution
      (45,000x45,0000 px)
    * standard error maps of respective measurement at same resolution

    Dataset format:

    * estimate maps are single-channel
    * standard error maps are single-channel

    The data can be manually downloaded from `this website
    <https://globbiomass.org/wp-content/uploads/GB_Maps/
    Globbiomass_global_dataset.html>`_.

    If you use this dataset please cite it with the following citation:

    * Santoro, M. et al. (2018): GlobBiomass - global datasets of forest biomass.
      PANGAEA, https://doi.org/10.1594/PANGAEA.894711

    .. versionadded:: 0.3
    """

    is_image = False

    filename_regex = r"""^
        (?P<tile>[0-9A-Z]*)
        _(?P<measurement>[a-z]{3})
    """

    measurements = ["agb", "gsv"]

    md5s = {
        "N00E020_agb.zip": "bd83a3a4c143885d1962bde549413be6",
        "N00E020_gsv.zip": "da5ddb88e369df2d781a0c6be008ae79",
        "N00E060_agb.zip": "85eaca95b939086cc528e396b75bd097",
        "N00E060_gsv.zip": "ec84174697c17ca4db2967374446ab30",
        "N00E100_agb.zip": "c50c7c996615c1c6f19cb383ef11812a",
        "N00E100_gsv.zip": "6e0ff834db822d3710ed40d00a200e8f",
        "N00E140_agb.zip": "73f0b44b9e137789cefb711ef9aa281b",
        "N00E140_gsv.zip": "43be3dd4563b63d12de006d240ba5edf",
        "N00W020_agb.zip": "4fb979732f0a22cc7a2ca3667698084b",
        "N00W020_gsv.zip": "ac5bbeedaa0f94a5e01c7a86751d6891",
        "N00W060_agb.zip": "59da0b32b08fbbcd2dd76926a849562b",
        "N00W060_gsv.zip": "5ca9598f621a7d10ab1d623ee5b44aa6",
        "N00W100_agb.zip": "a819b75a39e8d4d37b15745c96ea1e35",
        "N00W100_gsv.zip": "71aad3669d522f7190029ec33350831a",
        "N00W180_agb.zip": "5a1d7486d8310fbaf4980a76e9ffcd78",
        "N00W180_gsv.zip": "274be7dbb4e6d7563773cc302129a9c7",
        "N40E020_agb.zip": "38bc7170f94734b365d614a566f872e7",
        "N40E020_gsv.zip": "b52c1c777d68c331cc058a273530536e",
        "N40E060_agb.zip": "1d94ad59f3f26664fefa4d7308b63f05",
        "N40E060_gsv.zip": "3b68786b7641400077ef340a7ef748f4",
        "N40E100_agb.zip": "3ccb436047c0db416fb237435645989c",
        "N40E100_gsv.zip": "c44efe9e7ce2ae0f2e39b0db10f06c71",
        "N40E140_agb.zip": "35ea51da229af1312ba4aaafc0dbd5d6",
        "N40E140_gsv.zip": "8431828708c84263a4971a8779864f69",
        "N40W020_agb.zip": "38345a1826719301ab1a0251b4835cc2",
        "N40W020_gsv.zip": "5e136b7c2f921cd425cb5cc5669e7693",
        "N40W060_agb.zip": "e3f54df1d188c0132ecf5aef3dc54ca6",
        "N40W060_gsv.zip": "09093d78ffef0220cb459a88e61e3093",
        "N40W100_agb.zip": "cc21ce8793e5594dc7a0b45f0d0f1466",
        "N40W100_gsv.zip": "21be1398df88818d04dcce422e2010a6",
        "N40W140_agb.zip": "64665f53fad7386abb1cf4a44a1c8b1a",
        "N40W140_gsv.zip": "b59405219fc807cbe745789fbb6936a6",
        "N40W180_agb.zip": "f83ef786da8333ee739e49be108994c1",
        "N40W180_gsv.zip": "1f2eb8912b1a204eaeb2858b7e398baa",
        "N80E020_agb.zip": "7f7aed44802890672bd908e28eda6f15",
        "N80E020_gsv.zip": "6e285eec66306e56dc3a81adc0da2a27",
        "N80E060_agb.zip": "55e7031e0207888f25f27efa9a0ab8f4",
        "N80E060_gsv.zip": "8d14c7f61ad2aed527e124f9aacae30c",
        "N80E100_agb.zip": "562eafd2813ff06e47284c48324bb1c7",
        "N80E100_gsv.zip": "73067e0fac442c330ae2294996280042",
        "N80E140_agb.zip": "1b51ce0df0dba925c5ef2149bebca115",
        "N80E140_gsv.zip": "37ee3047d281fc34fa3a9e024a8317a1",
        "N80W020_agb.zip": "60dde6adc0dfa219a34c976367f571c0",
        "N80W020_gsv.zip": "b7be4e97bb4179710291ee8dee27f538",
        "N80W060_agb.zip": "db7d35d0375851c4a181c3a8fa8b480e",
        "N80W060_gsv.zip": "d36ffcf4622348382454c979baf53234",
        "N80W100_agb.zip": "c0dbf53e635dabf9a4d7d1756adeda69",
        "N80W100_gsv.zip": "abdeaf0d65da1216c326b6d0ce27d61b",
        "N80W140_agb.zip": "7719c0efd23cd86215fea0285fd0ea4a",
        "N80W140_gsv.zip": "499969bed381197ee9427a2e3f455a2e",
        "N80W180_agb.zip": "e3a163d1944e1989a07225d262a01c6f",
        "N80W180_gsv.zip": "5d39ec0368cfe63c40c66d61ae07f577",
        "S40E140_agb.zip": "263eb077a984117b41cc7cfa0c32915b",
        "S40E140_gsv.zip": "e0ffad85fbade4fb711cc5b3c7543898",
        "S40W060_agb.zip": "2cbf6858c48f36add896db660826829b",
        "S40W060_gsv.zip": "04dbfd4aca0bd2a2a7d8f563c8659252",
        "S40W100_agb.zip": "ae89f021e7d9c2afea433878f77d1dd6",
        "S40W100_gsv.zip": "b6aa3f276e1b51dade803a71df2acde6",
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        measurement: str = "agb",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            measurement: use data from 'agb' or 'gsv' measurement
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if dataset is missing or checksum fails
            AssertionError: if measurement argument is invalid, or not a str
        """
        self.root = root
        self.checksum = checksum

        assert isinstance(measurement, str), "Measurement argument must be a str."
        assert (
            measurement in self.measurements
        ), "You have entered an invalid measurement, please choose one of {}.".format(
            self.measurements
        )
        self.measurement = measurement

        self.filename_glob = f"*0_{self.measurement}*.tif"
        self.zipfile_glob = f"*0_{self.measurement}.zip"

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample at index consisting of measurement mask with 2 channels,
            where the first is the measurement and the second the error map

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        measurement_paths = [f for f in filepaths if "err" not in f]
        mask = self._merge_files(measurement_paths, query)

        std_error_paths = [f for f in filepaths if "err" in f]
        std_err_mask = self._merge_files(std_error_paths, query)

        mask = torch.cat((mask, std_err_mask), dim=0)

        sample = {"mask": mask, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
        """
        # Check if the extracted file already exists
        pathname = os.path.join(self.root, self.filename_glob)
        if glob.glob(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile_glob)
        if glob.glob(pathname):
            for zipfile in glob.iglob(pathname):
                filename = os.path.basename(zipfile)
                if self.checksum and not check_integrity(zipfile, self.md5s[filename]):
                    raise RuntimeError("Dataset found, but corrupted.")
                extract_archive(zipfile)
            return

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` "
            "either specify a different `root` directory or make sure you "
            "have manually downloaded the dataset as suggested in the documentation."
        )

    def plot(
        self,
        sample: Dict[str, Any],
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
        tensor = sample["mask"]
        mask = tensor[0, ...]
        error_mask = tensor[1, ...]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"][0, ...]
            ncols = 3
        else:
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(error_mask)
            axs[1].axis("off")
            axs[2].imshow(pred)
            axs[2].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Uncertainty Mask")
                axs[2].set_title("Prediction")
        else:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(error_mask)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Uncertainty Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
