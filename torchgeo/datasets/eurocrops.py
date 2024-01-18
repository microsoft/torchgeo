# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroCrops dataset."""

import csv
import os
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .geo import VectorDataset
from .utils import (
    DatasetNotFoundError,
    check_integrity,
    download_and_extract_archive,
    download_url,
)


def split_hcat_code(hcat_code: str, hcat_level: int) -> tuple[str, str]:
    """Splits the specified HCAT code into its prefix and suffix.

    The code is split at the given hcat_level.
    For example, a code 3301010100 at Level 4 has prefix "330101" and suffix "0100".

    Args:
        hcat_code: a class code like 3301010100.
        hcat_level: the HCAT level (3, 4, 5, or 6).

    Returns:
        A tuple containing, respectively, the HCAT prefix and suffix. At Level 3, the
        prefix is the first 4 digits and suffix is the second 6 digits, while at Level
        6, the prefix is the full code and the suffix is empty.
    """
    assert len(hcat_code) == 10
    if hcat_level == 3:
        return hcat_code[0:4], hcat_code[4:10]
    elif hcat_level == 4:
        return hcat_code[0:6], hcat_code[6:10]
    elif hcat_level == 5:
        return hcat_code[0:8], hcat_code[8:10]
    elif hcat_level == 6:
        return hcat_code, ""
    raise ValueError(f"Invalid HCAT level {hcat_level}.")


class EuroCrops(VectorDataset):
    """EuroCrops Dataset (Version 9).

    The `EuroCrops <https://www.eurocrops.tum.de/index.html>`__ dataset combines "all
    publicly available self-declared crop reporting datasets from countries of the
    European Union" into a unified format. The dataset is released under CC BY 4.0 Deed.

    The dataset consists of shapefiles containing a total of X polygons. Each polygon
    is tagged with a "EC_hcat_n" attribute indicating the harmonized crop name grown
    within the polygon in the year associated with the shapefile.

    If you use this dataset in your research, please follow the citation guidelines at
    https://github.com/maja601/EuroCrops#reference.
    """

    base_url = "https://zenodo.org/records/8229128/files/"

    hcat_fname = "HCAT2.csv"
    hcat_md5 = "2443cb1ef05ad16840f0f6a4c685b72d"
    # Valid HCAT levels.
    hcat_levels = [3, 4, 5, 6]
    # Name of the column containing HCAT code in CSV file.
    hcat_code_column = "HCAT2_code"

    label_name = "EC_hcat_c"

    # Filename, year, and md5 of files in this dataset on zenodo.
    zenodo_files = [
        ("AT_2021.zip", 2021, "490241df2e3d62812e572049fc0c36c5"),
        ("BE_VLG_2021.zip", 2021, "ac4b9e12ad39b1cba47fdff1a786c2d7"),
        ("DE_LS_2021.zip", 2021, "6d94e663a3ff7988b32cb36ea24a724f"),
        ("DE_NRW_2021.zip", 2021, "a5af4e520cc433b9014cf8389c8f4c1f"),
        ("DK_2019.zip", 2019, "d296478680edc3173422b379ace323d8"),
        ("EE_2021.zip", 2021, "a7596f6691ad778a912d5a07e7ca6e41"),
        ("ES_NA_2020.zip", 2020, "023f3b397d0f6f7a020508ed8320d543"),
        ("FR_2018.zip", 2018, "282304734f156fb4df93a60b30e54c29"),
        ("HR_2020.zip", 2020, "8bfe2b0cbd580737adcf7335682a1ea5"),
        ("LT_2021.zip", 2021, "c7597214b90505877ee0cfa1232ac45f"),
        ("LV_2021.zip", 2021, "b7253f96c8699d98ca503787f577ce26"),
        ("NL_2020.zip", 2020, "823da32d28695b8b016740449391c0db"),
        ("PT.zip", 2021, "3dba9c89c559b34d57acd286505bcb66"),
        ("SE_2021.zip", 2021, "cab164c1c400fce56f7f1873bc966858"),
        ("SI_2021.zip", 2021, "6b2dde6ba9d09c3ef8145ea520576228"),
        ("SK_2021.zip", 2021, "c7762b4073869673edc08502e7b22f01"),
        # Year is unknown for Romania portion (ny = no year).
        # ("RO_ny", ??),
    ]

    # Color palette to choose from.
    # There are hundreds of classes so we pick color via modulo of class index.
    colors = [
        (70, 107, 159, 255),
        (209, 222, 248, 255),
        (222, 197, 197, 255),
        (217, 146, 130, 255),
        (235, 0, 0, 255),
        (171, 0, 0, 255),
        (179, 172, 159, 255),
        (104, 171, 95, 255),
        (28, 95, 44, 255),
        (181, 197, 143, 255),
        (204, 184, 121, 255),
        (223, 223, 194, 255),
        (220, 217, 57, 255),
        (171, 108, 40, 255),
        (184, 217, 235, 255),
        (108, 159, 184, 255),
    ]

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = None,
        res: float = 0.00001,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
        hcat_level: int = 6,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search for files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            hcat_level: use labels at this level of the
                `HCATv2 hierarchy <https://www.eurocrops.tum.de/taxonomy.html>`__.
                Valid levels are 3, 4, 5, and 6.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.paths = paths
        self.checksum = checksum
        assert hcat_level in [3, 4, 5, 6]
        self.hcat_level = hcat_level

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self._load_classes()
        self.cmap = torch.zeros((len(self.classes) + 1, 4), dtype=torch.uint8)
        for class_index in self.classes.values():
            color = self.colors[class_index % len(self.colors)]
            self.cmap[class_index, :] = torch.tensor(color)

        super().__init__(
            paths=paths,
            crs=crs,
            res=res,
            transforms=transforms,
            label_fn=self._get_class_index,
        )

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        assert isinstance(self.paths, str)

        filepath = os.path.join(self.paths, self.hcat_fname)
        if not check_integrity(filepath, self.hcat_md5 if self.checksum else None):
            return False

        for fname, year, md5 in self.zenodo_files:
            filepath = os.path.join(self.paths, fname)
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        assert isinstance(self.paths, str)
        download_url(
            self.base_url + self.hcat_fname,
            self.paths,
            md5=self.hcat_md5 if self.checksum else None,
        )
        for fname, year, md5 in self.zenodo_files:
            download_and_extract_archive(
                self.base_url + fname, self.paths, md5=md5 if self.checksum else None
            )

    def _load_classes(self) -> None:
        """Load classes from the HCAT CSV file.

        The classes are loaded corresponding to the hcat_level specified by the user.
        """
        assert isinstance(self.paths, str)
        filepath = os.path.join(self.paths, self.hcat_fname)
        with open(filepath) as f:
            reader = csv.DictReader(f)
            # Create classes dict assigning each class to its row index in the CSV.
            # Only retain classes up to the specified HCAT level.
            self.classes = {}
            for row in reader:
                hcat_code = row[self.hcat_code_column]
                code_prefix, code_suffix = split_hcat_code(hcat_code, self.hcat_level)
                # Only keep classes where code_suffix is all 0s.
                if code_suffix != "" and int(code_suffix) != 0:
                    continue
                self.classes[code_prefix] = len(self.classes) + 1

    def _get_class_index(self, feat) -> int:
        """Get class index from a dataset feature."""
        # Convert the HCAT code of this feature to its index in self.classes.
        hcat_code = feat["properties"][self.label_name]
        code_prefix, _ = split_hcat_code(hcat_code, self.hcat_level)
        return self.classes[code_prefix]

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`VectorDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
            Method now takes a sample dict, not a Tensor. Additionally, it is possible
            to show subplot titles and/or use a custom suptitle.
        """
        mask = sample["mask"].squeeze()
        ncols = 1

        showing_prediction = "prediction" in sample
        if showing_prediction:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 4))

        if showing_prediction:
            axs[0].imshow(self.cmap[mask.int()], interpolation="none")
            axs[0].axis("off")
            axs[1].imshow(self.cmap[pred.int()], interpolation="none")
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(self.cmap[mask.int()], interpolation="none")
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
