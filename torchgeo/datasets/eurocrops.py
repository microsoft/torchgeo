# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroCrops dataset."""

import csv
import os
from collections.abc import Callable, Iterable
from typing import Any

import fiona
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import VectorDataset
from .utils import Path, check_integrity, download_and_extract_archive, download_url


class EuroCrops(VectorDataset):
    """EuroCrops Dataset (Version 9).

    The `EuroCrops <https://www.eurocrops.tum.de/index.html>`__ dataset combines "all
    publicly available self-declared crop reporting datasets from countries of the
    European Union" into a unified format. The dataset is released under CC BY 4.0 Deed.

    The dataset consists of shapefiles containing a total of 22M polygons. Each polygon
    is tagged with a "EC_hcat_n" attribute indicating the harmonized crop name grown
    within the polygon in the year associated with the shapefile.

    If you use this dataset in your research, please follow the citation guidelines at:

    * https://github.com/maja601/EuroCrops#reference.

    .. versionadded:: 0.6
    """

    base_url = 'https://zenodo.org/records/8229128/files/'

    hcat_fname = 'HCAT2.csv'
    hcat_md5 = 'b323e8de3d8d507bd0550968925b6906'
    # Name of the column containing HCAT code in CSV file.
    hcat_code_column = 'HCAT2_code'

    label_name = 'EC_hcat_c'

    filename_glob = '*_EC*.shp'

    # Override variables to automatically extract timestamp.
    filename_regex = r"""
        ^(?P<country>[A-Z]{2})
        (_(?P<region>[A-Z]+))?
        _
        (?P<date>\d{4})
        _
        (?P<suffix>EC(?:21)?)
        \.shp$
    """
    date_format = '%Y'

    # Filename and md5 of files in this dataset on zenodo.
    zenodo_files: tuple[tuple[str, str], ...] = (
        ('AT_2021.zip', '490241df2e3d62812e572049fc0c36c5'),
        ('BE_VLG_2021.zip', 'ac4b9e12ad39b1cba47fdff1a786c2d7'),
        ('DE_LS_2021.zip', '6d94e663a3ff7988b32cb36ea24a724f'),
        ('DE_NRW_2021.zip', 'a5af4e520cc433b9014cf8389c8f4c1f'),
        ('DK_2019.zip', 'd296478680edc3173422b379ace323d8'),
        ('EE_2021.zip', 'a7596f6691ad778a912d5a07e7ca6e41'),
        ('ES_NA_2020.zip', '023f3b397d0f6f7a020508ed8320d543'),
        ('FR_2018.zip', '282304734f156fb4df93a60b30e54c29'),
        ('HR_2020.zip', '8bfe2b0cbd580737adcf7335682a1ea5'),
        ('LT_2021.zip', 'c7597214b90505877ee0cfa1232ac45f'),
        ('LV_2021.zip', 'b7253f96c8699d98ca503787f577ce26'),
        ('NL_2020.zip', '823da32d28695b8b016740449391c0db'),
        ('PT.zip', '3dba9c89c559b34d57acd286505bcb66'),
        ('SE_2021.zip', 'cab164c1c400fce56f7f1873bc966858'),
        ('SI_2021.zip', '6b2dde6ba9d09c3ef8145ea520576228'),
        ('SK_2021.zip', 'c7762b4073869673edc08502e7b22f01'),
        # Year is unknown for Romania portion (ny = no year).
        # We skip since it is inconsistent with the rest of the data.
        # ("RO_ny.zip", "648e1504097765b4b7f825decc838882"),
    )

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS = CRS.from_epsg(4326),
        res: float | tuple[float, float] = (0.00001, 0.00001),
        classes: list[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EuroCrops instance.

        Args:
            paths: one or more root directories to search for files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to WGS-84)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
            classes: list of classes to include (specified by their HCAT code),
                the rest will be mapped to 0 (defaults to all classes)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.paths = paths
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        self._load_class_map(classes)

        super().__init__(
            paths=paths,
            crs=crs,
            res=res,
            transforms=transforms,
            label_name=self.label_name,
        )

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        # Check if the extracted files already exist
        if self.files and not self.checksum:
            return True

        assert isinstance(self.paths, str | os.PathLike)

        filepath = os.path.join(self.paths, self.hcat_fname)
        if not check_integrity(filepath, self.hcat_md5 if self.checksum else None):
            return False

        for fname, md5 in self.zenodo_files:
            filepath = os.path.join(self.paths, fname)
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        assert isinstance(self.paths, str | os.PathLike)
        download_url(
            self.base_url + self.hcat_fname,
            self.paths,
            md5=self.hcat_md5 if self.checksum else None,
        )
        for fname, md5 in self.zenodo_files:
            download_and_extract_archive(
                self.base_url + fname, self.paths, md5=md5 if self.checksum else None
            )

    def _load_class_map(self, classes: list[str] | None) -> None:
        """Load map from HCAT class codes to class indices.

        If classes is provided, then we simply use those codes. Otherwise, we load
        all the codes from CSV file.

        Args:
            classes: list of HCAT codes specifying classes to use
                (defaults to all classes)
        """
        if not classes:
            assert isinstance(self.paths, str | os.PathLike)
            classes = []
            filepath = os.path.join(self.paths, self.hcat_fname)
            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hcat_code = row[self.hcat_code_column]
                    classes.append(hcat_code)

        self.class_map = {}
        for idx, hcat_code in enumerate(classes):
            self.class_map[hcat_code] = idx + 1

    def get_label(self, feature: 'fiona.model.Feature') -> int:
        """Get label value to use for rendering a feature.

        Args:
            feature: the :class:`fiona.model.Feature` from which to extract the label.

        Returns:
            the integer label, or 0 if the feature should not be rendered.
        """
        # Convert the HCAT code of this feature to its index per self.class_map.
        # We go up the class hierarchy until there is a match.
        # (Parent code is computed by replacing rightmost non-0 character with 0.)
        hcat_code = feature['properties'][self.label_name]
        if hcat_code is None:
            return 0

        while True:
            if hcat_code in self.class_map:
                return self.class_map[hcat_code]
            hcat_code_list = list(hcat_code)
            if all(c == '0' for c in hcat_code_list):
                break
            for i in range(len(hcat_code_list) - 1, -1, -1):
                if hcat_code_list[i] == '0':
                    continue
                hcat_code_list[i] = '0'
                break
            hcat_code = ''.join(hcat_code_list)
        return 0

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`VectorDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample['mask'].squeeze()
        ncols = 1

        showing_prediction = 'prediction' in sample
        if showing_prediction:
            pred = sample['prediction'].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4, 4))

        def apply_cmap(
            arr: 'np.typing.NDArray[Any]',
        ) -> 'np.typing.NDArray[np.float64]':
            # Color 0 as black, while applying default color map for the class indices.
            cmap = plt.get_cmap('viridis')
            im: np.typing.NDArray[np.float64] = cmap(arr / len(self.class_map))
            im[arr == 0] = 0
            return im

        if showing_prediction:
            axs[0].imshow(apply_cmap(mask), interpolation='none')
            axs[0].axis('off')
            axs[1].imshow(apply_cmap(pred), interpolation='none')
            axs[1].axis('off')
            if show_titles:
                axs[0].set_title('Mask')
                axs[1].set_title('Prediction')
        else:
            axs.imshow(apply_cmap(mask), interpolation='none')
            axs.axis('off')
            if show_titles:
                axs.set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
