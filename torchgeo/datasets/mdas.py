# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MDAS dataset."""

import os
from collections.abc import Callable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_and_extract_archive, extract_archive


class MDAS(NonGeoDataset):
    """MDAS dataset.

    The `MDAS <https://essd.copernicus.org/articles/15/113/2023/>`__ multimodal dataset
    is a comprehensive dataset for the city of Augsburg, Germany, collected on 7th May 2018.
    It includes SAR, multispectral, hyperspectral, DSM, and GIS data,
    providing comprehensive options for data fusion research.
    MDAS supports applications like resolution enhancement, spectral unmixing, and land cover classification.

    Dataset features:

    * 3K DSM data
    * 3K high resolution RGB image
    * Original very high resolution HySpex airborne imagery
    * EeteS simulated imagery with 10m GSD and EnMAP spectral bands
    * EeteS simulated imagery with 30m GSD and EnMAP spectral bands
    * EeteS simulated imagery with 10m GSD and Sentinel-2 spectral bands
    * Sentinel-2 L2A product
    * Sentinel-1 GRD product
    * Open Street Map (OSM) labels

    Dataset format:

    * 3K_RGB.tif (Shape: (4, 15000, 18000)px, Data Type: uint8)
    * 3K_dsm.tif (Shape: (1, 10000, 12000)px, Data Type: float32)
    * HySpex.tif (Shape: (368, 1364, 1636)px, Data Type: int16)
    * EeteS_EnMAP_2dot2m.tif (Shape: (242, 1364, 1636)px, Data Type: float32)
    * EeteS_EnMAP_10m.tif (Shape: (242, 300, 360)px, Data Type: uint16)
    * EeteS_EnMAP_30m.tif (Shape: (242, 100, 120)px, Data Type: uint16)
    * EeteS_Sentinel_2_10m.tif (Shape: (4, 300, 360)px, Data Type: uint16)
    * Sentinel_2.tif (Shape: (12, 300, 360)px, Data Type: uint16)
    * Sentinel_1.tif (Shape: (2, 300, 360)px, Data Type: float32)
    * osm_buildings.tif (Shape: (1, 1364, 1636)px, Data Type: uint8)
    * osm_landuse.tif (Shape: (1, 1364, 1636)px, Data Type: float64)
    * osm_water.tif (Shape: (1, 1364, 1636)px, Data Type: float64)

    If you use this dataset in your research, please cite the following paper:

    * https://essd.copernicus.org/articles/15/113/2023/

    .. versionadded:: 0.7
    """

    valid_modalities = (
        '3K_DSM',
        '3K_RGB',
        'HySpex',
        'EeteS_EnMAP_10m',
        'EeteS_EnMAP_30m',
        'EeteS_Sentinel_2_10m',
        'Sentinel_2',
        'Sentinel_1',
        'osm_buildings',
        'osm_landuse',
        'osm_water',
    )
    landuse_class_names: ClassVar[dict[int, str]] = {
        0: 'no label',
        1: 'forest',
        2: 'park',
        3: 'residential',
        4: 'industrial',
        5: 'farm',
        6: 'cemetery',
        7: 'allotments',
        8: 'meadow',
        9: 'commercial',
        10: 'nature reserve',
        11: 'recreation ground',
        12: 'retail',
        13: 'military',
        14: 'quarry',
        15: 'orchard',
        16: 'scrub',
        17: 'grass',
        18: 'heath',
    }

    landuse_mapping: ClassVar[dict[int, int]] = {
        -2147483647: 0,
        7201: 1,
        7202: 2,
        7203: 3,
        7204: 4,
        7205: 5,
        7206: 6,
        7207: 7,
        7208: 8,
        7209: 9,
        7210: 10,
        7211: 11,
        7212: 12,
        7213: 13,
        7214: 14,
        7215: 15,
        7217: 16,
        7218: 17,
        7219: 18,
    }

    cmap = ListedColormap([plt.cm.tab20(i) for i in range(20)])  # type: ignore[attr-defined]

    ds_root_name = 'Augsburg_data_4_publication'

    zipfilename = f'{ds_root_name}.zip'

    valid_subareas = ('sub_area_1', 'sub_area_2', 'sub_area_3')

    url = 'https://huggingface.co/datasets/torchgeo/mdas/resolve/860226b74269f1cf1bed8ea3c03f571ae701144c/Augsburg_data_4_publication.zip'

    md5 = '7b63c26e3717cb52c6ba47d215f18d5b'

    enmap_rgb_band_idx: ClassVar[list[int]] = [43, 28, 10]
    sentinel_2_rgb_band_idx: ClassVar[list[int]] = [3, 2, 1]
    hyspex_rgb_band_idx: ClassVar[list[int]] = [100, 50, 10]

    def __init__(
        self,
        root: Path = 'data',
        subareas: list[str] = ['sub_area_1'],
        modalities: list[str] = ['3K_RGB', 'HySpex', 'Sentinel_2'],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new MDAS dataset instance.

        Args:
            root: Root directory where the dataset should be stored.
            subareas: The subareas to load. Options are 'sub_area_1', 'sub_area_2', 'sub_area_3'.
            modalities: The modalities to load. Options are '3K_DSM', '3K_RGB', 'HySpex', 'EeteS_EnMAP_10m', 'EeteS_EnMAP_30m', 'EeteS_Sentinel_2_10m', 'Sentinel-2', 'Sentinel-1', 'OSM_label'.
            transforms: A function/transform that takes in a dictionary and returns a transformed version.
            download: if True, download dataset and store it in the root directory
            checksum: If True, check the integrity of the dataset after download.

        Raises:
            AssertionError: If the subareas or modalities are not valid.
            DatasetNotFoundError: If the dataset is not found.
        """
        self.root = root
        self.download = download
        assert all(
            sub in self.valid_subareas for sub in subareas
        ), f'Subareas must be one of {self.valid_subareas}'
        self.subareas = subareas
        assert all(
            mod in self.valid_modalities for mod in modalities
        ), f'Modalities must be one of {self.valid_modalities}'
        self.modalities = modalities
        self.transforms = transforms
        self.checksum = checksum

        self._verify()
        self.files = self._load_files()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files)

    def _load_files(self) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset."""
        files = []
        for subarea in self.subareas:
            subarea_files = {}
            for modality in self.modalities:
                subarea_files[modality] = os.path.join(
                    self.root,
                    self.ds_root_name,
                    subarea,
                    f'{modality}_{self._format_subarea(subarea)}.tif',
                )
            files.append(subarea_files)
        return files

    def _format_subarea(self, subarea: str) -> str:
        """Format the subarea name.

        Args:
            subarea: The subarea string to format.

        Returns:
            formatted subarea string for files
        """
        parts = subarea.split('_')
        return parts[0] + '_' + parts[1] + parts[2]

    def _load_image(self, path: Path) -> Tensor:
        """Load an image from a given path."""
        with rio.open(path) as src:
            img = src.read()
            if img.dtype == np.uint16:
                img = img.astype(np.int32)
            if 'osm_landuse' in str(path):
                img = np.vectorize(self.landuse_mapping.get)(img)

            return torch.from_numpy(img)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the dataset sample at the given index."""
        sample_files = self.files[idx]
        sample: dict[str, Any] = {}
        for modality, path in sample_files.items():
            if 'osm' in modality:
                sample[f'{modality}_mask'] = self._load_image(path).long()
            else:
                sample[f'{modality}_image'] = self._load_image(path)

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # check if each desired modality file exists in specified subarea
        exists = []
        for subarea in self.subareas:
            for modality in self.modalities:
                path = os.path.join(
                    self.root,
                    self.ds_root_name,
                    subarea,
                    f'{modality}_{self._format_subarea(subarea)}.tif',
                )
                if not os.path.exists(path):
                    exists.append(False)
                else:
                    exists.append(True)
        if all(exists):
            return

        # check if zip file downloaded
        if os.path.exists(os.path.join(self.root, self.zipfilename)):
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfilename), self.root)

    def _download(self) -> None:
        """Download the dataset."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfilename,
            md5=self.md5 if self.checksum else None,
        )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by `__getitem__`.
            show_titles: Whether to display titles on the subplots.
            suptitle: An optional super title for the plot.
        """
        ncols = len(sample)
        fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        if ncols == 1:
            axs = [axs]

        for idx, (key, data) in enumerate(sample.items()):
            if key == '3K_RGB_image':
                # Use the first three bands
                img = data[:3].numpy().transpose(1, 2, 0) / 255.0
                axs[idx].imshow(img)
            elif key == '3K_DSM_image':
                img = data.numpy().squeeze(0)
                axs[idx].imshow(img, cmap='gray')
            elif key in ['EeteS_EnMAP_10m_image', 'EeteS_EnMAP_30m_image']:
                # Use specified EnMAP RGB bands
                img = data[self.enmap_rgb_band_idx].numpy().transpose(1, 2, 0) / 10000.0
                axs[idx].imshow(img)
            elif key == 'EeteS_Sentinel_2_10m_image':
                # Use specified Sentinel-2 RGB bands
                img = (
                    data[self.sentinel_2_rgb_band_idx].numpy().transpose(1, 2, 0)
                    / 10000.0
                )
                axs[idx].imshow(img)
            elif key == 'Sentinel_1_image':
                # Use the first band
                img = data[0].numpy().clip(0, 1)
                axs[idx].imshow(img)
            elif key == 'Sentinel_2_image':
                # Use specified Sentinel-2 RGB bands
                img = (
                    data[self.sentinel_2_rgb_band_idx].numpy().transpose(1, 2, 0)
                    / 10000.0
                )
                axs[idx].imshow(img)
            elif key == 'HySpex_image':
                # Use specified HySpex RGB bands
                img = (
                    data[self.hyspex_rgb_band_idx].numpy().transpose(1, 2, 0) / 15000.0
                )
                axs[idx].imshow(img)
            elif key == 'osm_landuse_mask':
                img = data.numpy().squeeze(0)
                im = axs[idx].imshow(img, cmap=self.cmap)
                cbar = plt.colorbar(im, ax=axs[idx], ticks=range(19))
                cbar.ax.set_yticklabels(
                    [self.landuse_class_names[i] for i in range(19)]
                )
            elif key == 'osm_buildings_mask':
                img = data.numpy().squeeze(0)
                axs[idx].imshow(img, cmap='gray')
            elif key == 'osm_water_mask':
                img = data.numpy().squeeze(0)
                axs[idx].imshow(img, cmap='Blues')
            else:
                # Plot the first band for other modalities
                img = data[0].numpy()
                axs[idx].imshow(img)
                axs[idx].axis('off')
                if show_titles:
                    axs[idx].set_title(key)
            axs[idx].axis('off')
            if show_titles:
                axs[idx].set_title(key)

        if suptitle:
            plt.suptitle(suptitle)

        return fig
