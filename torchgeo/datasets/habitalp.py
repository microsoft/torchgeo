# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""HabitAlp2 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from pyproj import CRS
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import GeoDataset, RasterDataset
from .utils import (
    GeoSlice,
    Path,
    download_url,
    draw_semantic_segmentation_masks,
    percentile_normalization,
)


class HabitAlp2RGB(RasterDataset):
    """RGB imagery component for HabitAlp2 dataset."""

    is_image = True
    all_bands = ('R', 'G', 'B')


class HabitAlp2CIR(RasterDataset):
    """CIR (NIR, R, G) imagery component for HabitAlp2 dataset."""

    is_image = True
    all_bands = ('NIR', 'R', 'G')


class HabitAlp2Terrain(RasterDataset):
    """Single-band terrain layer component for HabitAlp2 dataset."""

    is_image = True


class HabitAlp2Mask(RasterDataset):
    """Mask component for HabitAlp2 dataset."""

    is_image = False


class HabitAlp2(GeoDataset):
    """HabitAlp2 dataset for semantic segmentation.

    The `HabitAlp2 <https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0>`__ dataset
    is an ecological habitat mapping dataset for the Gesäuse National Park in Austria,
    covering approximately 154 km² with 30,241 annotated polygons.

    Dataset features:

    * RGB and CIR aerial orthophotos
    * LiDAR-derived terrain layers (DTM, DSM, nDSM, slope, aspect, etc.)
    * 23 habitat classes for semantic segmentation
    * Three temporal periods: 2003, 2013, 2020

    .. note::
       The 2020 epoch is intended as a held-out test period in the original dataset.
       No official train/val/test split is provided — use the ``year`` argument to
       select epochs manually.

    Dataset format:

    * images are multi-band GeoTIFFs
    * masks are single-band GeoTIFFs with class IDs 1-23

    Dataset classes:

    1. Waterbody
    2. Gravel bank, shoal, fluviatile
    3. Erosion area, gully
    4. Debris-covered areas
    5. Rock
    6. Young coniferous (growth, thicket)
    7. Young broad-leaved (growth, thicket)
    8. Coniferous pole timber CC<80
    9. Coniferous pole timber CC>=80
    10. Broad-leaved pole timber
    11. Coniferous mature forest CC<80
    12. Coniferous mature forest CC>=80
    13. Broad-leaved mature forest CC<80
    14. Broad-leaved mature forest CC>=80
    15. Old coniferous forest CC<80
    16. Old coniferous forest CC>=80
    17. Old broad-leaved forest CC<80
    18. Old broad-leaved forest CC>=80
    19. Clearcut areas
    20. Mountain dwarf forest (Krummholz)
    21. Grassland, buffer strip
    22. Alpine grassland, heath
    23. Low importance/small extent

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2511.00073

    .. versionadded:: 0.9.1
    """

    url = 'https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0/resolve/main/'

    valid_years: ClassVar[tuple[str, ...]] = ('2003', '2013', '2020')

    all_bands: ClassVar[tuple[str, ...]] = (
        'R',
        'G',
        'B',
        'NIR',
        'dtm',
        'dsm',
        'ndsm',
        'slope',
        'aspect',
        'curvature',
        'planform_curvature',
        'profile_curvature',
        'roughness_terrain',
        'roughness_canopy',
        'tpi',
        'tri',
    )

    rgb_bands: ClassVar[tuple[str, ...]] = ('R', 'G', 'B')

    terrain_bands: ClassVar[tuple[str, ...]] = (
        'dtm',
        'dsm',
        'ndsm',
        'slope',
        'aspect',
        'curvature',
        'planform_curvature',
        'profile_curvature',
        'roughness_terrain',
        'roughness_canopy',
        'tpi',
        'tri',
    )

    data_files: ClassVar[dict[str, dict[str, str]]] = {
        '2003': {'rgb': 'data_2003/aerial_rgb_2003_2007.tif'},
        '2013': {
            'rgb': 'data_2013/aerial_rgb_2013_2015.tif',
            'cir': 'data_2013/aerial_cir_2013_2015.tif',
            'dtm': 'data_2013/dtm.tif',
            'dsm': 'data_2013/dsm.tif',
            'ndsm': 'data_2013/ndsm.tif',
            'slope': 'data_2013/slope.tif',
            'aspect': 'data_2013/aspect.tif',
            'curvature': 'data_2013/curvature.tif',
            'planform_curvature': 'data_2013/planform_curvature.tif',
            'profile_curvature': 'data_2013/profile_curvature.tif',
            'roughness_terrain': 'data_2013/roughness_terrain.tif',
            'roughness_canopy': 'data_2013/roughness_canopy.tif',
            'tpi': 'data_2013/tpi.tif',
            'tri': 'data_2013/tri.tif',
        },
        '2020': {
            'rgb': 'data_2020/aerial_rgb_2019_2021.tif',
            'cir': 'data_2020/aerial_cir_2019_2021.tif',
            'dtm': 'data_2020/dtm.tif',
            'dsm': 'data_2020/dsm.tif',
            'ndsm': 'data_2020/ndsm.tif',
            'slope': 'data_2020/slope.tif',
            'aspect': 'data_2020/aspect.tif',
            'curvature': 'data_2020/curvature.tif',
            'planform_curvature': 'data_2020/planform_curvature.tif',
            'profile_curvature': 'data_2020/profile_curvature.tif',
            'roughness_terrain': 'data_2020/roughness_terrain.tif',
            'roughness_canopy': 'data_2020/roughness_canopy.tif',
            'tpi': 'data_2020/tpi.tif',
            'tri': 'data_2020/tri.tif',
        },
    }

    mask_files: ClassVar[dict[str, str]] = {
        '2003': 'labels/classes_2003.tif',
        '2013': 'labels/classes_2013.tif',
        '2020': 'labels/classes_2020.tif',
    }

    classes: ClassVar[tuple[str, ...]] = (
        'Background',
        'Waterbody',
        'Gravel bank, shoal, fluviatile',
        'Erosion area, gully',
        'Debris-covered areas',
        'Rock',
        'Young coniferous (growth, thicket)',
        'Young broad-leaved (growth, thicket)',
        'Coniferous pole timber CC<80',
        'Coniferous pole timber CC>=80',
        'Broad-leaved pole timber',
        'Coniferous mature forest CC<80',
        'Coniferous mature forest CC>=80',
        'Broad-leaved mature forest CC<80',
        'Broad-leaved mature forest CC>=80',
        'Old coniferous forest CC<80',
        'Old coniferous forest CC>=80',
        'Old broad-leaved forest CC<80',
        'Old broad-leaved forest CC>=80',
        'Clearcut areas',
        'Mountain dwarf forest (Krummholz)',
        'Grassland, buffer strip',
        'Alpine grassland, heath',
        'Low importance/small extent',
    )

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (0, 0, 0, 255),
        1: (0, 119, 190, 255),
        2: (194, 178, 128, 255),
        3: (139, 90, 43, 255),
        4: (128, 128, 128, 255),
        5: (105, 105, 105, 255),
        6: (144, 238, 144, 255),
        7: (50, 205, 50, 255),
        8: (34, 139, 34, 255),
        9: (0, 100, 0, 255),
        10: (107, 142, 35, 255),
        11: (85, 107, 47, 255),
        12: (0, 128, 0, 255),
        13: (46, 139, 87, 255),
        14: (60, 179, 113, 255),
        15: (32, 178, 170, 255),
        16: (0, 139, 139, 255),
        17: (72, 61, 139, 255),
        18: (75, 0, 130, 255),
        19: (255, 165, 0, 255),
        20: (154, 205, 50, 255),
        21: (255, 255, 0, 255),
        22: (240, 230, 140, 255),
        23: (192, 192, 192, 255),
    }

    def __init__(
        self,
        root: Path = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        year: str = '2013',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new HabitAlp2 dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`CRS` to warp to (defaults to CRS of first file found)
            res: resolution in units of CRS (defaults to resolution of first file)
            year: one of "2003", "2013", or "2020"
            bands: bands to load (defaults to RGB only for 2003, RGB+NIR for 2013/2020)
            transforms: a function/transform that takes input sample and returns
                a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``year`` or ``bands`` arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        year = str(year)
        assert year in self.valid_years, f'year must be one of {self.valid_years}'

        super().__init__()

        self.root = root
        self.year = year
        self.download = download
        self.checksum = checksum
        self.transforms = transforms

        available_bands = self._get_available_bands(year)
        if bands is None:
            if year == '2003':
                bands = ('R', 'G', 'B')
            else:
                bands = ('R', 'G', 'B', 'NIR')

        for band in bands:
            assert band in available_bands, (
                f"Band '{band}' not available for year {year}. "
                f'Available bands: {available_bands}'
            )

        self.bands = bands

        self._verify()

        year_files = self.data_files[year]
        mask_path = os.path.join(root, self.mask_files[year])

        needs_rgb = any(b in ['R', 'G', 'B'] for b in self.bands)
        needs_cir = 'NIR' in self.bands

        image_datasets: list[GeoDataset] = []

        if needs_rgb and 'rgb' in year_files:
            rgb_path = os.path.join(root, year_files['rgb'])
            rgb_ds = HabitAlp2RGB(rgb_path, crs=crs, res=res, cache=cache)
            image_datasets.append(rgb_ds)

        if needs_cir and 'cir' in year_files:
            cir_path = os.path.join(root, year_files['cir'])
            cir_ds = HabitAlp2CIR(
                cir_path, crs=crs, res=res, bands=('NIR',), cache=cache
            )
            image_datasets.append(cir_ds)

        for band in self.terrain_bands:
            if band in self.bands and band in year_files:
                terrain_path = os.path.join(root, year_files[band])
                terrain_ds = HabitAlp2Terrain(
                    terrain_path, crs=crs, res=res, cache=cache
                )
                image_datasets.append(terrain_ds)

        image_ds: GeoDataset = image_datasets[0]
        for ds in image_datasets[1:]:
            image_ds = image_ds & ds

        mask_ds = HabitAlp2Mask(mask_path, crs=crs, res=res, cache=cache)

        self.dataset = image_ds & mask_ds

        self._res = self.dataset.res
        self.index = self.dataset.index

        lc_colors = np.zeros((max(self.cmap.keys()) + 1, 4))
        lc_colors[list(self.cmap.keys())] = list(self.cmap.values())
        self._lc_cmap = ListedColormap(lc_colors[:, :3] / 255)

    def _get_available_bands(self, year: str) -> tuple[str, ...]:
        """Get available bands for a given year.

        Args:
            year: the year to check

        Returns:
            tuple of available band names
        """
        available = []
        year_files = self.data_files[year]

        if 'rgb' in year_files:
            available.extend(['R', 'G', 'B'])
        if 'cir' in year_files:
            available.append('NIR')

        for band in self.terrain_bands:
            if band in year_files:
                available.append(band)

        return tuple(available)

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve image and mask indexed by query.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates
                to index

        Returns:
            sample containing image and mask at that index

        Raises:
            IndexError: if query is not found in the index
        """
        sample = self.dataset[query]
        sample['image'] = sample['image'].float()
        sample['mask'] = sample['mask'].long().squeeze(0)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        year_files = self.data_files[self.year]
        paths = []

        if any(b in ('R', 'G', 'B') for b in self.bands) and 'rgb' in year_files:
            paths.append(os.path.join(self.root, year_files['rgb']))
        if 'NIR' in self.bands and 'cir' in year_files:
            paths.append(os.path.join(self.root, year_files['cir']))
        for band in self.terrain_bands:
            if band in self.bands and band in year_files:
                paths.append(os.path.join(self.root, year_files[band]))
        paths.append(os.path.join(self.root, self.mask_files[self.year]))

        if all(os.path.exists(p) for p in paths):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, f'data_{self.year}'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'labels'), exist_ok=True)

        year_files = self.data_files[self.year]
        needs_rgb = any(b in ['R', 'G', 'B'] for b in self.bands)
        needs_cir = 'NIR' in self.bands

        if needs_rgb and 'rgb' in year_files:
            filepath = os.path.join(self.root, year_files['rgb'])
            if not os.path.exists(filepath):
                download_url(self.url + year_files['rgb'], self.root, year_files['rgb'])

        if needs_cir and 'cir' in year_files:
            filepath = os.path.join(self.root, year_files['cir'])
            if not os.path.exists(filepath):
                download_url(self.url + year_files['cir'], self.root, year_files['cir'])

        for band in self.terrain_bands:
            if band in self.bands and band in year_files:
                filepath = os.path.join(self.root, year_files[band])
                if not os.path.exists(filepath):
                    download_url(
                        self.url + year_files[band], self.root, year_files[band]
                    )

        mask_file = self.mask_files[self.year]
        mask_path = os.path.join(self.root, mask_file)
        if not os.path.exists(mask_path):
            download_url(self.url + mask_file, self.root, mask_file)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2
        showing_predictions = 'prediction' in sample

        if showing_predictions:
            ncols += 1

        fig, axs = plt.subplots(1, ncols, figsize=(ncols * 4, 4))

        image = sample['image'][:3].numpy()
        image = np.transpose(image, (1, 2, 0))
        image = percentile_normalization(image, axis=(0, 1))
        image = np.clip(image, 0, 1)

        mask = sample['mask'].numpy()

        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, vmin=0, vmax=23, cmap=self._lc_cmap, interpolation='none')
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            prediction = sample['prediction'].numpy()
            axs[2].imshow(
                prediction, vmin=0, vmax=23, cmap=self._lc_cmap, interpolation='none'
            )
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class HabitAlp2CD(GeoDataset):
    """HabitAlp2 dataset for change detection.

    The `HabitAlp2 <https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0>`__ dataset
    is an ecological habitat mapping dataset for the Gesäuse National Park in Austria,
    covering approximately 154 km² with 30,241 annotated polygons.

    This class provides access to the change detection task, with bi-temporal image
    pairs and binary change masks.

    Dataset features:

    * RGB and CIR aerial orthophotos
    * LiDAR-derived terrain layers (DTM, DSM, nDSM, slope, aspect, etc.)
    * Binary change detection masks
    * Two temporal pairs: 2003→2013 and 2013→2020

    .. note::
       The 2013→2020 pair is intended as a held-out test pair in the original dataset.
       No official train/val/test split is provided — use the ``pair`` argument to
       select pairs manually.

    Dataset format:

    * images are multi-band GeoTIFFs
    * masks are single-band GeoTIFFs with 0=no change, 1=change

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2511.00073

    .. versionadded:: 0.9.1
    """

    url = HabitAlp2.url

    valid_pairs: ClassVar[tuple[str, ...]] = ('2003_2013', '2013_2020')

    all_bands: ClassVar[tuple[str, ...]] = HabitAlp2.all_bands
    rgb_bands: ClassVar[tuple[str, ...]] = HabitAlp2.rgb_bands
    terrain_bands: ClassVar[tuple[str, ...]] = HabitAlp2.terrain_bands
    data_files: ClassVar[dict[str, dict[str, str]]] = HabitAlp2.data_files

    classes: ClassVar[tuple[str, ...]] = (
        'Class 0',
        'Class 1',
        'Class 2',
        'Class 3',
        'Class 4',
        'Class 5',
        'Class 6',
        'Class 7',
        'Class 8',
    )

    change_mask_files: ClassVar[dict[str, str]] = {
        '2003_2013': 'labels/habitalp_change_2003_2013.tif',
        '2013_2020': 'labels/habitalp_change_2013_2020.tif',
    }

    colormap: ClassVar[tuple[str, ...]] = ('blue',)

    def __init__(
        self,
        root: Path = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        pair: str = '2013_2020',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new HabitAlp2CD dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`CRS` to warp to (defaults to CRS of first file found)
            res: resolution in units of CRS (defaults to resolution of first file)
            pair: one of "2003_2013" or "2013_2020"
            bands: bands to load (defaults to RGB only)
            transforms: a function/transform that takes input sample and returns
                a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``pair`` or ``bands`` arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.

        Note:
            The pair parameter is converted to string to handle YAML configs that may
            parse year pairs (e.g., 2013_2020) as integers (20132020) instead of strings.
            The underscore is reconstructed if needed for 8-digit integers.
        """
        pair = str(pair)
        if '_' not in pair and len(pair) == 8 and pair.isdigit():
            pair = f'{pair[:4]}_{pair[4:]}'
        assert pair in self.valid_pairs, f'pair must be one of {self.valid_pairs}'

        super().__init__()

        self.root = root
        self.pair = pair
        self.download = download
        self.checksum = checksum
        self.transforms = transforms

        self.year1, self.year2 = pair.split('_')

        # Determine available bands (intersection of both years)
        available_bands = self._get_available_bands()

        if bands is None:
            bands = ('R', 'G', 'B')

        for band in bands:
            assert band in available_bands, (
                f"Band '{band}' not available for pair {pair}. "
                f'Available bands: {available_bands}'
            )

        self.bands = bands

        # Compose two HabitAlp2 instances (they handle bands, download, verify)
        self.ds1 = HabitAlp2(
            root,
            crs=crs,
            res=res,
            year=self.year1,
            bands=bands,
            cache=cache,
            download=download,
            checksum=checksum,
        )
        self.ds2 = HabitAlp2(
            root,
            crs=crs,
            res=res,
            year=self.year2,
            bands=bands,
            cache=cache,
            download=download,
            checksum=checksum,
        )

        # Verify and load change mask
        mask_path = os.path.join(root, self.change_mask_files[pair])
        if not os.path.exists(mask_path):
            if not download:
                raise DatasetNotFoundError(self)
            self._download_change_mask()

        self.mask_ds = HabitAlp2Mask(mask_path, crs=crs, res=res, cache=cache)

        # Intersect all three
        self.dataset = self.ds1.dataset & self.ds2.dataset & self.mask_ds

        self._res = self.dataset.res
        self.index = self.dataset.index

    def _get_available_bands(self) -> tuple[str, ...]:
        """Get available bands for the current pair.

        Returns:
            tuple of available band names (intersection of both years)
        """
        year1_files = self.data_files[self.year1]
        year2_files = self.data_files[self.year2]

        bands1: list[str] = []
        bands2: list[str] = []

        for year_files, bands_list in [(year1_files, bands1), (year2_files, bands2)]:
            if 'rgb' in year_files:
                bands_list.extend(['R', 'G', 'B'])
            if 'cir' in year_files:
                bands_list.append('NIR')
            for band in self.terrain_bands:
                if band in year_files:
                    bands_list.append(band)

        return tuple(set(bands1) & set(bands2))

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve bi-temporal image pair and change mask indexed by query.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates
                to index

        Returns:
            sample containing image (T, C, H, W) and mask (1, H, W) at that index

        Raises:
            IndexError: if query is not found in the index
        """
        sample1 = self.ds1[query]
        sample2 = self.ds2[query]
        mask_sample = self.mask_ds[query]

        image1 = sample1['image'].float()
        image2 = sample2['image'].float()
        image = torch.stack([image1, image2], dim=0)

        mask = mask_sample['mask'].long()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[0] != 1:
            mask = mask[:1]

        sample = mask_sample | {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample['image'] = sample['image'].unsqueeze(0)
            sample['mask'] = sample['mask'].unsqueeze(0)
            sample = self.transforms(sample)
            sample['image'] = sample['image'].squeeze(0)
            sample['mask'] = sample['mask'].squeeze(0)

        return sample

    def _download_change_mask(self) -> None:
        """Download the change mask file."""
        os.makedirs(os.path.join(self.root, 'labels'), exist_ok=True)
        mask_file = self.change_mask_files[self.pair]
        download_url(self.url + mask_file, self.root, mask_file)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            alpha: opacity with which to render change mask overlay

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2

        def get_rgb_image(img: Tensor) -> 'np.typing.NDArray[np.uint8]':
            rgb_img = img[:3].float().numpy()
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            rgb_img = percentile_normalization(rgb_img, axis=(0, 1))
            rgb_img = np.clip(rgb_img, 0, 1)
            rgb_img = (rgb_img * 255).astype(np.uint8)
            return rgb_img

        def get_masked(img: Tensor, mask: Tensor) -> 'np.typing.NDArray[np.uint8]':
            rgb_img = get_rgb_image(img)
            array: np.typing.NDArray[np.uint8] = draw_semantic_segmentation_masks(
                torch.from_numpy(np.transpose(rgb_img, (2, 0, 1))),
                mask.squeeze(0),
                alpha=alpha,
                colors=list(self.colormap),
            )
            return array

        image1 = get_masked(sample['image'][0], sample['mask'])
        image2 = get_masked(sample['image'][1], sample['mask'])

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 5, 5))
        axs[0].imshow(image1)
        axs[0].axis('off')
        axs[1].imshow(image2)
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title(f'Pre change ({self.year1})')
            axs[1].set_title(f'Post change ({self.year2})')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
