# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""HabitAlp2 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import GeoDataset, RasterDataset
from .utils import GeoSlice, Path, Sample, download_url, quantile_normalization

# These are private implementation helpers. RasterDataset's `separate_files`
# mechanism assumes all band files share the same directory, which is not the
# case here (RGB, CIR, per-band terrain, and mask files span different
# subdirectories). We therefore compose separate RasterDataset instances and
# intersect them with the & operator.


class _HabitAlp2RGB(RasterDataset):
    """RGB imagery component for HabitAlp2 dataset."""

    is_image = True
    all_bands = ('R', 'G', 'B')


class _HabitAlp2CIR(RasterDataset):
    """CIR (NIR, R, G) imagery component for HabitAlp2 dataset."""

    is_image = True
    all_bands = ('NIR', 'R', 'G')


class _HabitAlp2Terrain(RasterDataset):
    """Single-band terrain layer component for HabitAlp2 dataset."""

    is_image = True


class _HabitAlp2Mask(RasterDataset):
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
       No official train/val/test split is provided; use the ``year`` argument to
       select epochs manually.

    Dataset format:

    * images are multi-band GeoTIFFs
    * masks are single-band GeoTIFFs with class IDs 1-23

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2511.00073

    .. versionadded:: 0.10
    """

    url = 'https://huggingface.co/datasets/JR-DIGITAL/habitalp2.0/resolve/df01fe8ae22df182e7bf1c2e3e713dcfd92c0c81/'

    valid_years = ('2003', '2013', '2020')

    all_bands = (
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

    rgb_bands = ('R', 'G', 'B')

    terrain_bands = (
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

    classes = (
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
            crs: :term:`coordinate reference system (CRS)` to warp to
                (passed through to internal RasterDataset instances; HabitAlp2
                extends GeoDataset directly so crs/res must be forwarded
                explicitly)
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

        rgb_requested = tuple(b for b in self.bands if b in ('R', 'G', 'B'))
        needs_rgb = len(rgb_requested) > 0
        needs_cir = 'NIR' in self.bands

        image_datasets: list[GeoDataset] = []

        if needs_rgb:
            rgb_path = os.path.join(root, year_files['rgb'])
            rgb_ds = _HabitAlp2RGB(
                rgb_path, crs=crs, res=res, bands=rgb_requested, cache=cache
            )
            image_datasets.append(rgb_ds)

        if needs_cir:
            cir_path = os.path.join(root, year_files['cir'])
            cir_ds = _HabitAlp2CIR(
                cir_path, crs=crs, res=res, bands=('NIR',), cache=cache
            )
            image_datasets.append(cir_ds)

        for band in self.terrain_bands:
            if band in self.bands:
                terrain_path = os.path.join(root, year_files[band])
                terrain_ds = _HabitAlp2Terrain(
                    terrain_path, crs=crs, res=res, cache=cache
                )
                image_datasets.append(terrain_ds)

        image_ds: GeoDataset = image_datasets[0]
        for ds in image_datasets[1:]:
            image_ds = image_ds & ds

        mask_ds = _HabitAlp2Mask(mask_path, crs=crs, res=res, cache=cache)

        self.dataset = image_ds & mask_ds

        # Canonicalize band order to match actual tensor layout
        canonical: list[str] = []
        canonical.extend(rgb_requested)
        if needs_cir:
            canonical.append('NIR')
        canonical.extend(b for b in self.terrain_bands if b in self.bands)
        self.bands = tuple(canonical)

        self._res = self.dataset.res
        self.index = self.dataset.index

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

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve image and mask indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates
                to index

        Returns:
            sample containing image and mask at that index

        Raises:
            IndexError: if index is not found in the index
        """
        sample = self.dataset[index]
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
        needs_rgb = any(b in ('R', 'G', 'B') for b in self.bands)
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
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
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
        if not all(b in self.bands for b in self.rgb_bands):
            raise RGBBandsMissingError()

        ncols = 2
        showing_predictions = 'prediction' in sample

        if showing_predictions:
            ncols += 1

        fig, axs = plt.subplots(1, ncols, figsize=(ncols * 4, 4))

        image = quantile_normalization(sample['image'][:3].float())
        image = image.permute(1, 2, 0).numpy()

        mask = sample['mask'].numpy()

        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(
            mask,
            vmin=0,
            vmax=23,
            cmap=plt.colormaps.get_cmap('tab20').resampled(24),
            interpolation='none',
        )
        axs[1].axis('off')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            prediction = sample['prediction'].numpy()
            axs[2].imshow(
                prediction,
                vmin=0,
                vmax=23,
                cmap=plt.colormaps.get_cmap('tab20').resampled(24),
                interpolation='none',
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
    pairs and either binary or multiclass change masks.

    Dataset features:

    * RGB and CIR aerial orthophotos
    * LiDAR-derived terrain layers (DTM, DSM, nDSM, slope, aspect, etc.)
    * Change detection masks with 9 change classes (0=no change, 1-8=change types)
    * Two temporal pairs: 2003→2013 and 2013→2020

    .. note::
       The 2013→2020 pair is intended as a held-out test pair in the original dataset.
       No official train/val/test split is provided; use the ``pair`` argument to
       select pairs manually.

    Dataset format:

    * images are multi-band GeoTIFFs
    * masks are single-band GeoTIFFs with class IDs 0-8 (binary task binarizes to 0/1)

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2511.00073

    .. versionadded:: 0.10
    """

    url = HabitAlp2.url

    valid_pairs = ('2003_2013', '2013_2020')
    valid_tasks = ('binary', 'multiclass')

    all_bands = HabitAlp2.all_bands
    rgb_bands = HabitAlp2.rgb_bands
    terrain_bands = HabitAlp2.terrain_bands
    data_files: ClassVar[dict[str, dict[str, str]]] = HabitAlp2.data_files

    change_mask_files: ClassVar[dict[str, str]] = {
        '2003_2013': 'labels/habitalp_change_2003_2013.tif',
        '2013_2020': 'labels/habitalp_change_2013_2020.tif',
    }

    multiclass_classes = (
        'No change',
        'Mature Tree Density Loss',
        'Clearcut Loss',
        'Forest Density Gain',
        'Other Transition',
        'Forest Stage Progression',
        'Early Forest Establishment',
        'Forest Setback Young Loss',
        'Old Growth Density Loss',
    )

    def __init__(
        self,
        root: Path = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        pair: str = '2013_2020',
        task: str = 'binary',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new HabitAlp2CD dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (passed through to internal RasterDataset instances; HabitAlp2CD
                extends GeoDataset directly so crs/res must be forwarded
                explicitly)
            res: resolution in units of CRS (defaults to resolution of first file)
            pair: one of "2003_2013" or "2013_2020"
            task: one of "binary" (mask binarized to 0/1) or "multiclass" (mask with
                original 0-8 change class IDs)
            bands: bands to load (defaults to RGB only)
            transforms: a function/transform that takes input sample and returns
                a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``pair``, ``task``, or ``bands`` arguments are invalid
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
        assert task in self.valid_tasks, f'task must be one of {self.valid_tasks}'

        super().__init__()

        self.root = root
        self.pair = pair
        self.task = task
        self.download = download
        self.checksum = checksum
        self.transforms = transforms

        self.year1, self.year2 = pair.split('_')

        if task == 'binary':
            self.classes: tuple[str, ...] = ('no change', 'change')
        else:
            self.classes = self.multiclass_classes

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
        self.bands = self.ds1.bands

        mask_path = os.path.join(root, self.change_mask_files[pair])
        if not os.path.exists(mask_path):
            if not download:
                raise DatasetNotFoundError(self)
            self._download_change_mask()
        self.mask_ds = _HabitAlp2Mask(mask_path, crs=crs, res=res, cache=cache)

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

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve bi-temporal image pair and change mask indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates
                to index

        Returns:
            sample containing image (2, C, H, W) and mask (1, H, W); for
            ``task="binary"`` mask values are 0/1, for ``task="multiclass"``
            mask values are 0-8 change class IDs

        Raises:
            IndexError: if index is not found in the index
        """
        sample1 = self.ds1[index]
        sample2 = self.ds2[index]
        mask_sample = self.mask_ds[index]

        image = torch.stack([sample1['image'].float(), sample2['image'].float()], dim=0)
        mask = mask_sample['mask'].long().unsqueeze(0)
        if self.task == 'binary':
            mask = (mask > 0).long()

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
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
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
        if not all(b in self.bands for b in self.rgb_bands):
            raise RGBBandsMissingError()

        def get_rgb(img: Tensor) -> Tensor:
            return quantile_normalization(img[:3].float()).permute(1, 2, 0)

        ncols = 3
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            ncols += 1

        if self.task == 'binary':
            fig, axs = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
            axs[0].imshow(get_rgb(sample['image'][0]))
            axs[1].imshow(get_rgb(sample['image'][1]))
            axs[2].imshow(
                sample['mask'].squeeze(0).numpy(),
                cmap='gray',
                interpolation='none',
                vmin=0,
                vmax=1,
            )
            if show_titles:
                axs[0].set_title(f'Pre change ({self.year1})')
                axs[1].set_title(f'Post change ({self.year2})')
                axs[2].set_title('Change mask')
            if showing_predictions:
                axs[3].imshow(
                    sample['prediction'].squeeze(0).numpy(),
                    cmap='gray',
                    interpolation='none',
                    vmin=0,
                    vmax=1,
                )
                if show_titles:
                    axs[3].set_title('Prediction')
        else:
            fig, axs = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
            axs[0].imshow(get_rgb(sample['image'][0]))
            axs[1].imshow(get_rgb(sample['image'][1]))
            axs[2].imshow(
                sample['mask'].squeeze(0).numpy(),
                vmin=0,
                vmax=8,
                cmap='tab10',
                interpolation='none',
            )
            if show_titles:
                axs[0].set_title(f'Image ({self.year1})')
                axs[1].set_title(f'Image ({self.year2})')
                axs[2].set_title('Change mask')
            if showing_predictions:
                axs[3].imshow(
                    sample['prediction'].squeeze(0).numpy(),
                    vmin=0,
                    vmax=8,
                    cmap='tab10',
                    interpolation='none',
                )
                if show_titles:
                    axs[3].set_title('Prediction')

        for ax in axs:
            ax.axis('off')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
