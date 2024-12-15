# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# FLAIR dataset is realeasd under open license 2.0
# ..... https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf
# ..... https://ignf.github.io/FLAIR/#FLAIR2

"""FLAIR2 dataset."""

import glob
import json
import os
from collections.abc import Callable, Collection, Sequence
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_url, extract_archive


class FLAIR2(NonGeoDataset):
    """FLAIR #2 (The French Land cover from Aerospace ImageRy) dataset.

    The `FLAIR #2 <https://github.com/IGNF/FLAIR-2>` dataset is an extensive dataset from the French National Institute
    of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis.
    The  dataset is sampled countrywide and is composed of over 20 billion annotated pixels of very high resolution aerial
    imagery at 0.2 m spatial resolution, acquired over three years and different months (spatio-temporal domains).

    The FLAIR2 dataset is a dataset for semantic segmentation of aerial images. It contains aerial images, sentinel-2 images and masks for 13 classes.
    The dataset is split into a training and test set.

    Dataset features:

    * over 20 billion annotated pixels
    * aerial imagery
        * 5x512x512
        * 0.2m spatial resolution
        * 5 channels (RGB-NIR-Elevation)
    * Sentinel-2 imagery
        * 10-20m spatial resolution
        * 10 spectral bands
        * snow/cloud masks (with 0-100 probability)
        * multiple time steps (T)
        * Tx10xWxH, T, W, H are variable
    * label (masks)
        * 512x512
        * 13 classes

    Dataset classes:

    0: "building",
    1: "pervious surface",
    2: "impervious surface",
    3: "bare soil",
    4: "water",
    5: "coniferous",
    6: "deciduous",
    7: "brushwood",
    8: "vineyard",
    9: "herbaceous vegetation",
    10: "agricultural land",
    11: "plowed land",
    12: "other"

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2310.13336
    """

    splits: ClassVar[Sequence[str]] = ('train', 'test')

    url_prefix: ClassVar[str] = (
        'https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2'
    )
    md5s: ClassVar[dict[str, str]] = {
        'flair-2_centroids_sp_to_patch': 'f8ba3b176197c254b6c165c97e93c759',
        'flair_aerial_train': '0f575b360800f58add19c08f05e18429',
        'flair_sen_train': '56fbbd465726ea4dfeea02734edd7cc5',
        'flair_labels_train': '80d3cd2ee117a61128faa08cbb842c0c',
        'flair_2_aerial_test': 'a647e0ba7e5345b28c48d7887ee79888',
        'flair_2_sen_test': 'aae8649dbe457620a76269d915d07624',
        'flair_2_labels_test': '394a769ffcb4a783335eecd3f8baef57',
    }

    dir_names: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'images': 'flair_aerial_train',
            'sentinels': 'flair_sen_train',
            'masks': 'flair_labels_train',
        },
        'test': {
            'images': 'flair_2_aerial_test',
            'sentinels': 'flair_2_sen_test',
            'masks': 'flair_2_labels_test',
        },
    }
    globs: ClassVar[dict[str, str]] = {
        'images': 'IMG_*.tif',
        'sentinels': 'SEN2_*{0}.npy',
        'masks': 'MSK_*.tif',
    }
    centroids_file: str = 'flair-2_centroids_sp_to_patch'
    # Sentinel super patch size according to datapaper
    super_patch_size: int = 40

    # Band information
    aerial_rgb_bands: tuple[str, str, str] = ('B01', 'B02', 'B03')
    aerial_all_bands: tuple[str, str, str, str, str] = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
    )
    sentinel_rgb_bands: tuple[str, str, str] = ('B03', 'B02', 'B01')
    # Order refers to 2, 3, 4, 5, 6, 7, 8, 8A, 11, 12 as described in the dataset paper
    sentinel_all_bands: tuple[str, ...] = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B08',
        'B09',
        'B10',
    )

    # Note: the original dataset contains 18 classes, but the dataset paper suggests
    # grouping all classes >13 into "other" class, due to underrepresentation
    classes: tuple[str, ...] = (
        'building',
        'pervious surface',
        'impervious surface',
        'bare soil',
        'water',
        'coniferous',
        'deciduous',
        'brushwood',
        'vineyard',
        'herbaceous vegetation',
        'agricultural land',
        'plowed land',
        'other',
    )
    # Define a colormap for the classes
    cmap = ListedColormap(
        [
            'cyan',  # building
            'lightgray',  # pervious surface
            'darkgray',  # impervious surface
            'saddlebrown',  # bare soil
            'blue',  # water
            'darkgreen',  # coniferous
            'forestgreen',  # deciduous
            'olive',  # brushwood
            'purple',  # vineyard
            'lime',  # herbaceous vegetation
            'yellow',  # agricultural land
            'orange',  # plowed land
            'red',  # other
        ]
    )

    statistics: ClassVar[dict[str, dict[str, dict[str, float]]]] = {
        'train': {
            'B01': {
                'min': 0.0,
                'max': 255.0,
                'mean': 113.77526983072,
                'stdv': 35.40773785,
            },
            'B02': {
                'min': 0.0,
                'max': 255.0,
                'mean': 118.08112962721,
                'stdv': 32.05745038,
            },
            'B03': {
                'min': 0.0,
                'max': 255.0,
                'mean': 109.27393364381,
                'stdv': 30.70572577,
            },
            'B04': {
                'min': 0.0,
                'max': 255.0,
                'mean': 102.36417944851,
                'stdv': 27.09264083,
            },
            'B05': {
                'min': 0.0,
                'max': 255.0,
                'mean': 16.697295721745,
                'stdv': 15.85104252,
            },
        }
    }

    @staticmethod
    def per_band_statistics(
        split: str, bands: Sequence[str] = aerial_all_bands
    ) -> tuple[list[float], ...]:
        """Get statistics (min, max, means, stdvs) for each used band in order.

        Args:
            split: Split for which to get statistics (currently only for train)
            bands: Bands of interest, will be returned in ordered manner. Defaults to all_bands.

        Returns:
            tuple: Filtered, ordered statistics for each band
        """
        assert (
            split in FLAIR2.statistics.keys()
        ), f"Statistics for '{split}' not available; use: '{list(FLAIR2.statistics.keys())}'"
        ordered_bands_statistics = list(
            dict(
                filter(
                    lambda keyval: keyval[0] in bands, FLAIR2.statistics[split].items()
                )
            ).values()
        )
        mins = list(map(lambda dict: dict['min'], ordered_bands_statistics))
        maxs = list(map(lambda dict: dict['max'], ordered_bands_statistics))
        means = list(map(lambda dict: dict['mean'], ordered_bands_statistics))
        stdvs = list(map(lambda dict: dict['stdv'], ordered_bands_statistics))

        return mins, maxs, means, stdvs

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        aerial_bands: Sequence[str] = aerial_all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
        use_sentinel: bool = False,
        sentinel_bands: Sequence[str] = sentinel_all_bands,
    ) -> None:
        """Initialize a new FLAIR2 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: which split to load, one of 'train' or 'test'
            aerial_bands: which bands to load (B01, B02, B03, B04, B05)
            transforms: optional transforms to apply to sample
            download: whether to download the dataset if it is not found
            checksum: whether to verify the dataset using checksums
            use_sentinel: whether to use sentinel data in the dataset # FIXME: sentinel does not work with dataloader due to varying dimensions
            sentinel_bands: which bands to load from sentinel data (B01, B02, ..., B10)

        Raises:
            DatasetNotFoundError

        ..versionadded:: 0.7
        """
        assert (
            split in self.splits
        ), f"Split '{split}' not in supported splits: '{self.splits}'"

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.aerial_bands = aerial_bands
        self.use_sentinel = use_sentinel
        self.sentinel_bands = sentinel_bands

        self._verify()
        self.centroids = self._load_centroids(self.centroids_file)

        self.files = self._load_files()

    def get_num_bands(self, include_sentinel_bands: bool = False) -> int:
        """Return the number of bands in the dataset.

        Returns:
            int: number of bands in the initialized dataset (might vary from all_bands)
        """
        return (
            len(self.aerial_bands)
            if not include_sentinel_bands
            else len(self.aerial_bands) + len(self.sentinel_bands)
        )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and mask at that index with image of dimension `get_num_bands()`x512x512,
            sentinel image of dimension Tx10x512x512 and mask of dimension 512x512
        """
        aerial_fn = self.files[index]['image']
        sentinel_fn = cast(dict[str, str], self.files[index]['sentinel'])
        mask_fn = self.files[index]['mask']

        aerial = self._load_image(cast(Path, aerial_fn))
        mask = self._load_target(cast(Path, mask_fn))

        sample: dict[str, Any] = {'image': aerial, 'mask': mask}

        if self.use_sentinel:
            img_id = os.path.basename(cast(Path, aerial_fn))
            centroid_x_y = cast(tuple[int, int], self.centroids[img_id])
            crop_indices = self._get_crop_indices(centroid_x_y)
            sentinel_data = self._load_sentinel(cast(Path, sentinel_fn['data']))
            sentinel_mask = self._load_sentinel(
                cast(Path, sentinel_fn['snow_cloud_mask'])
            )
            sample['sentinel_data'] = sentinel_data
            sample['snow_cloud_mask'] = sentinel_mask
            sample['crop_indices'] = crop_indices

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)

    def _load_centroids(self, filename: str) -> dict[str, list[int]]:
        """Load centroids for mapping sentinel super-areas to aerial patches.

        For detailed information on super-patches, see p.4f of datapaper and  `flair-2_centroids_sp_to_patch.json`.
        CAUTION: centroids are stored as y, x

        Args:
            filename: name of the file containing centroids

        Returns:
            dict: centroids for super-patches
        """
        with open(os.path.join(self.root, f'{filename}.json')) as f:
            centroids = json.load(f)
            return cast(dict[str, list[int]], centroids)

    def _get_crop_indices(self, centroid: tuple[int, int]) -> tuple[slice, slice]:
        """Return indices to crop a super-patch from sentinel data based centroid coordinates.

        For detailed information on super-patches, see p.4f of datapaper and `flair-2_centroids_sp_to_patch.json`.

        Args:
            centroid: centroid coordinates

        Returns:
            tuple[slice, slice]: crop indices for sentinel data
        """
        y, x = centroid
        eigth_size = self.super_patch_size // 8

        return (
            slice(x - eigth_size, x + eigth_size),
            slice(y - eigth_size, y + eigth_size),
        )

    def _load_files(self) -> list[dict[str, Collection[str]]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image, masks
        """
        images = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]['images'],
                    '**',
                    self.globs['images'],
                ),
                recursive=True,
            )
        )

        sentinels_data = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]['sentinels'],
                    '**',
                    self.globs['sentinels'].format('_data'),
                ),
                recursive=True,
            )
        )
        sentinels_mask = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]['sentinels'],
                    '**',
                    self.globs['sentinels'].format('_masks'),
                ),
                recursive=True,
            )
        )
        sentinels = [
            {'data': data, 'snow_cloud_mask': mask}
            for data, mask in zip(sentinels_data, sentinels_mask)
        ]

        masks = sorted(
            glob.glob(
                os.path.join(
                    self.root,
                    self.dir_names[self.split]['masks'],
                    '**',
                    self.globs['masks'],
                ),
                recursive=True,
            )
        )

        # One sentinel image might contain multiple aerial images, thus we need to match them
        # without assuming a 1:1 mapping
        sentinel_lookup = {'/'.join(s['data'].split('/')[-4:-2]): s for s in sentinels}
        files = [
            dict(
                image=image,
                sentinel=sentinel_lookup['/'.join(image.split('/')[-4:-2])],
                mask=mask,
            )
            for image, mask in zip(images, masks)
        ]

        return files

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            tensor: the loaded image
        """
        with rasterio.open(path) as f:
            array: np.typing.NDArray[np.int_] = f.read()
            tensor = torch.from_numpy(array).float() / 255

        # Extract the bands of interest
        tensor = tensor[[int(band[-2:]) - 1 for band in self.aerial_bands]]

        if 'B05' in self.aerial_bands:
            # Height channel will always be the last dimension
            tensor[-1] = torch.div(tensor[-1], 5)

        return tensor

    def _load_sentinel(self, path: Path) -> Tensor:
        """Load a sentinel array.

        Args:
            path: path to sentinel img (data or snow cloud mask)

        Returns:
            tensor: image as tensors of shape TxCxHxW (time, channels, height, width)
        """
        tensor = torch.from_numpy(np.load(path)).float()
        # Extract sentinel bands of interest
        return tensor[:, [int(band[-2:]) - 1 for band in self.sentinel_bands]]

    def _load_target(self, path: Path) -> Tensor:
        """Load a single mask corresponding to image.

        Args:
            path: path to the mask

        Returns:
            tensor: the mask of the image
        """
        with rasterio.open(path) as f:
            array: np.typing.NDArray[np.int_] = f.read(1)
            tensor = torch.from_numpy(array).long()
            # According to datapaper, the dataset contains classes beyond 13
            # however, those are grouped into a single "other" class
            # Rescale the classes to be in the range [0, 12] by subtracting 1
            torch.clamp(tensor - 1, 0, len(self.classes) - 1, out=tensor)

        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if centroids metadata file or zip is present
        # Naming inconsistencies in download url and file name (flair_2_centroids_sp_to_patch.zip vs. flair-2_centroids_sp_to_patch.json)
        if not os.path.isfile(os.path.join(self.root, f'{self.centroids_file}.json')):
            if not os.path.isfile(
                os.path.join(self.root, f'{self.centroids_file}.zip'.replace('-', '_'))
            ):
                if not self.download:
                    raise DatasetNotFoundError(self)
                self._download(self.centroids_file.replace('-', '_'))
            self._extract(self.centroids_file.replace('-', '_'))

        # Files to be extracted
        to_extract: list[str] = []

        # Check if dataset files (by checking glob) are present already
        for train_or_test, dir_name in self.dir_names[self.split].items():
            downloaded_path = os.path.join(self.root, dir_name)
            if not os.path.isdir(downloaded_path):
                to_extract.append(dir_name)
                continue

            files_glob = os.path.join(downloaded_path, '**', self.globs[train_or_test])
            # Format the glob of sentinel `SEN_{0}.npy` to match the actual file name
            # in other cases, where it is not a format string, the glob will be the same
            if not glob.glob(files_glob.format(''), recursive=True):
                to_extract.append(dir_name)

        if not to_extract:
            print('Data has been downloaded and extracted already...')
            return

        # Deepcopy files to be extracted and check wether the zip is downloaded
        to_download = list(map(lambda x: x, to_extract))
        for candidate in to_extract:
            zipfile = os.path.join(self.root, f'{candidate}.zip')
            if glob.glob(zipfile):
                print(f'Extracting: {candidate}')
                self._extract(candidate)
                to_download.remove(candidate)

        # Check if there are still files to download
        if not to_download:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        print('Downloading: ', to_download)
        for candidate in to_download:
            self._download(candidate)
            self._extract(candidate)

    def _download(self, url: str, suffix: str = '.zip') -> None:
        """Download the dataset."""
        download_url(
            os.path.join(self.url_prefix, f'{url}{suffix}'),
            self.root,
            md5=self.md5s.get(url, None) if self.checksum else None,
        )
        # FIXME: Why is download_url not checking integrity (tests run through)?
        assert check_integrity(
            os.path.join(self.root, f'{url}{suffix}'),
            self.md5s.get(url, None) if self.checksum else None,
        )

    def _extract(self, file_path: str) -> None:
        """Extract the dataset."""
        assert isinstance(self.root, str | os.PathLike)
        zipfile = os.path.join(self.root, f'{file_path}.zip')
        extract_archive(zipfile)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """

        def normalize_plot(tensor: Tensor) -> Tensor:
            """Normalize the plot."""
            return (tensor - tensor.min()) / (tensor.max() - tensor.min())

        rgb_indices = [
            self.aerial_all_bands.index(band) for band in self.aerial_rgb_bands
        ]
        # Check if RGB bands are present in self.bands
        if not all([band in self.aerial_bands for band in self.aerial_rgb_bands]):
            raise RGBBandsMissingError()

        # Stretch to the full range of the image
        image = normalize_plot(sample['image'][rgb_indices].permute(1, 2, 0))

        # Get elevation and NIR, R, G if available
        elevation, nir_r_g = None, None
        if 'B05' in self.aerial_bands:
            elevation = sample['image'][self.aerial_bands.index('B05')]
        if 'B04' in self.aerial_bands:
            nir_r_g_indices = [
                self.aerial_bands.index('B04'),
                rgb_indices[0],
                rgb_indices[1],
            ]
            nir_r_g = normalize_plot(sample['image'][nir_r_g_indices].permute(1, 2, 0))

        # Sentinel is a time-series, i.e. use [0]->T=0
        sentinel = None
        if self.use_sentinel:
            crop_indices = cast(Sequence[slice], sample['crop_indices'])
            sentinel = sample['sentinel_data']
            sentinel = sentinel[0]
            sentinel = normalize_plot(sentinel[[2, 1, 0], :, :].permute(1, 2, 0))

        # Obtain mask and predictions if available
        mask = sample['mask'].numpy().astype('uint8').squeeze()

        showing_predictions = 'prediction' in sample
        predictions = None
        if showing_predictions:
            predictions = sample['prediction'].numpy().astype('uint8').squeeze()

        # Remove none available plots
        plot_candidates = zip(
            [
                'image (R+G+B)',
                'NIR+R+G',
                'elevation',
                'sentinel',
                'predictions',
                'mask',
            ],
            [image, nir_r_g, elevation, sentinel, predictions, mask],
        )
        plots = [plot for plot in plot_candidates if plot[1] is not None]

        num_panels = len(plots)

        kwargs = {
            'cmap': self.cmap,
            'vmin': 0,
            'vmax': len(self.classes),
            'interpolation': 'none',
        }
        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))

        for plot in plots:
            im_kwargs = (
                kwargs.copy() if plot[0] == 'mask' or plot[0] == 'predictions' else {}
            )
            if plot[0] == 'sentinel':
                axs[0].add_patch(
                    Rectangle(
                        (crop_indices[0].start, crop_indices[1].start),
                        crop_indices[0].stop - crop_indices[0].start,
                        crop_indices[1].stop - crop_indices[1].start,
                        fill=False,
                        edgecolor='red',
                        lw=0.5,
                    )
                )

            axs[0].imshow(plot[1], **im_kwargs)
            axs[0].axis('off')
            if show_titles:
                axs[0].set_title(plot[0])

            axs = axs[1:]

        if suptitle is not None:
            plt.suptitle(suptitle)

        # Create a legend for the mask
        if 'mask' in [plot[0] for plot in plots]:
            # Create a legend with class names
            legend_elements = [
                Patch(facecolor=self.cmap(i), edgecolor='k', label=cls)
                for i, cls in enumerate(self.classes)
            ]
            fig.legend(
                handles=legend_elements,
                loc='upper left',
                bbox_to_anchor=(0.92, 0.85),
                fontsize='large',
            )

        return fig


class FLAIR2Toy(FLAIR2):
    """FLAIR #2 (The French Land cover from Aerospace ImageRy) dataset.

    Toy Version of the dataset. For further information refer to the FLAIR2 dataset.
    """

    md5s: ClassVar[dict[str, str]] = {
        'flair_2_toy_dataset': 'ffde17f275fc258dce19331b5e17e10a'
    }

    dir_names: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'images': 'flair_2_toy_dataset/flair_2_toy_aerial_train',
            'sentinels': 'flair_2_toy_dataset/flair_2_toy_sen_train',
            'masks': 'flair_2_toy_dataset/flair_2_toy_labels_train',
        },
        'test': {
            'images': 'flair_2_toy_dataset/flair_2_toy_aerial_test',
            'sentinels': 'flair_2_toy_dataset/flair_2_toy_sen_test',
            'masks': 'flair_2_toy_dataset/flair_2_toy_labels_test',
        },
    }
    centroids_file: str = 'flair_2_toy_dataset/flair-2_centroids_sp_to_patch'

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        aerial_bands: Sequence[str] = FLAIR2.aerial_all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
        use_sentinel: bool = False,
        sentinel_bands: Sequence[str] = FLAIR2.sentinel_all_bands
    ) -> None:
        """Initialize a new FLAIR2Toy dataset instance.

        Args:
            root: root directory where dataset can be found
            split: which split to load, one of 'train' or 'test'
            aerial_bands: which bands to load (B01, B02, B03, B04, B05)
            transforms: optional transforms to apply to sample
            download: whether to download the dataset if it is not found
            checksum: whether to verify the dataset using checksums
            use_sentinel: whether to use sentinel data in the dataset # FIXME: sentinel does not work with dataloader due to varying dimensions
            sentinel_bands: which bands to load from sentinel data (B01, B02, ..., B10)

        Raises:
            DatasetNotFoundError

        ..versionadded:: 0.7
        """
        print('-' * 80)
        print('WARNING: Using toy dataset.')
        print('This dataset should be used for testing purposes only.')
        print(
            'Disabling use_toy-flag when initializing the dataset will initialize the full dataset.'
        )
        print('-' * 80)
        super().__init__(
            root, split, aerial_bands, transforms, download, checksum, use_sentinel, sentinel_bands
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        if os.path.isdir(os.path.join(self.root, 'flair_2_toy_dataset')):
            print(os.path.join(self.root, 'flair_2_toy_dataset'))
            print('Toy dataset downloaded and extracted already...')
            return

        if os.path.isfile(os.path.join(self.root, 'flair_2_toy_dataset.zip')):
            print('Extracting toy dataset...')
            self._extract()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download('flair_2_toy_dataset')
        self._extract()

    def _download(self, url: str, suffix: str = '.zip') -> None:
        """Download the dataset."""
        download_url(
            os.path.join(self.url_prefix, f'{url}{suffix}'),
            self.root,
            md5=self.md5s.get(url, None) if self.checksum else None,
        )
        # FIXME: Why is download_url not checking integrity (tests run through)?
        # assert check_integrity(os.path.join(self.root, f"{url}{suffix}"), self.md5s.get(url, None) if self.checksum else None)

    def _extract(self, file_name: str = 'flair_2_toy_dataset.zip') -> None:
        """Extract the dataset."""
        assert isinstance(self.root, str | os.PathLike)
        assert os.path.isfile(os.path.join(self.root, file_name))
        zipfile = os.path.join(self.root, file_name)
        extract_archive(zipfile)
