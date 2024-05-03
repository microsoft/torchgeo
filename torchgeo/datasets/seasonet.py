# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SeasoNet dataset."""

import os
import random
from collections.abc import Callable, Collection, Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    DatasetNotFoundError,
    RGBBandsMissingError,
    download_url,
    extract_archive,
    percentile_normalization,
)


class SeasoNet(NonGeoDataset):
    """SeasoNet Semantic Segmentation dataset.

    The `SeasoNet <https://doi.org/10.5281/zenodo.5850306>`__ dataset consists of
    1,759,830 multi-spectral Sentinel-2 image patches, taken from 519,547 unique
    locations, covering the whole surface area of Germany. Annotations are
    provided in the form of pixel-level land cover and land usage segmentation
    masks from the German land cover model LBM-DE2018 with land cover classes
    based on the CORINE Land Cover database (CLC) 2018. The set is split into
    two overlapping grids, consisting of roughly 880,000 samples each, which are
    shifted by half the patch size in both dimensions. The images in each of the
    both grids themselves do not overlap.

    Dataset format:

    * images are 16-bit GeoTiffs, split into seperate files based on resolution
    * images include 12 spectral bands with 10, 20 and 60 m per pixel resolutions
    * masks are single-channel 8-bit GeoTiffs

    Dataset classes:

    0. Continuous urban fabric
    1. Discontinuous urban fabric
    2. Industrial or commercial units
    3. Road and rail networks and associated land
    4. Port areas
    5. Airports
    6. Mineral extraction sites
    7. Dump sites
    8. Construction sites
    9. Green urban areas
    10. Sport and leisure facilities
    11. Non-irrigated arable land
    12. Vineyards
    13. Fruit trees and berry plantations
    14. Pastures
    15. Broad-leaved forest
    16. Coniferous forest
    17. Mixed forest
    18. Natural grasslands
    19. Moors and heathland
    20. Transitional woodland/shrub
    21. Beaches, dunes, sands
    22. Bare rock
    23. Sparsely vegetated areas
    24. Inland marshes
    25. Peat bogs
    26. Salt marshes
    27. Intertidal flats
    28. Water courses
    29. Water bodies
    30. Coastal lagoons
    31. Estuaries
    32. Sea and ocean

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS46834.2022.9884079

    .. versionadded:: 0.5
    """

    metadata = [
        {
            'name': 'spring',
            'ext': '.zip',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/spring.zip',  # noqa: E501
            'md5': 'de4cdba7b6196aff624073991b187561',
        },
        {
            'name': 'summer',
            'ext': '.zip',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/summer.zip',  # noqa: E501
            'md5': '6a54d4e134d27ae4eb03f180ee100550',
        },
        {
            'name': 'fall',
            'ext': '.zip',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/fall.zip',  # noqa: E501
            'md5': '5f94920fe41a63c6bfbab7295f7d6b95',
        },
        {
            'name': 'winter',
            'ext': '.zip',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/winter.zip',  # noqa: E501
            'md5': 'dc5e3e09e52ab5c72421b1e3186c9a48',
        },
        {
            'name': 'snow',
            'ext': '.zip',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/snow.zip',  # noqa: E501
            'md5': 'e1b300994143f99ebb03f51d6ab1cbe6',
        },
        {
            'name': 'splits',
            'ext': '.zip',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/splits.zip',  # noqa: E501
            'md5': 'e4ec4a18bc4efc828f0944a7cf4d5fed',
        },
        {
            'name': 'meta.csv',
            'ext': '',
            'url': 'https://zenodo.org/api/files/e2288446-9ee8-4b2e-ae76-cd80366a40e1/meta.csv',  # noqa: E501
            'md5': '43ea07974936a6bf47d989c32e16afe7',
        },
    ]
    classes = [
        'Continuous urban fabric',
        'Discontinuous urban fabric',
        'Industrial or commercial units',
        'Road and rail networks and associated land',
        'Port areas',
        'Airports',
        'Mineral extraction sites',
        'Dump sites',
        'Construction sites',
        'Green urban areas',
        'Sport and leisure facilities',
        'Non-irrigated arable land',
        'Vineyards',
        'Fruit trees and berry plantations',
        'Pastures',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grasslands',
        'Moors and heathland',
        'Transitional woodland/shrub',
        'Beaches, dunes, sands',
        'Bare rock',
        'Sparsely vegetated areas',
        'Inland marshes',
        'Peat bogs',
        'Salt marshes',
        'Intertidal flats',
        'Water courses',
        'Water bodies',
        'Coastal lagoons',
        'Estuaries',
        'Sea and ocean',
    ]
    all_seasons = {'Spring', 'Summer', 'Fall', 'Winter', 'Snow'}
    all_bands = ('10m_RGB', '10m_IR', '20m', '60m')
    band_nums = {'10m_RGB': 3, '10m_IR': 1, '20m': 6, '60m': 2}
    splits = ['train', 'val', 'test']
    cmap = {
        0: (230, 000, 77, 255),
        1: (255, 000, 000, 255),
        2: (204, 77, 242, 255),
        3: (204, 000, 000, 255),
        4: (230, 204, 204, 255),
        5: (230, 204, 230, 255),
        6: (166, 000, 204, 255),
        7: (166, 77, 000, 255),
        8: (255, 77, 255, 255),
        9: (255, 166, 255, 255),
        10: (255, 230, 255, 255),
        11: (255, 255, 168, 255),
        12: (230, 128, 000, 255),
        13: (242, 166, 77, 255),
        14: (230, 230, 77, 255),
        15: (128, 255, 000, 255),
        16: (000, 166, 000, 255),
        17: (77, 255, 000, 255),
        18: (204, 242, 77, 255),
        19: (166, 255, 128, 255),
        20: (166, 242, 000, 255),
        21: (230, 230, 230, 255),
        22: (204, 204, 204, 255),
        23: (204, 255, 204, 255),
        24: (166, 166, 255, 255),
        25: (77, 77, 255, 255),
        26: (204, 204, 255, 255),
        27: (166, 166, 230, 255),
        28: (000, 204, 242, 255),
        29: (128, 242, 230, 255),
        30: (000, 255, 166, 255),
        31: (166, 255, 230, 255),
        32: (230, 242, 255, 255),
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        seasons: Collection[str] = all_seasons,
        bands: Iterable[str] = all_bands,
        grids: Iterable[int] = [1, 2],
        concat_seasons: int = 1,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SeasoNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val" or "test"
            seasons: list of seasons to load
            bands: list of bands to load
            grids: which of the overlapping grids to load
            concat_seasons: number of seasonal images to return per sample.
                if 1, each seasonal image is returned as its own sample,
                otherwise seasonal images are randomly picked from the seasons
                specified in ``seasons`` and returned as stacked tensors
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        assert set(seasons) <= self.all_seasons
        assert set(bands) <= set(self.all_bands)
        assert set(grids) <= {1, 2}
        assert concat_seasons in range(1, len(seasons) + 1)

        self.root = root
        self.bands = bands
        self.concat_seasons = concat_seasons
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.channels = 0
        for b in bands:
            self.channels += self.band_nums[b]

        csv = pd.read_csv(os.path.join(self.root, 'meta.csv'), index_col='Index')

        if split is not None:
            # Filter entries by split
            split_csv = pd.read_csv(
                os.path.join(self.root, f'splits/{split}.csv'), header=None
            )[0]
            csv = csv.iloc[split_csv]

        # Filter entries by grids and seasons
        csv = csv[csv['Grid'].isin(grids)]
        csv = csv[csv['Season'].isin(seasons)]

        # Replace relative data paths with absolute paths
        csv['Path'] = csv['Path'].apply(
            lambda p: [os.path.join(self.root, p, os.path.basename(p))]
        )

        if self.concat_seasons > 1:
            # Group entries by location
            self.files = csv.groupby(['Latitude', 'Longitude'])
            self.files = self.files['Path'].agg('sum')

            # Remove entries with less than concat_seasons available seasons
            self.files = self.files[
                self.files.apply(lambda d: len(d) >= self.concat_seasons)
            ]
        else:
            self.files = csv['Path']

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample at that index containing the image with shape SCxHxW
            and the mask with shape HxW, where ``S = self.concat_seasons``
        """
        image = self._load_image(index)
        mask = self._load_target(index)
        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load image(s) for a single location.

        Args:
            index: index to return

        Returns:
            the stacked seasonal images
        """
        paths = self.files.iloc[index]
        if self.concat_seasons > 1:
            paths = random.sample(paths, self.concat_seasons)
        tensor = torch.empty(self.concat_seasons * self.channels, *self.image_size)
        for img_idx, path in enumerate(paths):
            bnd_idx = 0
            for band in self.bands:
                with rasterio.open(f'{path}_{band}.tif') as f:
                    array = f.read(
                        out_shape=[f.count] + list(self.image_size),
                        out_dtype='int32',
                        resampling=Resampling.bilinear,
                    )
                image = torch.from_numpy(array).float()
                c = img_idx * self.channels + bnd_idx
                tensor[c : c + image.shape[0]] = image
                bnd_idx += image.shape[0]
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single location.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        path = self.files.iloc[index][0]
        with rasterio.open(f'{path}_labels.tif') as f:
            array = f.read() - 1
        tensor = torch.from_numpy(array).squeeze().long()
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if all files already exist
        if all(
            os.path.exists(os.path.join(self.root, file_info['name']))
            for file_info in self.metadata
        ):
            return

        # Check for downloaded files
        missing = []
        extractable = []
        for file_info in self.metadata:
            file_path = os.path.join(self.root, file_info['name'] + file_info['ext'])
            if not os.path.exists(file_path):
                missing.append(file_info)
            elif file_info['ext'] == '.zip':
                extractable.append(file_path)

        # Check if the user requested to download the dataset
        if missing and not self.download:
            raise DatasetNotFoundError(self)

        # Download missing files
        for file_info in missing:
            download_url(
                file_info['url'],
                self.root,
                filename=file_info['name'] + file_info['ext'],
                md5=file_info['md5'] if self.checksum else None,
            )
            if file_info['ext'] == '.zip':
                extractable.append(os.path.join(self.root, file_info['name'] + '.zip'))

        # Extract downloaded files
        for file_path in extractable:
            extract_archive(file_path)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        show_legend: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            show_legend: flag indicating whether to show a legend for
                the segmentation masks
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        if '10m_RGB' not in self.bands:
            raise RGBBandsMissingError()

        ncols = self.concat_seasons + 1

        images, mask = sample['image'], sample['mask']
        show_predictions = 'prediction' in sample
        if show_predictions:
            prediction = sample['prediction']
            ncols += 1

        plt_cmap = ListedColormap(np.array(list(self.cmap.values())) / 255)

        start = 0
        for b in self.bands:
            if b == '10m_RGB':
                break
            start += self.band_nums[b]
        rgb_indices = [start + s * self.channels for s in range(self.concat_seasons)]

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4.5, 5))
        fig.subplots_adjust(wspace=0.05)
        for ax, index in enumerate(rgb_indices):
            image = images[index : index + 3].permute(1, 2, 0).numpy()
            image = percentile_normalization(image)
            axs[ax].imshow(image)
            axs[ax].axis('off')
            if show_titles:
                axs[ax].set_title(f'Image {ax+1}')

        axs[ax + 1].imshow(mask, vmin=0, vmax=32, cmap=plt_cmap, interpolation='none')
        axs[ax + 1].axis('off')
        if show_titles:
            axs[ax + 1].set_title('Mask')

        if show_predictions:
            axs[ax + 2].imshow(
                prediction, vmin=0, vmax=32, cmap=plt_cmap, interpolation='none'
            )
            axs[ax + 2].axis('off')
            if show_titles:
                axs[ax + 2].set_title('Prediction')

        if show_legend:
            lgd = np.unique(mask)

            if show_predictions:
                lgd = np.union1d(lgd, np.unique(prediction))
            patches = [
                mpatches.Patch(color=plt_cmap(i), label=self.classes[i]) for i in lgd
            ]
            plt.legend(
                handles=patches, bbox_to_anchor=(1.05, 1), borderaxespad=0, loc=2
            )

        if suptitle is not None:
            plt.suptitle(suptitle, size='xx-large')

        return fig
