# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet dataset."""

import glob
import json
import os
import textwrap
from collections.abc import Callable
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_url, extract_archive, sort_sentinel2_bands


class BigEarthNet(NonGeoDataset):
    """BigEarthNet dataset.

    The `BigEarthNet <https://bigearth.net/>`__
    dataset is a dataset for multilabel remote sensing image scene classification.

    Dataset features:

    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

    Dataset classes (43):

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
    12. Permanently irrigated land
    13. Rice fields
    14. Vineyards
    15. Fruit trees and berry plantations
    16. Olive groves
    17. Pastures
    18. Annual crops associated with permanent crops
    19. Complex cultivation patterns
    20. Land principally occupied by agriculture, with significant
        areas of natural vegetation
    21. Agro-forestry areas
    22. Broad-leaved forest
    23. Coniferous forest
    24. Mixed forest
    25. Natural grassland
    26. Moors and heathland
    27. Sclerophyllous vegetation
    28. Transitional woodland/shrub
    29. Beaches, dunes, sands
    30. Bare rock
    31. Sparsely vegetated areas
    32. Burnt areas
    33. Inland marshes
    34. Peatbogs
    35. Salt marshes
    36. Salines
    37. Intertidal flats
    38. Water courses
    39. Water bodies
    40. Coastal lagoons
    41. Estuaries
    42. Sea and ocean

    Dataset classes (19):

    0. Urban fabric
    1. Industrial or commercial units
    2. Arable land
    3. Permanent crops
    4. Pastures
    5. Complex cultivation patterns
    6. Land principally occupied by agriculture, with significant
       areas of natural vegetation
    7. Agro-forestry areas
    8. Broad-leaved forest
    9. Coniferous forest
    10. Mixed forest
    11. Natural grassland and sparsely vegetated areas
    12. Moors, heathland and sclerophyllous vegetation
    13. Transitional woodland, shrub
    14. Beaches, dunes, sands
    15. Inland wetlands
    16. Coastal wetlands
    17. Inland waters
    18. Marine waters

    The source for the above dataset classes, their respective ordering, and
    43-to-19-class mappings can be found here:

    * https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/blob/master/label_indices.json

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """

    class_sets: ClassVar[dict[int, list[str]]] = {
        19: [
            'Urban fabric',
            'Industrial or commercial units',
            'Arable land',
            'Permanent crops',
            'Pastures',
            'Complex cultivation patterns',
            'Land principally occupied by agriculture, with significant areas of'
            ' natural vegetation',
            'Agro-forestry areas',
            'Broad-leaved forest',
            'Coniferous forest',
            'Mixed forest',
            'Natural grassland and sparsely vegetated areas',
            'Moors, heathland and sclerophyllous vegetation',
            'Transitional woodland, shrub',
            'Beaches, dunes, sands',
            'Inland wetlands',
            'Coastal wetlands',
            'Inland waters',
            'Marine waters',
        ],
        43: [
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
            'Permanently irrigated land',
            'Rice fields',
            'Vineyards',
            'Fruit trees and berry plantations',
            'Olive groves',
            'Pastures',
            'Annual crops associated with permanent crops',
            'Complex cultivation patterns',
            'Land principally occupied by agriculture, with significant areas of'
            ' natural vegetation',
            'Agro-forestry areas',
            'Broad-leaved forest',
            'Coniferous forest',
            'Mixed forest',
            'Natural grassland',
            'Moors and heathland',
            'Sclerophyllous vegetation',
            'Transitional woodland/shrub',
            'Beaches, dunes, sands',
            'Bare rock',
            'Sparsely vegetated areas',
            'Burnt areas',
            'Inland marshes',
            'Peatbogs',
            'Salt marshes',
            'Salines',
            'Intertidal flats',
            'Water courses',
            'Water bodies',
            'Coastal lagoons',
            'Estuaries',
            'Sea and ocean',
        ],
    }

    label_converter: ClassVar[dict[int, int]] = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    splits_metadata: ClassVar[dict[str, dict[str, str]]] = {
        'train': {
            'url': 'https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/train.csv?inline=false',
            'filename': 'bigearthnet-train.csv',
            'md5': '623e501b38ab7b12fe44f0083c00986d',
        },
        'val': {
            'url': 'https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/val.csv?inline=false',
            'filename': 'bigearthnet-val.csv',
            'md5': '22efe8ed9cbd71fa10742ff7df2b7978',
        },
        'test': {
            'url': 'https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/test.csv?inline=false',
            'filename': 'bigearthnet-test.csv',
            'md5': '697fb90677e30571b9ac7699b7e5b432',
        },
    }
    metadata: ClassVar[dict[str, dict[str, str]]] = {
        's1': {
            'url': 'https://zenodo.org/records/12687186/files/BigEarthNet-S1-v1.0.tar.gz',
            'md5': '94ced73440dea8c7b9645ee738c5a172',
            'filename': 'BigEarthNet-S1-v1.0.tar.gz',
            'directory': 'BigEarthNet-S1-v1.0',
        },
        's2': {
            'url': 'https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz',
            'md5': '5a64e9ce38deb036a435a7b59494924c',
            'filename': 'BigEarthNet-S2-v1.0.tar.gz',
            'directory': 'BigEarthNet-v1.0',
        },
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: str = 'all',
        num_classes: int = 19,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits_metadata
        assert bands in ['s1', 's2', 'all']
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}
        self._verify()
        self.folders = self._load_folders()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]['filename']
        dir_s1 = self.metadata['s1']['directory']
        dir_s2 = self.metadata['s2']['directory']

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(',') for line in lines]

        folders = [
            {
                's1': os.path.join(self.root, dir_s1, pair[1]),
                's2': os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders

    def _load_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        if self.bands == 'all':
            folder_s1 = self.folders[index]['s1']
            folder_s2 = self.folders[index]['s2']
            paths_s1 = glob.glob(os.path.join(folder_s1, '*.tif'))
            paths_s2 = glob.glob(os.path.join(folder_s2, '*.tif'))
            paths_s1 = sorted(paths_s1)
            paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
            paths = paths_s1 + paths_s2
        elif self.bands == 's1':
            folder = self.folders[index]['s1']
            paths = glob.glob(os.path.join(folder, '*.tif'))
            paths = sorted(paths)
        else:
            folder = self.folders[index]['s2']
            paths = glob.glob(os.path.join(folder, '*.tif'))
            paths = sorted(paths, key=sort_sentinel2_bands)

        return paths

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype='int32',
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: np.typing.NDArray[np.int_] = np.stack(images, axis=0)
        tensor = torch.from_numpy(arrays).float()
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        if self.bands == 's2':
            folder = self.folders[index]['s2']
        else:
            folder = self.folders[index]['s1']

        path = glob.glob(os.path.join(folder, '*.json'))[0]
        with open(path) as f:
            labels = json.load(f)['labels']

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]

        # Map 43 to 19 class labels
        if self.num_classes == 19:
            indices_optional = [self.label_converter.get(idx) for idx in indices]
            indices = [idx for idx in indices_optional if idx is not None]

        target = torch.zeros(self.num_classes, dtype=torch.long)
        target[indices] = 1
        return target

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        keys = ['s1', 's2'] if self.bands == 'all' else [self.bands]
        urls = [self.metadata[k]['url'] for k in keys]
        md5s = [self.metadata[k]['md5'] for k in keys]
        filenames = [self.metadata[k]['filename'] for k in keys]
        directories = [self.metadata[k]['directory'] for k in keys]
        urls.extend([self.splits_metadata[k]['url'] for k in self.splits_metadata])
        md5s.extend([self.splits_metadata[k]['md5'] for k in self.splits_metadata])
        filenames_splits = [
            self.splits_metadata[k]['filename'] for k in self.splits_metadata
        ]
        filenames.extend(filenames_splits)

        # Check if the split file already exist
        exists = []
        for filename in filenames_splits:
            exists.append(os.path.exists(os.path.join(self.root, filename)))

        # Check if the files already exist
        for directory in directories:
            exists.append(os.path.exists(os.path.join(self.root, directory)))

        if all(exists):
            return

        # Check if zip file already exists (if so then extract)
        exists = []
        for filename in filenames:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                exists.append(True)
                self._extract(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        for url, filename, md5 in zip(urls, filenames, md5s):
            self._download(url, filename, md5)
            filepath = os.path.join(self.root, filename)
            self._extract(filepath)

    def _download(self, url: str, filename: Path, md5: str) -> None:
        """Download the dataset.

        Args:
            url: url to download file
            filename: output filename to write downloaded file
            md5: md5 of downloaded file
        """
        download_url(
            url, self.root, filename=filename, md5=md5 if self.checksum else None
        )

    def _extract(self, filepath: Path) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        if not str(filepath).endswith('.csv'):
            extract_archive(filepath)

    def _onehot_labels_to_names(
        self, label_mask: 'np.typing.NDArray[np.bool_]'
    ) -> list[str]:
        """Gets a list of class names given a label mask.

        Args:
            label_mask: a boolean mask corresponding to a set of labels or predictions

        Returns:
            a list of class names corresponding to the input mask
        """
        labels = []
        for i, mask in enumerate(label_mask):
            if mask:
                labels.append(self.class_sets[self.num_classes][i])
        return labels

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

        .. versionadded:: 0.2
        """
        if self.bands == 's2':
            image = np.rollaxis(sample['image'][[3, 2, 1]].numpy(), 0, 3)
            image = np.clip(image / 2000, 0, 1)
        elif self.bands == 'all':
            image = np.rollaxis(sample['image'][[5, 4, 3]].numpy(), 0, 3)
            image = np.clip(image / 2000, 0, 1)
        elif self.bands == 's1':
            image = sample['image'][0].numpy()

        label_mask = sample['label'].numpy().astype(np.bool_)
        labels = self._onehot_labels_to_names(label_mask)

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction_mask = sample['prediction'].numpy().astype(np.bool_)
            predictions = self._onehot_labels_to_names(prediction_mask)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis('off')
        if show_titles:
            title = f'Labels: {", ".join(labels)}'
            if showing_predictions:
                title += f'\nPredictions: {", ".join(predictions)}'
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class BigEarthNetV2(NonGeoDataset):
    """BigEarthNetV2 dataset.

    The `BigEarthNet V2 <https://bigearth.net/>`__ dataset contains improved labels, improved
    geospatial data splits and additionally pixel-level labels from CORINE Land
    Cover (CLC) map of 2018. Additionally, some problematic patches from V1 have been removed.

    .. versionadded:: 0.7
    """

    class_set = BigEarthNet.class_sets[19]

    image_size = BigEarthNet.image_size

    url = 'https://hf.co/datasets/torchgeo/bigearthnet/resolve/3cf3a5910a5302d449fdb8e570e5b78de24fe07f/V2/{}'

    metadata_locs: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        's1': {
            'files': {
                'BigEarthNet-S1.tar.gzaa': '039b9ce305fc6582b2c3d60d1573f5b7',
                'BigEarthNet-S1.tar.gzab': 'e94f0ea165d04992ca91d8e58e82ec6d',
            }
        },
        's2': {
            'files': {
                'BigEarthNet-S2.tar.gzaa': '94e8ed32065234d3ab46353d814778d1',
                'BigEarthNet-S2.tar.gzab': '24c223d9e36166136c13b24a27debe34',
            }
        },
        'maps': {
            'files': {'Reference_Maps.tar.gzaa': 'b0cd1f0a31b49fcbfd61d80f963e759d'}
        },
        'metadata': {'files': {'metadata.parquet': '55687065e77b6d0b0f1ff604a6e7b49c'}},
    }

    dir_file_names: ClassVar[dict[str, str]] = {
        's1': 'BigEarthNet-S1',
        's2': 'BigEarthNet-S2',
        'maps': 'Reference_Maps',
        'metadata': 'metadata.parquet',
    }

    # https://collections.sentinel-hub.com/corine-land-cover/readme.html
    # Table 1 of https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
    clc_colors: ClassVar[dict[str, str]] = {
        'Urban fabric': '#e6004d',
        'Industrial or commercial units': '#cc4df2',
        'Arable land': '#ffffa8',
        'Permanent crops': '#e68000',
        'Pastures': '#e6e64d',
        'Complex cultivation patterns': '#ffe64d',
        'Land principally occupied by agriculture, with significant areas of natural vegetation': '#e6cc4d',
        'Agro-forestry areas': '#f2cca6',
        'Broad-leaved forest': '#80ff00',
        'Coniferous forest': '#00a600',
        'Mixed forest': '#4dff00',
        'Natural grassland and sparsely vegetated areas': '#ccf24d',
        'Moors, heathland and sclerophyllous vegetation': '#a6ff80',
        'Transitional woodland, shrub': '#a6f200',
        'Beaches, dunes, sands': '#e6e6e6',
        'Inland wetlands': '#a6a6ff',
        'Coastal wetlands': '#ccccff',
        'Inland waters': '#80f2e6',
        'Marine waters': '#e6f2ff',
    }

    clc_codes: ClassVar[dict[int, int]] = {
        111: 0,  # Continuous Urban fabric
        112: 0,  # Discontinuous Urban fabric
        121: 1,  # Industrial or commercial units
        211: 2,  # Non-irrigated arable land
        212: 2,  # Permanently irrigated land
        213: 2,  # Rice fields
        221: 3,  # Vineyards
        222: 3,  # Fruit trees and berry plantations
        223: 3,  # Olive groves
        231: 4,  # Pastures
        241: 3,  # Annual crops with permanent crops
        242: 5,  # Complex cultivation patterns
        243: 6,  # Land principally occupied by agriculture...
        244: 7,  # Agro-forestry areas
        311: 8,  # Broad-leaved forest
        312: 9,  # Coniferous forest
        313: 10,  # Mixed forest
        321: 11,  # Natural grassland
        322: 12,  # Moors and heathland
        323: 12,  # Sclerophyllous vegetation
        324: 13,  # Transitional woodland/shrub
        331: 14,  # Beaches, dunes, sands
        333: 11,  # Sparsely vegetated areas
        411: 15,  # Inland marshes
        412: 15,  # Peatbogs
        421: 16,  # Salt marshes
        422: 16,  # Salines
        511: 17,  # Water courses
        512: 17,  # Water bodies
        521: 18,  # Coastal lagoons
        522: 18,  # Estuaries
        523: 18,  # Sea and ocean
    }

    valid_splits = ('train', 'val', 'test')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: str = 'all',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet V2 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            AssertionError: If *split*, or *bands*, are not valid.
        """
        assert split in self.valid_splits, f'split must be one of {self.valid_splits}'
        assert bands in ['s1', 's2', 'all']
        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.num_classes = 19
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_set)}
        self._verify()

        self.metadata_df = pd.read_parquet(os.path.join(self.root, 'metadata.parquet'))
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == self.split
        ].reset_index(drop=True)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        self.ordinal_map = torch.zeros(19)
        for corine, ordinal in self.clc_codes.items():
            self.ordinal_map[ordinal] = corine

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Tensor] = {}

        match self.bands:
            case 's1':
                sample['image'] = self._load_image(index, 's1')
            case 's2':
                sample['image'] = self._load_image(index, 's2')
            case 'all':
                sample['image_s1'] = self._load_image(index, 's1')
                sample['image_s2'] = self._load_image(index, 's2')

        sample['mask'] = self._load_map(index)
        sample['label'] = self._load_target(index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int, sensor: str) -> Tensor:
        """Generic image loader for both S1 and S2.

        Args:
            index: index to return
            sensor: 's1' or 's2'

        Returns:
            the sensor image
        """
        row = self.metadata_df.loc[index]
        id_field = 's1_name' if sensor == 's1' else 'patch_id'
        patch_id = row[id_field]
        if sensor == 's2':
            patch_dir = '_'.join(patch_id.split('_')[0:-2])
        else:
            patch_dir = '_'.join(patch_id.split('_')[0:-3])

        paths = glob.glob(
            os.path.join(
                self.root, self.dir_file_names[sensor], patch_dir, patch_id, '*.tif'
            )
        )

        if sensor == 's2':
            paths = sorted(paths, key=sort_sentinel2_bands)
        else:
            paths = sorted(paths)

        images = []
        for path in paths:
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype='int32',
                    resampling=Resampling.bilinear,
                )
                images.append(array)

        return torch.from_numpy(np.stack(images, axis=0)).float()

    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the Corine Land Cover map
        """
        row = self.metadata_df.loc[index]
        patch_id = row['patch_id']
        patch_dir = '_'.join(patch_id.split('_')[0:-2])
        path = os.path.join(
            self.root,
            self.dir_file_names['maps'],
            patch_dir,
            patch_id,
            patch_id + '_reference_map.tif',
        )
        with rasterio.open(path) as dataset:
            map = dataset.read(out_dtype='int32')

        tensor = torch.from_numpy(map)
        # remap to ordinal values
        for corine, ordinal in self.clc_codes.items():
            tensor[tensor == corine] = ordinal
        return tensor.long()

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        label_names = self.metadata_df.iloc[index]['labels']

        indices = [self.class2idx[label_names] for label_names in label_names]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1
        return image_target

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        exists = []
        for key, metadata in self.metadata_locs.items():
            exists.append(
                os.path.exists(os.path.join(self.root, self.dir_file_names[key]))
            )

        if all(exists):
            return

        # check if compressed files already exist
        exists = []
        for key, metadata in self.metadata_locs.items():
            if key == 'metadata':
                exists.append(
                    os.path.exists(os.path.join(self.root, self.dir_file_names[key]))
                )
            else:
                for fname in metadata['files']:
                    fpath = os.path.join(self.root, fname)
                    exists.append(os.path.exists(fpath))

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the required tarball parts using the URL template and md5 sums."""
        for key, meta in self.metadata_locs.items():
            for fname, md5 in meta['files'].items():
                target_path = os.path.join(self.root, fname)
                if not os.path.exists(target_path):
                    download_url(self.url.format(fname), self.root, md5)

    def _extract(self) -> None:
        """Extract the tarball parts.

        For each modality (s1, s2, maps), its parts are concatenated together and then extracted.
        """
        chunk_size = 2**15  # same as used in torchvision and ssl4eo
        for key, meta in self.metadata_locs.items():
            if key == 'metadata':
                continue
            parts = [os.path.join(self.root, f) for f in meta['files'].keys()]
            concat_path = os.path.join(self.root, self.dir_file_names[key] + '.tar.gz')
            with open(concat_path, 'wb') as outfile:
                for part in parts:
                    with open(part, 'rb') as g:
                        while chunk := g.read(chunk_size):
                            outfile.write(chunk)
            extract_archive(concat_path, self.root)

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
        fig, axes = plt.subplots(1, 2 if self.bands != 'all' else 3, figsize=(12, 4))

        if self.bands in ['s2', 'all']:
            s2_img = sample['image_s2' if self.bands == 'all' else 'image']
            rgb = np.rollaxis(s2_img[[3, 2, 1]].numpy(), 0, 3)
            axes[0].imshow(np.clip(rgb / 2000, 0, 1))
            if show_titles:
                axes[0].set_title('Sentinel-2 RGB')
            axes[0].axis('off')

        if self.bands in ['s1', 'all']:
            idx = 0 if self.bands == 's1' else 1
            s1_img = sample['image_s1' if self.bands == 'all' else 'image']
            axes[idx].imshow(s1_img[0].numpy())
            if show_titles:
                axes[idx].set_title('Sentinel-1 VV')
            axes[idx].axis('off')

        # Handle mask plotting
        mask_idx = 1 if self.bands != 'all' else 2
        mask = sample['mask'][0].numpy()

        # Get unique ordinal labels from mask
        unique_labels = sorted(np.unique(mask))

        # Map ordinal labels to class names and colors directly
        colors = []
        class_names = []
        for label in unique_labels:
            name = self.class_set[label]  # Get class name from ordinal index
            colors.append(self.clc_colors[name])  # Get color for class name
            class_names.append(name)

        # Create custom colormap
        cmap = ListedColormap(colors)
        bounds = [*unique_labels, unique_labels[-1] + 1]
        norm = BoundaryNorm(bounds, len(colors))

        axes[mask_idx].imshow(mask, cmap=cmap, norm=norm)

        # Add legend with class names
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color) for color in colors]
        wrapped_names = [textwrap.fill(name, width=25) for name in class_names]
        axes[mask_idx].legend(
            legend_elements,
            wrapped_names,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize='x-small',
        )
        axes[mask_idx].axis('off')

        if show_titles:
            axes[mask_idx].set_title('Land Cover Map')

        if 'label' in sample:
            label_indices = sample['label'].nonzero().squeeze(1).tolist()
            label_names = [self.class_set[idx] for idx in label_indices]
            if suptitle:
                suptitle = f'{suptitle}\nLabels: {", ".join(label_names)}'
            else:
                suptitle = f'Labels: {", ".join(label_names)}'

        if suptitle:
            plt.suptitle(suptitle)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        return fig
