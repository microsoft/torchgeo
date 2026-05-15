# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""FLAIRHUB dataset."""

import pathlib
from collections.abc import Callable
from typing import ClassVar, Literal, TypedDict

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from einops import rearrange
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    array_to_tensor,
    download_url,
    extract_archive,
    quantile_normalization,
)

AvailableBands = Literal[
    'AERIAL_RGBI',
    'SPOT_RGBI',
    'DEM_ELEV',
    'AERIAL-RLT_PAN',
    'SENTINEL2_TS',
    'SENTINEL2_MSK-SC',
    'SENTINEL1-ASC_TS',
    'SENTINEL1-DESC_TS',
]


class _Task(TypedDict, total=False):
    classes: list[str]
    cmap: ListedColormap


class _PlotData(TypedDict):
    plot_type: str
    data: Tensor
    title: str


_PLOT_KEYS: dict[str, tuple[str, str]] = {
    'image_aerial_rgbi': ('aerial_rgbi', 'Aerial RGBI'),
    'image_spot_rgbi': ('spot_rgbi', 'SPOT RGBI'),
    'image_dem_elev': ('dem', 'DEM Elevation'),
    'image_aerial_rlt_pan': ('aerial_rlt_pan', 'Aerial RLT PAN'),
    'image_sentinel2': ('sentinel2_ts', 'Sentinel-2 Time Series'),
    'mask_sentinel2_snowcloud': ('sentinel2_msk_sc', 'Sentinel-2 Mask SC'),
    'image_sentinel1_asc': ('sentinel1_asc_ts', 'Sentinel-1 ASC Time Series'),
    'image_sentinel1_desc': ('sentinel1_desc_ts', 'Sentinel-1 DESC Time Series'),
}


TASKS: dict[str, _Task] = {
    'land_cover': {
        # Note: the original dataset contains 19 classes, but the dataset paper
        # suggests not using clear cut, ligneous & mixed as they are nearly empty
        'classes': [
            'building',
            'greenhouse',
            'swimming_pool',
            'impervious surface',
            'pervious surface',
            'bare soil',
            'water',
            'snow',
            'herbaceous vegetation',
            'agricultural land',
            'plowed land',
            'vineyard',
            'deciduous',
            'coniferous',
            'brushwood',
            'clear cut',
            'ligneous',
            'mixed',
            'undefined',
        ],
        'cmap': ListedColormap(
            [
                '#db0e9a',  # building
                '#9999ff',  # greenhouse
                '#3de6eb',  # swimming_pool
                '#f80c00',  # impervious surface
                '#938e7b',  # pervious surface
                '#a97101',  # bare soil
                '#1553ae',  # water
                '#ffffff',  # snow
                '#55ff00',  # herbaceous vegetation
                '#fff30d',  # agricultural land
                '#e4df7c',  # plowed land
                '#660082',  # vineyard
                '#46e483',  # deciduous
                '#194a26',  # coniferous
                '#f3a60d',  # brushwood
                '#8ab3a0',  # clear cut
                '#c5dc42',  # ligneous
                '#6b714f',  # mixed
                '#000000',  # undefined
            ]
        ),
    },
    'crop_type': {
        'classes': [
            'grasses',
            'wheat',
            'barley',
            'maize',
            'other cereals',
            'rice',
            'flax/hemp/tobacco',
            'sunflower',
            'rapeseed',
            'other oilseed crops',
            'soy',
            'other protein crops',
            'fodder legumes',
            'beetroots',
            'potatoes',
            'other arable crops',
            'vineyard',
            'olive groves',
            'fruits orchards',
            'nut orchards',
            'other permanent crops',
            'mixed crops',
            'background',
        ],
        'cmap': ListedColormap(
            [
                '#92d050',  # grasses
                '#d7e600',  # wheat
                '#e0e000',  # barley
                '#fff100',  # maize
                '#ffff00',  # other cereals
                '#e8e8e8',  # rice
                '#dceaf7',  # flax/hemp/tobacco
                '#d29ead',  # sunflower
                '#d29ed0',  # rapeseed
                '#ffbe99',  # other oilseed crops
                '#ffc000',  # soy
                '#ff9000',  # other protein crops
                '#009999',  # fodder legumes
                '#808000',  # beetroots
                '#a7a700',  # potatoes
                '#89896d',  # other arable crops
                '#f2cfee',  # vineyard
                '#6f6633',  # olive groves
                '#ac8141',  # fruits orchards
                '#996633',  # nut orchards
                '#80c1d7',  # other permanent crops
                '#000000',  # mixed crops
                '#000000',  # background
            ]
        ),
    },
    'crop_type_2': {
        'classes': [
            'grasses',
            'wheat',
            'barley',
            'maize',
            'sorghum/millet',
            'other winter cereals',
            'other spring cereals',
            'other cereals',
            'rice',
            'hemp/tobacco',
            'flax',
            'sunflower',
            'rapeseed',
            'other oilseed crops',
            'soy',
            'other protein crops',
            'alfalfa',
            'other fodder legumes',
            'beetroots',
            'potatoes',
            'fruits and vegetables',
            'aromatic/medicinal plants',
            'other arable crops',
            'vineyard',
            'olive groves',
            'fruit orchards',
            'nut orchards',
            'lavandin',
            'other permanent crops',
            'mixed crops',
            'background',
        ],
        'cmap': ListedColormap(
            [
                '#92d050',  # grasses
                '#d7e600',  # wheat
                '#e0e000',  # barley
                '#fff100',  # maize
                '#ffff00',  # sorghum/millet
                '#ffff00',  # other winter cereals
                '#ffff00',  # other spring cereals
                '#ffff00',  # other cereals
                '#e8e8e8',  # rice
                '#dceaf7',  # hemp/tobacco
                '#dceaf7',  # flax
                '#d29ead',  # sunflower
                '#d29ed0',  # rapeseed
                '#ffbe99',  # other oilseed crops
                '#ffc000',  # soy
                '#ff9000',  # other protein crops
                '#009999',  # alfalfa
                '#009999',  # other fodder legumes
                '#808000',  # beetroots
                '#a7a700',  # potatoes
                '#89896d',  # fruits and vegetables
                '#89896d',  # aromatic/medicinal plants
                '#89896d',  # other arable crops
                '#f2cfee',  # vineyard
                '#6f6633',  # olive groves
                '#ac8141',  # fruit orchards
                '#996633',  # nut orchards
                '#80c1d7',  # lavandin
                '#80c1d7',  # other permanent crops
                '#000000',  # mixed crops
                '#000000',  # background
            ]
        ),
    },
    'crop_type_3': {
        'classes': [
            'grasses monoculture',
            'grasses mixture',
            'winter wheat',
            'spring wheat',
            'winter barley',
            'spring barley',
            'maize',
            'sorghum',
            'millet / Foxtail millet',
            'winter durum wheat',
            'winter triticale',
            'winter oat',
            'winter rye',
            'spring oat',
            'other spring cereals',
            'other cereals',
            'rice',
            'hemp/tobacco',
            'fiber flax',
            'other flax',
            'sunflower',
            'rapeseed',
            'Mustard',
            'other oilseed crops',
            'soy',
            'spring peas',
            'winter protein crops',
            'other protein crops',
            'alfalfa',
            'clover',
            'other fodder legumes',
            'beetroots',
            'potatoes',
            'fruits and vegetables',
            'aromatic/medicinal plants',
            'buckwheat',
            'other arable crops',
            'vineyard',
            'olive groves',
            'fruit orchards',
            'nut orchards',
            'lavandin',
            'berries',
            'other permanent crops',
            'mixed crops',
            'background',
        ],
        'cmap': ListedColormap(
            [
                '#92d050',  # grasses monoculture
                '#92d050',  # grasses mixture
                '#d7e600',  # winter wheat
                '#d7e600',  # spring wheat
                '#e0e000',  # winter barley
                '#e0e000',  # spring barley
                '#fff100',  # maize
                '#ffff00',  # sorghum
                '#ffff00',  # millet / Foxtail millet
                '#ffff00',  # winter durum wheat
                '#ffff00',  # winter triticale
                '#ffff00',  # winter oat
                '#ffff00',  # winter rye
                '#ffff00',  # spring oat
                '#ffff00',  # other spring cereals
                '#ffff00',  # other cereals
                '#e8e8e8',  # rice
                '#dceaf7',  # hemp/tobacco
                '#dceaf7',  # fiber flax
                '#dceaf7',  # other flax
                '#d29ead',  # sunflower
                '#d29ed0',  # rapeseed
                '#ffbe99',  # Mustard
                '#ffbe99',  # other oilseed crops
                '#ffc000',  # soy
                '#ff9000',  # spring peas
                '#ff9000',  # winter protein crops
                '#ff9000',  # other protein crops
                '#009999',  # alfalfa
                '#009999',  # clover
                '#009999',  # other fodder legumes
                '#808000',  # beetroots
                '#a7a700',  # potatoes
                '#89896d',  # fruits and vegetables
                '#89896d',  # aromatic/medicinal plants
                '#89896d',  # buckwheat
                '#89896d',  # other arable crops
                '#f2cfee',  # vineyard
                '#6f6633',  # olive groves
                '#ac8141',  # fruit orchards
                '#996633',  # nut orchards
                '#80c1d7',  # lavandin
                '#80c1d7',  # berries
                '#80c1d7',  # other permanent crops
                '#000000',  # mixed crops
                '#000000',  # background
            ]
        ),
    },
}


class FLAIRHUB(NonGeoDataset):
    """FLAIR-HUB Dataset.

    Large-scale Multimodal Dataset for Land Cover and Crop Mapping dataset.

    `FLAIR-HUB <https://github.com/IGNF/FLAIR-HUB>`__ builds upon and includes the FLAIR#1 and FLAIR#2 datasets,
    expanding them into a unified, large-scale,
    multi-sensor land-cover resource with very-high-resolution
    annotations. Spanning over 2,500 km² of diverse French ecoclimatic regions
    and landscapes, FLAIR-HUB features 63 billion hand-annotated pixels across
    19 land-cover and 23 crop type classes.

    The dataset integrates complementary sources including aerial imagery,
    SPOT and Sentinel satellite acquisitions, surface models, and historical
    aerial photographs. This offers rich spatial, spectral, and temporal
    diversity, supporting a broad range of research tasks, including semantic
    segmentation, multimodal fusion, and self-supervised learning. FLAIR-HUB
    is designed as a continuously growing resource, with new modalities and
    annotations to be released in future updates.

    Dataset features:

    - ROI / Area Covered: 2,822 ROIs / 2,528 km²
    - Departments (France): 74
    - AI Patches (512x512 px): 241,100
    - Annotated Pixels: 63.2 billion
    - Sentinel-2 Acquisitions: 256,221
    - Sentinel-1 Acquisitions: 532,696
    - Total Files: ~2.5 million
    - Total Dataset Size: ~750 GB

    Dataset structure:

    The dataset is organized by domains (geographical areas) and years.
    Each domain-year combination has multiple modalities available for
    download.
    The full dataset contains
    66 unique domains (D004-D091, non-consecutive), with years ranging from 2017-2022.
    Most domains have data for a single year, but some have multiple years available.

    Available modalities (100% coverage across all domains):

    - ``AERIAL_RGBI``: High-resolution aerial imagery (RGB + NIR, 0.2m) — key: ``image_aerial_rgbi``
    - ``SPOT_RGBI``: SPOT satellite imagery (RGB + NIR, 1.5m) — key: ``image_spot_rgbi``
    - ``DEM_ELEV``: Digital Elevation Model (DSM + DTM, 1m) — key: ``image_dem_elev``
    - ``AERIAL-RLT_PAN``: Historical aerial panchromatic (1950s) — key: ``image_aerial_rlt_pan``
    - ``SENTINEL1-ASC_TS``: Sentinel-1 SAR Ascending time series (VV + VH) — key: ``image_sentinel1_asc``
    - ``SENTINEL1-DESC_TS``: Sentinel-1 SAR Descending time series (VV + VH) — key: ``image_sentinel1_desc``
    - ``SENTINEL2_TS``: Sentinel-2 multispectral time series (12 bands, 10m) — key: ``image_sentinel2``
    - ``SENTINEL2_MSK-SC``: Sentinel-2 scene classification mask — key: ``mask_sentinel2_snowcloud``
    - ``AERIAL_LABEL-COSIA``: Land cover labels (19 classes)
    - ``ALL_LABEL-LPIS``: Crop type labels (23 classes)

    Automatic download:

    Set ``download=True`` to automatically download requested modalities from
    HuggingFace. Only the modalities you select will be downloaded, saving
    bandwidth and storage.

    Dataset classes:

    - **AERIAL_LABEL-COSIA** (Land Cover):

        - 0:  urban
        - 1:  greenhouse
        - 2:  swimming_pool
        - 3:  impervious surfaces
        - 4:  pervious surface
        - 5:  bare soil
        - 6:  water
        - 7:  snow
        - 8:  herbaceous vegetation
        - 9:  agricultural land
        - 10: plowed land
        - 11: vineyard
        - 12: deciduous
        - 13: coniferous
        - 14: brushwood
        - 15: clear cut
        - 16: ligneous
        - 17: mixed
        - 18: undefined

    - **ALL_LABEL-LPIS** (Crop Type):

        - 0: grasses
        - 1: wheat
        - 2: barley
        - 3: maize
        - 4: other cereals
        - 5: rice
        - 6: flax/hemp/tobacco
        - 7: sunflower
        - 8: rapeseed
        - 9: other oilseed crops
        - 10: soy
        - 11: other protein crops
        - 12: fodder legumes
        - 13: beetroots
        - 14: potatoes
        - 15: other arable crops
        - 16: vineyard
        - 17: olive groves
        - 18: fruits orchards
        - 19: nut orchards
        - 20: other permanent crops
        - 21: mixed crops
        - 22: background

    If you use this dataset in your research, please cite the following paper:

    - https://arxiv.org/abs/2506.07080

    .. versionadded:: 0.10
    """

    # AERIAL_RGBI
    aerial_rgb_bands = ('B01', 'B02', 'B03')
    aerial_all_bands = ('B01', 'B02', 'B03', 'B04')  # B04 is the NIR band

    # SPOT_RGBI
    spot_all_bands = ('B01', 'B02', 'B03', 'B04')  # B04 is the NIR band
    spot_rgb_bands = ('B01', 'B02', 'B03')

    # DEM_ELEV (Digital Elevation Model)
    dem_elev_bands = ('DSM', 'DTM')

    # aerial-rlt_pan (Historical aerial panchromatic)
    aerial_rlt_pan_bands = 'PAN'

    # SENTINEL2_TS (Sentinel-2 time series)
    sentinel2_ts_bands = (
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B11',
        'B12',
    )
    sentinel2_ts_rgb_bands = ('B04', 'B03', 'B02')

    # SENTINEL2_MSK-SC (Sentinel-2 scene classification mask)
    sentinel2_msk_sc_bands = ('snow', 'cloud')  # snow and cloud probabilty mask

    # SENTINEL1_ASC_TS (Sentinel-1 ASC time series)
    sentinel1_asc_ts_bands = ('VV', 'VH')
    # SENTINEL1_DESC_TS (Sentinel-1 DESC time series)
    sentinel1_desc_ts_bands = ('VV', 'VH')

    download_link = 'https://hf.co/datasets/IGNF/FLAIR-HUB/resolve/e8ed7981d488508aa70bb05c37cf6585432b7d5f/data'

    # Note: Some domains have multiple years available
    domain_years: ClassVar[dict[str, list[str]]] = {
        'D004': ['2021'],
        'D005': ['2018'],
        'D006': ['2020'],
        'D007': ['2020'],
        'D008': ['2019'],
        'D009': ['2019'],
        'D010': ['2019'],
        'D011': ['2021'],
        'D012': ['2019'],
        'D013': ['2020'],
        'D014': ['2020'],
        'D015': ['2020'],
        'D016': ['2020'],
        'D017': ['2018', '2021'],
        'D018': ['2020'],
        'D020': ['2019'],
        'D021': ['2020'],
        'D022': ['2021'],
        'D023': ['2020'],
        'D024047': ['2021'],
        'D025039': ['2020'],
        'D026': ['2020'],
        'D029': ['2021'],
        'D030': ['2021'],
        'D031': ['2019'],
        'D032': ['2019'],
        'D033': ['2018', '2021'],
        'D034': ['2021'],
        'D035': ['2020'],
        'D036': ['2020'],
        'D037': ['2021'],
        'D038': ['2021'],
        'D040': ['2021'],
        'D041': ['2021'],
        'D044': ['2020', '2022'],
        'D045': ['2020'],
        'D046': ['2019'],
        'D049': ['2020'],
        'D051': ['2019'],
        'D052': ['2019'],
        'D054057': ['2018'],
        'D055': ['2018'],
        'D056': ['2019'],
        'D058': ['2020'],
        'D059062': ['2021'],
        'D060': ['2021'],
        'D061': ['2020'],
        'D063': ['2019'],
        'D064': ['2021'],
        'D065': ['2019'],
        'D066': ['2021'],
        'D067': ['2021'],
        'D068': ['2021'],
        'D069': ['2020'],
        'D070': ['2020'],
        'D071': ['2020'],
        'D072': ['2019'],
        'D073': ['2022'],
        'D074': ['2020'],
        'D075': ['2021'],
        'D076': ['2019'],
        'D077': ['2021'],
        'D078': ['2021'],
        'D080': ['2017', '2021'],
        'D081': ['2020'],
        'D083': ['2020'],
        'D084': ['2021'],
        'D085': ['2019'],
        'D086': ['2020'],
        'D091': ['2021'],
    }

    modality_key_map: ClassVar[dict[AvailableBands, str]] = {
        'AERIAL_RGBI': 'image_aerial_rgbi',
        'SPOT_RGBI': 'image_spot_rgbi',
        'DEM_ELEV': 'image_dem_elev',
        'AERIAL-RLT_PAN': 'image_aerial_rlt_pan',
        'SENTINEL2_TS': 'image_sentinel2',
        'SENTINEL2_MSK-SC': 'mask_sentinel2_snowcloud',
        'SENTINEL1-ASC_TS': 'image_sentinel1_asc',
        'SENTINEL1-DESC_TS': 'image_sentinel1_desc',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        split_column: Literal[
            'split_1',
            'split_2',
            'split_3',
            'split_4',
            'split_5',
            'split_flairchallenge',
        ]
        | Literal['split_toy'] = 'split_1',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        bands: list[AvailableBands] | None = None,
        dataset_type: Literal[
            'land_cover', 'crop_type', 'crop_type_2', 'crop_type_3'
        ] = 'land_cover',
    ) -> None:
        """Initialize a new FLAIRHUB dataset instance.

        The FLAIR-HUB dataset provides multiple complementary data modalities for land
        cover and crop type mapping. You can selectively load any combination of the
        available modalities by specifying them in the bands parameter.

        Only samples belonging to the specified split (train, val, or test) are loaded,
        using the official splits from ``GLOBAL_ALL_MTD_SPLIT.gpkg``.

        Args:
            root: Root directory where dataset can be found or will be downloaded.
            split: One of ``train``, ``val``, or ``test``.
            split_column: Column name in the official splits GeoPackage (e.g. ``split_1``,
                ``split_5``, ``split_flairchallenge``).
            transforms: Optional transforms to apply to samples.
            download: If True, download the dataset if it is not found.
            bands: List of bands/modalities to load. Available options:

                - ``AERIAL_RGBI``: High-resolution aerial imagery (RGB + NIR, 0.2m) — key: ``image_aerial_rgbi``
                - ``SPOT_RGBI``: SPOT satellite imagery (RGB + NIR, 1.5m) — key: ``image_spot_rgbi``
                - ``DEM_ELEV``: Digital Elevation Model (DSM + DTM, 1m) — key: ``image_dem_elev``
                - ``AERIAL-RLT_PAN``: Historical aerial panchromatic (1950s) — key: ``image_aerial_rlt_pan``
                - ``SENTINEL2_TS``: Sentinel-2 time series data — key: ``image_sentinel2``
                - ``SENTINEL2_MSK-SC``: Sentinel-2 cloud and snow probability masks — key: ``mask_sentinel2_snowcloud``
                - ``SENTINEL1-ASC_TS``: Sentinel-1 ASC time series data — key: ``image_sentinel1_asc``
                - ``SENTINEL1-DESC_TS``: Sentinel-1 DESC time series data — key: ``image_sentinel1_desc``

                Defaults to None, which enables all bands.
            dataset_type: Type of labels to use. Choose ``land_cover`` for
                19-class COSIA labels or ``crop_type`` for 23-class LPIS crop
                classification labels (baseline) or ``crop_type_2`` for 31-class
                LPIS crop classification labels or ``crop_type_3`` for 46-class
                LPIS crop classification labels. ``crop_type_2`` and
                ``crop_type_3`` are deeper levels of the LPIS crop classification
                labels.

        Raises:
            AssertionError: If ``split`` or ``split_column`` is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
            ValueError: If *dataset_type* is not ``land_cover``, ``crop_type``,
                ``crop_type_2``, or ``crop_type_3``.
            ValueError: If no data modality is enabled.
            ValueError: If an invalid band name is provided.
        """
        self.root = root
        self.split = split
        self.split_column = split_column
        self.transforms = transforms
        self.download = download
        self.dataset_type = dataset_type

        if bands is None:
            bands = list(self.modality_key_map.keys())

        self.bands = bands
        self._verify()
        self.files = self._load_files()

    def _ensure_splits_available(self) -> pathlib.Path:
        """Download and extract the official splits file if missing.

        Returns:
            Path to ``GLOBAL_ALL_MTD_SPLIT.gpkg``.

        Raises:
            DatasetNotFoundError: If the splits file is missing and *download* is False.
        """
        root = pathlib.Path(self.root)
        zip_path = root / 'GLOBAL_ALL_MTD.zip'
        gpkg_path = root / 'GLOBAL_ALL_MTD' / 'GLOBAL_ALL_MTD_SPLIT.gpkg'

        if gpkg_path.exists():
            return gpkg_path

        if not zip_path.exists():
            if not self.download:
                raise DatasetNotFoundError(self)
            download_url(f'{self.download_link}/GLOBAL_ALL_MTD.zip', str(root))
        extract_archive(str(zip_path), str(root))
        return gpkg_path

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            Dictionary containing tensors for each modality. Keys are the
            modality names prefixed with ``image_`` or ``mask_``: ``mask``,
            ``image_aerial_rgbi``, ``image_spot_rgbi``, ``image_dem_elev``,
            ``image_aerial_rlt_pan``, ``image_sentinel2``,
            ``mask_sentinel2_snowcloud``, ``image_sentinel1_asc``,
            ``image_sentinel1_desc``.
        """
        file_dict = self.files[index]
        sample: Sample = {}

        # Load mask (always present)
        mask_path = file_dict['mask']
        sample['mask'] = self._load_mask(mask_path)

        # Load requested modalities
        for modality_name in self.bands:
            sample_key = self.modality_key_map[modality_name]
            modality_path = file_dict[modality_name]
            if 'TS' in modality_name or 'SC' in modality_name:
                band_names: tuple[str, ...]
                match modality_name:
                    case 'SENTINEL2_TS':
                        band_names = self.sentinel2_ts_bands
                    case 'SENTINEL2_MSK-SC':
                        band_names = self.sentinel2_msk_sc_bands
                    case 'SENTINEL1-ASC_TS':
                        band_names = self.sentinel1_asc_ts_bands
                    case 'SENTINEL1-DESC_TS':
                        band_names = self.sentinel1_desc_ts_bands

                sample[sample_key] = self._load_time_series(
                    modality_path, len(band_names)
                )
            else:
                sample[sample_key] = self._load_image(modality_path)

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str | pathlib.Path]]:
        """Load the files for the dataset.

        Only patches belonging to the configured split are included.

        Returns:
            List of dictionaries with paths to each modality for each sample.
        """
        files_list: list[dict[str, str | pathlib.Path]] = []

        if self.dataset_type == 'land_cover':
            label_dir = 'AERIAL_LABEL-COSIA'
        else:
            label_dir = 'ALL_LABEL-LPIS'

        filename_glob = f'D*_{label_dir}/*/*.tif'

        gpkg_path = self._ensure_splits_available()
        gdf = gpd.read_file(gpkg_path)
        # Dataset uses 'valid', API uses 'val'
        gpkg_split = 'valid' if self.split == 'val' else self.split
        allowed_patch_ids: set[str] = set(
            gdf.loc[gdf[self.split_column] == gpkg_split, 'patch_id'].astype(str)
        )

        # Iterate through all label files and build file dictionaries
        for label_path in pathlib.Path(self.root).glob(filename_glob):
            # Extract patch_id: D{region}-{year}_{tile}_{coords}
            filename_stem = label_path.stem
            patch_id = filename_stem.replace(f'{label_dir}_', '')
            if allowed_patch_ids is not None and patch_id not in allowed_patch_ids:
                continue

            file_dict: dict[str, str | pathlib.Path] = {}
            file_dict['mask'] = label_path
            file_dict['patch_id'] = patch_id

            # Add modality
            for modality_name in self.bands:
                file_path = pathlib.Path(
                    str(label_path).replace(label_dir, modality_name)
                )
                if modality_name == 'AERIAL-RLT_PAN':
                    region_year = file_path.stem.split('_')[0]
                    new_region_year = region_year.replace(region_year[-4:], '195X')
                    new_file_path = pathlib.Path(
                        str(file_path).replace(region_year, new_region_year)
                    )
                    file_dict[modality_name] = new_file_path
                else:
                    file_dict[modality_name] = file_path

            files_list.append(file_dict)

        return files_list

    def _verify(self) -> None:
        """Verify dataset integrity and download missing files."""
        label_modality = (
            'AERIAL_LABEL-COSIA'
            if self.dataset_type == 'land_cover'
            else 'ALL_LABEL-LPIS'
        )

        for domain, years in self.domain_years.items():
            for year in years:
                for modality in [label_modality, *self.bands]:
                    year_str = '195X' if modality == 'AERIAL-RLT_PAN' else year
                    dir_name = f'{domain}-{year_str}_{modality}'
                    dir_path = pathlib.Path(self.root) / dir_name
                    zip_path = pathlib.Path(self.root) / f'{dir_name}.zip'

                    if dir_path.is_dir() and list(dir_path.rglob('*.tif')):
                        continue

                    if not zip_path.is_file():
                        if not self.download:
                            raise DatasetNotFoundError(self)
                        download_url(
                            f'{self.download_link}/{dir_name}.zip', str(self.root)
                        )

                    extract_archive(str(zip_path), str(self.root))
                    zip_path.unlink()

    def _load_mask(self, path: Path) -> Tensor:
        """Load a mask from a path.

        Args:
            path: path to the mask

        Returns:
            the mask as tensor
        """
        match self.dataset_type:
            case 'crop_type_3':
                num_bands = 3
            case 'crop_type_2':
                num_bands = 2
            case _:
                num_bands = 1

        with rasterio.open(str(path)) as f:
            array: np.typing.NDArray[np.uint8] = f.read(num_bands)
            tensor = torch.from_numpy(array).long()
        return tensor

    def _load_time_series(self, path: Path, num_bands: int) -> Tensor:
        """Load a time series from a path.

        Process it to be in the T * C * H * W format instead of the usual
        (T*C) * H * W format.

        Args:
            path: path to the time series
            num_bands: number of bands to load

        Returns:
            the time series as tensor
        """
        with rasterio.open(str(path)) as f:
            tensor = array_to_tensor(f.read()).float()
            # Reshape from (T*C) * H * W to T * C * H * W
            tensor = rearrange(tensor, '(t c) h w -> t c h w', c=num_bands)
        return tensor

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image
        Returns:
            Tensor: the loaded image
        """
        with rasterio.open(str(path)) as f:
            tensor = array_to_tensor(f.read()).float()
        return tensor

    def _plot_mask(self, mask: Tensor, ax: Axes, show_legend: bool = True) -> None:
        """Plot a label mask with appropriate colormap.

        Args:
            mask: Label mask tensor (H, W)
            ax: Matplotlib axes to plot on
            show_legend: Whether to show the legend
        """
        task = TASKS[self.dataset_type]
        class_names = task['classes']
        cmap = task['cmap']

        mask_np = mask.numpy()

        n_classes = len(class_names)
        bounds = np.arange(n_classes + 1) - 0.5
        norm = BoundaryNorm(bounds, n_classes)
        ax.imshow(mask_np, cmap=cmap, norm=norm)
        ax.set_title('Label Mask')

        if show_legend:
            present_classes = np.unique(mask_np)
            legend_elements = [
                Patch(facecolor=cmap(i), edgecolor='k', label=class_names[i])
                for i in present_classes
                if i < len(class_names)
            ]
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.0,
                fontsize='small',
            )

    def _plot_aerial_rgbi(self, data: Tensor, ax: Axes, title: str) -> None:
        """Plot aerial RGBI imagery.

        Args:
            data: Aerial RGBI tensor (C, H, W) with values in [0, 255]
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        data_np = data.numpy()

        # Select RGB bands and transpose from (C, H, W) to (H, W, C) for matplotlib
        rgb_image = data_np[:3].transpose(1, 2, 0)
        rgb_image = rgb_image / 255.0
        rgb_image = np.clip(rgb_image, 0, 1)
        ax.imshow(rgb_image)
        ax.set_title(title)

    def _plot_spot_rgbi(self, data: Tensor, ax: Axes, title: str) -> None:
        """Plot SPOT RGBI imagery (surface reflectance).

        Args:
            data: SPOT RGBI tensor (C, H, W) with surface reflectance values
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        # Take only RGB bands (first 3), normalize for better visualization, rearrange, convert to numpy
        image = rearrange(data[:3], 'c h w -> h w c')
        image = quantile_normalization(image)
        ax.imshow(image.numpy())
        ax.set_title(title)

    def _plot_dem(self, data: Tensor, ax: Axes, title: str) -> None:
        """Plot DEM elevation data.

        Args:
            data: DEM tensor (2, H, W) - DSM and DTM
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        data_np = data.numpy()

        dsm = data_np[0]
        dtm = data_np[1]

        chm = dtm - dsm
        ax.imshow(chm, cmap='gray', vmin=np.min(chm), vmax=np.max(chm))
        ax.set_title(title)

    def _plot_aerial_rlt_pan(self, data: Tensor, ax: Axes, title: str) -> None:
        """Plot aerial RLT PAN imagery.

        Args:
            data: Aerial RLT PAN tensor (C, H, W) with values in [0, 255]
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        data_np = data.numpy()
        data_np = data_np.transpose(1, 2, 0)
        ax.imshow(data_np, cmap='gray')
        ax.set_title(title)

    def _plot_sentinel2_ts(self, data: Tensor, ax: Axes, title: str) -> None:
        """Method to plot an example of Sentinel-2 time series data.

        To keep the same plot style as the other plots,
        we will only plot the last timepoint.
        We show the band B04, B03, B02 to get a RGB image.

        Args:
            data: Sentinel-2 time series tensor (C, H, W) with values in [0, 255]
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        data_np = data.numpy()

        # The format of the data is T * C * H * W
        last_timepoint = data_np[-1]

        # Map RGB band names to indices in sentinel2_ts_bands
        rgb_indices = [
            self.sentinel2_ts_bands.index(band) for band in self.sentinel2_ts_rgb_bands
        ]
        # Select RGB bands and transpose from (C, H, W) to (H, W, C) for matplotlib
        rgb_image = last_timepoint[rgb_indices].transpose(1, 2, 0)
        # Clip between 0 and 3000 (reflectance 0.0 to 0.3).
        # Stretch that to 0-255 for display.
        rgb_image = np.clip(rgb_image, 0, 3000)
        rgb_image = (rgb_image / 3000.0) * 255.0
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        ax.imshow(rgb_image)
        ax.set_title(title)

    def _plot_sentinel2_msk_sc(self, data: Tensor, ax: Axes, title: str) -> None:
        """Show the snow and cloud probability mask.

        Red for snow probability, blue for cloud probability.
        Adds a legend for the colors.

        Args:
            data: Sentinel-2 mask time series tensor (T, C, H, W)
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        data_np = data.numpy()
        # The format of the data is T * C * H * W
        last_timepoint = data_np[-1]
        snow_probability = last_timepoint[0]
        cloud_probability = last_timepoint[1]

        # Visualize both probability masks as channels in an image (H, W, 2)
        img = np.stack([snow_probability, cloud_probability], axis=-1)

        # Display both masks as an rgb image: snow=red (c 0), cloud=blue (c 1)
        rgb_img = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        rgb_img[..., 0] = (snow_probability / 100.0) ** (1 / 2)  # Red channel
        rgb_img[..., 2] = (cloud_probability / 100.0) ** (1 / 2)  # Blue channel
        # Square root of the probability to make the low probabilities more visible

        ax.imshow(rgb_img)
        ax.set_title(title)
        ax.axis('off')

        # Show only legend entries for probability masks that are present
        # We scale by 100, so >=5 is a significant probability
        present = []
        if np.any(snow_probability >= 5):
            present.append((0, 'Snow probability', 'red'))
        if np.any(cloud_probability >= 5):
            present.append((1, 'Cloud probability', 'blue'))

        legend_elements = [
            Patch(facecolor=color, edgecolor='k', label=label)
            for idx, label, color in present
        ]

        if present:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.0,
                fontsize='small',
            )

    def _plot_sentinel1_ts(self, data: Tensor, ax: Axes, title: str) -> None:
        """Method to plot an example of Sentinel-1 time series data.

        Shows the last timepoint as grayscale
        using VV band with percentile normalization.

        Args:
            data: Sentinel-1 time series tensor (T, C, H, W)
            ax: Matplotlib axes to plot on
            title: Title for the subplot
        """
        # T * C * H * W -> take last timepoint, VV band (index 0)
        vv = data[-1, 0].numpy()
        p2, p98 = np.percentile(vv, (2, 98))
        vv_norm = np.clip((vv - p2) / (p98 - p2 + 1e-6), 0, 1)
        ax.imshow(vv_norm, cmap='gray')
        ax.set_title(title)

    def plot(self, sample: Sample, suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        plot_data: dict[str, _PlotData] = {}

        plot_data['mask'] = {
            'plot_type': 'mask',
            'data': sample['mask'],
            'title': 'mask',
        }

        for key, (plot_type, title) in _PLOT_KEYS.items():
            if key in sample:
                plot_data[key] = {
                    'plot_type': plot_type,
                    'data': sample[key],
                    'title': title,
                }

        num_plots = len(plot_data)
        ncols = min(4, num_plots)
        nrows = (num_plots + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axs = np.atleast_1d(axs).flatten()
        for ax in axs.flat:
            ax.axis('off')

        for idx, (_, plot_info) in enumerate(plot_data.items()):
            plot_type = plot_info['plot_type']
            data = plot_info['data']
            title = plot_info['title']

            match plot_type:
                case 'mask':
                    self._plot_mask(data, axs[idx], show_legend=True)
                case 'aerial_rgbi':
                    self._plot_aerial_rgbi(data, axs[idx], title)
                case 'dem':
                    self._plot_dem(data, axs[idx], title)
                case 'spot_rgbi':
                    self._plot_spot_rgbi(data, axs[idx], title)
                case 'aerial_rlt_pan':
                    self._plot_aerial_rlt_pan(data, axs[idx], title)
                case 'sentinel2_ts':
                    self._plot_sentinel2_ts(data, axs[idx], title)
                case 'sentinel2_msk_sc':
                    self._plot_sentinel2_msk_sc(data, axs[idx], title)
                case 'sentinel1_asc_ts' | 'sentinel1_desc_ts':
                    self._plot_sentinel1_ts(data, axs[idx], title)

        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

        plt.tight_layout()
        return fig


class FLAIRHUBToy(FLAIRHUB):
    """Toy version of the FLAIRHUB dataset.

    For further information refer to :class:`~torchgeo.datasets.FLAIRHUB`.
    """

    download_link = 'https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_hub/FLAIR-HUB_TOY_DATASET.zip'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        split_column: Literal['split_toy'] = 'split_toy',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        bands: list[AvailableBands] | None = None,
        dataset_type: Literal['land_cover', 'crop_type'] = 'land_cover',
    ) -> None:
        """Initialize a new FLAIRHUBToy dataset instance.

        This is a toy/sample version of the FLAIR-HUB dataset intended for testing and
        development purposes. It contains a small subset of the full dataset with all
        available data modalities.

        Args:
            root: Root directory where toy dataset can be found or will be downloaded.
            split: One of ``train``, ``val``, or ``test``.
            split_column: Column name in the official splits GeoPackage.
            transforms: Optional transforms to apply to samples.
            download: If True, download the toy dataset if not found (~10 MB).
            bands: List of bands/modalities to load. See
                :class:`~torchgeo.datasets.FLAIRHUB` for available options.
                Defaults to None, which enables all bands.
            dataset_type: ``land_cover`` (19 classes) or ``crop_type``
                (23 classes).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            ValueError: If *dataset_type* is not ``land_cover`` or ``crop_type``.
            FileNotFoundError: Requested modality file is missing from the toy dataset.

        See Also:
            :class:`~torchgeo.datasets.FLAIRHUB`: Full dataset class for
                production use.

        .. versionadded:: 0.10
        """
        self.root_folder = pathlib.Path(root)
        modified_root = self.root_folder / 'FLAIR-HUB_TOY'
        super().__init__(
            root=modified_root,
            split=split,
            split_column=split_column,
            transforms=transforms,
            download=download,
            bands=bands,
            dataset_type=dataset_type,
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        toy_dir = pathlib.Path(self.root)
        toy_zip = self.root_folder / 'FLAIR-HUB_TOY_DATASET.zip'

        if toy_dir.is_dir():
            return

        if not toy_zip.is_file():
            if not self.download:
                raise DatasetNotFoundError(self)
            download_url(self.download_link, self.root_folder)

        extract_archive(str(toy_zip), str(self.root_folder))
        self.files = self._load_files()
