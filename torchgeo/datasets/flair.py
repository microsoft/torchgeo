# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""FLAIR (French Land cover from Aerospace ImageRy) datasets."""

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
    check_integrity,
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
    Each domain-year combination has multiple modalities available for download.
    The full dataset contains 66 unique domains (D004-D091, non-consecutive), with years ranging from 2017-2022.
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

    sha256s: ClassVar[dict[str, str]] = {
        'D004-195X_AERIAL-RLT_PAN.zip': '7f1995cf3300a8622905f077d77aa18ff0d43d1245b1976638d9d32900524004',
        'D004-2021_AERIAL_LABEL-COSIA.zip': 'b55f6fe27f6265ffece3215b9c3769ad1019094e314c3889dc514c2988cd4948',
        'D004-2021_AERIAL_RGBI.zip': 'a0522a8a57a76639aca87b6c4d8340eae2697c0f5c9696be010c62a156b15aaf',
        'D004-2021_ALL_LABEL-LPIS.zip': '2c458e7ae9b6773b85d0d8a1449fe67fa1f4f7f15941bf165d1443fca6562871',
        'D004-2021_DEM_ELEV.zip': 'b829fd47bf81f9e0c00c79ffecc127d23fdd1c3cdffd134fe7b039af8099458a',
        'D004-2021_SENTINEL1-ASC_TS.zip': 'ff28c9c8daf44b9d6587bc8b79693708a667a65b9842b769010afee8e918501a',
        'D004-2021_SENTINEL1-DESC_TS.zip': '6db928333346cc93235dde582b03d7dc7a385326f66966739623c5f58fbe8f82',
        'D004-2021_SENTINEL2_MSK-SC.zip': '98497adf59414147a5c8f1e8b857fdf8b9481bddb917afee01da78b4f5d75ba8',
        'D004-2021_SENTINEL2_TS.zip': '75fca520e28c203e2fa53b9f6b3ce063a1829de903c4e12fe1e0324eb3549302',
        'D004-2021_SPOT_RGBI.zip': '4f98d93dd3fe8340dd434628b760f632665f3305c1484a71e32ab8cd58d7ad0d',
        'D005-195X_AERIAL-RLT_PAN.zip': '4d92fc4db661f61a47c6d3b17ae5bd475dab1fd57bf1357d0358c8ba8dbe07e1',
        'D005-2018_AERIAL_LABEL-COSIA.zip': 'cefb66a7c69d46aa3dae15ec95677e8b0c8238cffbc0eb4914ecdaa0733dbbbb',
        'D005-2018_AERIAL_RGBI.zip': '4b5f7f38af52f97864f281ec0f9ff65df3fc928c3c038d091da9398292020ed0',
        'D005-2018_ALL_LABEL-LPIS.zip': 'a5e242e3cf99395ea62a4acd67eab1f8f973b20977e0eba48194b42d2a1e80e4',
        'D005-2018_DEM_ELEV.zip': '63325c141b03740b0c7eb975c9742b0915338735924809d944b270a54a1df031',
        'D005-2018_SENTINEL1-ASC_TS.zip': '05935d602ebbd9687a04a8ec6c0c0091c997e27c3cb9bd410a1d9dec38a54fba',
        'D005-2018_SENTINEL1-DESC_TS.zip': '0f7fe789846536e08c16ffb87b2910b47a4138778443475dce9ab05850899bca',
        'D005-2018_SENTINEL2_MSK-SC.zip': '168cdf695abff4ac48f1cc3658ec84e92d546a88766499b2a0de08c7c463c253',
        'D005-2018_SENTINEL2_TS.zip': '46f0e542bd3a8619c6be4c83414cc090ed283759222545d04131aa5fe9d2c52d',
        'D005-2018_SPOT_RGBI.zip': 'f2461d659a9c640b1764f4420cc2ba2f9bed647130cdeeb534dd678bee747728',
        'D006-195X_AERIAL-RLT_PAN.zip': 'db7c2e3a9456cb95b979bcd1f8bd287cac09399a94aa0ef8ac15efeec42393bd',
        'D006-2020_AERIAL_LABEL-COSIA.zip': 'fd24700a5b62a162b62de2793f61c4bd276d36850c5da79700a01987bffc153a',
        'D006-2020_AERIAL_RGBI.zip': 'b2c6dcb620ca147dfa23487d9317b652b571d3407f1ac72f158db5f2f02efe3a',
        'D006-2020_ALL_LABEL-LPIS.zip': 'c7378d22986ed99811787e3232ff2f5e320971e05cac05f9798d1dec620a0746',
        'D006-2020_DEM_ELEV.zip': '21420884b9dd31e09caa23901cceaf90dd9a2c77c5fc570b982323914b4bce8e',
        'D006-2020_SENTINEL1-ASC_TS.zip': 'c013a9a8d3d553fc093920c7b44afeac036d07714625cf2ae8c0b1c71b9bc900',
        'D006-2020_SENTINEL1-DESC_TS.zip': '7833a6a865168fbd8b39703b15fd5b8f29032d51fde19d2cd7e922bce03d3554',
        'D006-2020_SENTINEL2_MSK-SC.zip': '76b50885e843c91716b610490b8d966d2cf9df78ffa4675d05852aa71cd6b99a',
        'D006-2020_SENTINEL2_TS.zip': '5c8e44569da4aabb5b356df69486fd73b2891adfb70767601f5e9947b9c77f24',
        'D006-2020_SPOT_RGBI.zip': '32eebe6067eb87e6bc51a01fa573c32e93c40ce07408a591c799cfd16dee3d5c',
        'D007-195X_AERIAL-RLT_PAN.zip': '445c9d5f88089ac294464b11ff4e75844a28d0459f844145f5e42b6e87d0b3af',
        'D007-2020_AERIAL_LABEL-COSIA.zip': 'cda71beb80130a322beca69ef46a26ec0467ef9e89d801831670e5b249d48c08',
        'D007-2020_AERIAL_RGBI.zip': 'ab4bdd343145b712804a0f8a70ade9f0daf8e95d1fbbc566ae4dbbfdf0e54e0d',
        'D007-2020_ALL_LABEL-LPIS.zip': '27cf4316beec1e93a198c6314af856b06029cc959665e2a3587470a302c7e5ed',
        'D007-2020_DEM_ELEV.zip': '13fbc54d76a50b169269f5969447f9ad3ced214f3d52d2a708a6e768ac239156',
        'D007-2020_SENTINEL1-ASC_TS.zip': 'a7cf5fa242d58b1e6bdcd13c59b34c8a57c33531a56096e1a777a6cd4ef88090',
        'D007-2020_SENTINEL1-DESC_TS.zip': '523ae22990ed7a49157f833fe00c201b5827a8487cda646c001b628e36491e97',
        'D007-2020_SENTINEL2_MSK-SC.zip': 'dc613a4cdadc410ef7620b240495fedf19b1f248ad9611c2e1cce8e3a507bfdd',
        'D007-2020_SENTINEL2_TS.zip': '36981a2c500b4fc1aec8497ed736a5104cee6f50f3543c83ec66c518edc79edd',
        'D007-2020_SPOT_RGBI.zip': 'b0d891dfde16d7159bfb89be6298c8687d8d3618c9b5c560b8ca922911959902',
        'D008-195X_AERIAL-RLT_PAN.zip': '6587ffe9ad9bc3ad6d3c734c0a8397b3652743988e7b9b555940e7b1cde705c5',
        'D008-2019_AERIAL_LABEL-COSIA.zip': '2c1cb0e297d50e94128648872d103fa3a9186a736f8b1900977d7146f12f512f',
        'D008-2019_AERIAL_RGBI.zip': '791cb8b650c9c45ebdb53fe9cfd333c4aeb0809a9c5381c7a722de141f42c84d',
        'D008-2019_ALL_LABEL-LPIS.zip': 'a993ffaff6507daa5470fa05b6971e6d3d3648eef5c0d597e7d654dd4fd6ff6f',
        'D008-2019_DEM_ELEV.zip': '9b4b9d54d09097a18d6ee5734de872cb6af8955922000d6d4e6458de189cd93e',
        'D008-2019_SENTINEL1-ASC_TS.zip': '2e8a014464af057d4332f8c2bc4d6b724dc5d10222035c76284be77a8cdd6912',
        'D008-2019_SENTINEL1-DESC_TS.zip': '5c22f0b7a14a220d7f01efd9af6ea4636b1b18dc9df60b1d1998e18e986017b6',
        'D008-2019_SENTINEL2_MSK-SC.zip': '56d7961944a8ed794a599fb87381185e5e7b933dfd5d871b0b09e66ebca50121',
        'D008-2019_SENTINEL2_TS.zip': 'f4cb96e8fba7b59102e5fc10d782565f41a3d5dc9a6859416046d8fb70d9b1f5',
        'D008-2019_SPOT_RGBI.zip': '6dbce88f5ba16b24e1d742b6fc5fba5f8dbc4748bf3a28cd3f8e4b84636168b8',
        'D009-195X_AERIAL-RLT_PAN.zip': '255fc8db04844295027b12682cb465209b95bb78c5c1326c8906f0b44a1c1f80',
        'D009-2019_AERIAL_LABEL-COSIA.zip': '824ee4df9524d6a44a64a8aee61bcb646292bf1cb6a701b1c8387d4e7cd12105',
        'D009-2019_AERIAL_RGBI.zip': '982e453d1ad3df769e8b720389e96e81fd51014ecbeaf7978c3e592fbb0d7517',
        'D009-2019_ALL_LABEL-LPIS.zip': '393c7196dd39563cda3b530e78d60ca201bc7ec4e3747ac1e24f6bf79e6b441b',
        'D009-2019_DEM_ELEV.zip': 'adf5263de61228260fc669bfb53d47cf3163fdad11a3c962cf2aec413e44cfec',
        'D009-2019_SENTINEL1-ASC_TS.zip': '7af6f3b2765fad2d17071df49942bfe7304ded6b499f5a48b2500273d105f32d',
        'D009-2019_SENTINEL1-DESC_TS.zip': 'e528c5db20f83c81be99160265824a68267d99cd9a9e5768e7ec3f85e0a288a1',
        'D009-2019_SENTINEL2_MSK-SC.zip': 'ab30f71f1d357c4e644614fdf9bc4c9d34cd4b8454f70b0d6ba4e0b13b453cab',
        'D009-2019_SENTINEL2_TS.zip': 'c247a602ec51c3179f286a4486c4017b40e1804a1636f1e7531be05ca5325b5c',
        'D009-2019_SPOT_RGBI.zip': '993c599dc92d7d43e5bda5e4133d2d24e4710db3a131d1de684fb9a320a409d9',
        'D010-195X_AERIAL-RLT_PAN.zip': 'be57869cc12b0baa1b0554654ac0f451aebd3990f08176a4ea0e03cc6f351137',
        'D010-2019_AERIAL_LABEL-COSIA.zip': '5820d84cea9611bffd676abad028eba7c73625ce77a83cae15ecb6af96e44e31',
        'D010-2019_AERIAL_RGBI.zip': 'da7a879d941ffd6dca9e12742ee73c1409a69bb6313adecb7ef5afda460a4256',
        'D010-2019_ALL_LABEL-LPIS.zip': '839d79d896043aad93a17c56be0af1837c70a3ed9d56c7eb5a7084a618883fe0',
        'D010-2019_DEM_ELEV.zip': '48c29d875497c4269873689bbfd8c106ab3a8a8ae93fff64b3efc6fb24e00fdc',
        'D010-2019_SENTINEL1-ASC_TS.zip': '7ae8bb5da8d63ece35f245fa9c4702d25c543626df9b1fb8cf76bb2a95b79b6e',
        'D010-2019_SENTINEL1-DESC_TS.zip': '5ffcf949bb3e175142332aa1137ae00406c0dbd74c34b9a94855639807234edc',
        'D010-2019_SENTINEL2_MSK-SC.zip': 'c273a4002883cae4befaf50c59040c508037475d9c789ade60560c47eccf20a5',
        'D010-2019_SENTINEL2_TS.zip': 'd1cefdcbb67321867c0dadee9894d7e5feb225d6380c653a863a7f6deb75e645',
        'D010-2019_SPOT_RGBI.zip': '2bbf3b55d2a9022fba7bad26ae5a33102d8ea2ebc513895a5c55c6350abfd164',
        'D011-195X_AERIAL-RLT_PAN.zip': '656c1810512fe9a1ed8a9585908a3ba03e4bcda29e1b3e19e24d47f92e8d93ae',
        'D011-2021_AERIAL_LABEL-COSIA.zip': 'ef3fe27fd2469eff665062725269b567be7a450dfc0853da3c93bf766e2ecb31',
        'D011-2021_AERIAL_RGBI.zip': '505e25904b8d15391b83536d7170c705417a765de53e47adcdc4bfd1675530a8',
        'D011-2021_ALL_LABEL-LPIS.zip': 'd17208b73a71469211d79d788c453d6b8bde8535f17603884f90807655b9e0b6',
        'D011-2021_DEM_ELEV.zip': '0bc3906ad60fa681fec91965c79f57ba2c2239018af61a38a08002097827abf8',
        'D011-2021_SENTINEL1-ASC_TS.zip': 'ebcd5cb2e5ed5e5c787b6be6b60338de4f7e0c8f767ce0f6ea3f40a7e69c7052',
        'D011-2021_SENTINEL1-DESC_TS.zip': '2afc5951f9a74acc02f8b4170e930b530fcaffb13a3c8889d8de0387176c208b',
        'D011-2021_SENTINEL2_MSK-SC.zip': 'ee18709a437dac3828918a0461c888c3580db9d6cf48a2121023e285f8eed944',
        'D011-2021_SENTINEL2_TS.zip': '2439f4ae5a26e006e5f5e0c617ccd0531ae674dfa4913f3974e438ac36ca5821',
        'D011-2021_SPOT_RGBI.zip': '533b25b2baa426a779789bc524f19c8ddf441cf4d4fd7d5ccf4123385092e84a',
        'D012-195X_AERIAL-RLT_PAN.zip': 'd17b7a801d09c205b458989ab9d4af2cdaeeab3decfb62fbd4f040cff5fe8857',
        'D012-2019_AERIAL_LABEL-COSIA.zip': '4f9d1e14f4741fed293adba7ec83b31a0b634db6ec869a919345a08667b74f68',
        'D012-2019_AERIAL_RGBI.zip': '5ce860343e33aac69c74608cddfaef364f2b025385e22260c54dd1a05a296f8c',
        'D012-2019_ALL_LABEL-LPIS.zip': 'f36899a905a7dc7723fa1799d7510e4238177ed913e1d6c59cb29ac36b49e8fa',
        'D012-2019_DEM_ELEV.zip': 'bf9135b2a21236584e6937f0b32c0e24bb3ca2c82131f56d6654831fc8daa8ba',
        'D012-2019_SENTINEL1-ASC_TS.zip': 'bf4e87edd21b0a3d1907c54662a319b5bf78335f06f76554c6d1b7071bc5c037',
        'D012-2019_SENTINEL1-DESC_TS.zip': '0279d5da6df6a9bef5b72028f67f9fc788a80f11789ecb3e4102880c2cc993a7',
        'D012-2019_SENTINEL2_MSK-SC.zip': '8e03754171745b2e50e7e6d52656d16d974be36476f6b7f86926272249c551e6',
        'D012-2019_SENTINEL2_TS.zip': 'e6481dcacc242306b5686ea1f2b9b8231f136af3970e229593e4315046fdff0f',
        'D012-2019_SPOT_RGBI.zip': 'e63e8703970cfe2b4fa036353e1f70a8ea34b85be2994b48dd8672f556fd56ab',
        'D013-195X_AERIAL-RLT_PAN.zip': '1086e37084f49760d97c9c30be69dc5c4a6e93d0e146fd21d51a988aadb4dfcc',
        'D013-2020_AERIAL_LABEL-COSIA.zip': '1adfb6315ac2df44690ec4c92cb9e24c5c1f483296710da375bd4d6a4269fde4',
        'D013-2020_AERIAL_RGBI.zip': '92ad203a6bc05ad611967c7704e81732ec142cd4f882c0591a8d082a4e28cee6',
        'D013-2020_ALL_LABEL-LPIS.zip': '58f2c7d6085b3a14672cdcdce437f4a715b7dfdc47dd9a99e89b0021fe03be4b',
        'D013-2020_DEM_ELEV.zip': '8daf3af355c7198de1c0ca166f292c0c0e845b429c49dd9c0a4f753b70142767',
        'D013-2020_SENTINEL1-ASC_TS.zip': '90b0872b352fc44ac318df6ee4b6778ba41923377bd3615f3821b5b320fa175f',
        'D013-2020_SENTINEL1-DESC_TS.zip': '3850d9dd14996b846b52edb60f56dfbdadb45106321304951ac0682f964bd08f',
        'D013-2020_SENTINEL2_MSK-SC.zip': '2a1c820cfbb948866bf3b4d354a81cb22182d8c696f118421d2888a38f1fe3c0',
        'D013-2020_SENTINEL2_TS.zip': '59d5ac93454d8f6a2c75cbdb172349123241fa6fe97d202e8ed7974f02022b31',
        'D013-2020_SPOT_RGBI.zip': '623eb2fcf35466f6b8410378a260c2c208ad27e0a26546802110ce7644abcae3',
        'D014-195X_AERIAL-RLT_PAN.zip': '48e9045c0d96a8118478ac319ecd9b3ac154d6330bcffb5125ce387953cc39b4',
        'D014-2020_AERIAL_LABEL-COSIA.zip': '048b12179f0b3167e32f1f3e5626268aa0fcb792b6ea5ac794212bf9fbb1de93',
        'D014-2020_AERIAL_RGBI.zip': '0d2c9bc41874f77a4772cf5018f20b7599a361f0319be1b7d4b92c61c17b0605',
        'D014-2020_ALL_LABEL-LPIS.zip': '3e76006145e5fbf31a57fcc897b2692a8ac5992affd16ddd4fa5b08fe6d0edde',
        'D014-2020_DEM_ELEV.zip': '728be0834b9b15ce60e6c0a88ecbf64687e288b07516fd1590cfd007db16d41f',
        'D014-2020_SENTINEL1-ASC_TS.zip': 'f2882f65cabe4a6a9a83e3e048c38e539451ea3daaccfaa164956d0cb6b14873',
        'D014-2020_SENTINEL1-DESC_TS.zip': 'bcf66f2b8f647baaa7dbd6ab0749541b4a45b1985368aa9eda06c250464a65ea',
        'D014-2020_SENTINEL2_MSK-SC.zip': 'c188c198d3950ba11f9f1d726dadead64e8793570bc66c5e0d35810c4f0fbb36',
        'D014-2020_SENTINEL2_TS.zip': 'c918cb416f72957280653ee8a529b5ca85980fa53b2bc02ccb47be0304dc8cde',
        'D014-2020_SPOT_RGBI.zip': 'fd5115536e67fc825eb8bab4e7412e371887c7d5f280003f52923e9bbef05ae7',
        'D015-195X_AERIAL-RLT_PAN.zip': '38fe9f4b92b8d45fe769b12faf4b0ed047005dae90deda9ec8a3b47d039ad223',
        'D015-2020_AERIAL_LABEL-COSIA.zip': '9361bdad5b7187f2fad1b9a594ef0087f60a68f344b20ef7dddec50d3e9707b9',
        'D015-2020_AERIAL_RGBI.zip': '53a4fc5b84b46bbce0cd93117efdc83fe9c2a09da2839d7175098c7ed39508c6',
        'D015-2020_ALL_LABEL-LPIS.zip': '99af93dabfad74f8be61a8f11215e1e875384846ec5c11e9320a177ee4b4c804',
        'D015-2020_DEM_ELEV.zip': 'fe22ac42d9ea297d4126b204b76aa198067128c39dcafaf2d0cda73d9c78cd7f',
        'D015-2020_SENTINEL1-ASC_TS.zip': '24fc1fa6d07de3fb081dc732afd1cb681320ec1525da75823194349a302fd190',
        'D015-2020_SENTINEL1-DESC_TS.zip': '00e50fd7a9ff2a9861285022eb433a3df191c33cb17ecf09574aa16f7f75af67',
        'D015-2020_SENTINEL2_MSK-SC.zip': '5d347c5b6242c5e6327bb884c4729dbde769278c51d01c48f3b758224baed7d9',
        'D015-2020_SENTINEL2_TS.zip': '806020b2c39bd5af394648185ec42d0f82ff7b2a8302e03ec6d9335e16bb88de',
        'D015-2020_SPOT_RGBI.zip': '5ddc8db57c6066046d02308185909dab051998136aa44ee9b0c3ba58e5410be8',
        'D016-195X_AERIAL-RLT_PAN.zip': '0462a91a694350372292a2298615dbe2286b7b2fd8392bc452297ef07fbfaf78',
        'D016-2020_AERIAL_LABEL-COSIA.zip': '4e2bca0d1584f976c360c4cc9a0727fd48d1d8cb3665c9cfd4045d25395f5fa2',
        'D016-2020_AERIAL_RGBI.zip': '11d57a1001d0fc71ecc479518ec1083d3690514779224bbe714ce2f5ff5d6a22',
        'D016-2020_ALL_LABEL-LPIS.zip': '6523f10293e9989dcecbdfc9f05da4efcc7abc80e7a285d0690d0074c2f2ae83',
        'D016-2020_DEM_ELEV.zip': 'd1419399fc9f8326f75e4d19d8eb856613591f1b8bb7b91e9db06d01b13d4d42',
        'D016-2020_SENTINEL1-ASC_TS.zip': '336c92a300094e582463c11456c2cc8ba5ba84f10071b7a67deb6a062ef13b15',
        'D016-2020_SENTINEL1-DESC_TS.zip': 'f84258f56c8027d7a689e01ea277c2a11b67e37b98ca0babcff546d748ceadee',
        'D016-2020_SENTINEL2_MSK-SC.zip': '28c2af5ee9c72172e1c0d38f39582b9b40d9664c65cab1381781178650980389',
        'D016-2020_SENTINEL2_TS.zip': '54dcb6b92cb8c5330807c6e1012bd32c402c66a04957c1fef2f0d1fe944ec0e0',
        'D016-2020_SPOT_RGBI.zip': '02926eb4f1f7e3ccebc70d8fc2e3797436754e7e7f269f8e60b87e1a9a4adc37',
        'D017-195X_AERIAL-RLT_PAN.zip': '640e8195bb53c13210f356b0a98ce7a1135d0d574d2112c0fa58412d9a2b3220',
        'D017-2018_AERIAL_LABEL-COSIA.zip': '8b15cf820515ddf97ae2c3b56c73c3458c83ffc04290e8d3a32765d1cf94fec7',
        'D017-2018_AERIAL_RGBI.zip': 'de55ed84be2e9332ac6ad9e9ac956bb008a3ab78eea79d1ccf6f453796c00fb2',
        'D017-2018_ALL_LABEL-LPIS.zip': 'eb21edc33d726e8c634e0e38b8fa5df99deb21bfdf0e0bf68e17cd61dc5d46be',
        'D017-2018_DEM_ELEV.zip': 'd255e8472db63f032211b83c0b542f12557bdc2f40eb4ed36dba6ebceb46b8fc',
        'D017-2018_SENTINEL1-ASC_TS.zip': '2bff60b62c295de7151bcf7645bf92d11adf76d134349a6f8883a620eb5ed5f8',
        'D017-2018_SENTINEL1-DESC_TS.zip': '2341b5c532857a4da639a6bbe999b1ae71865bf7cc7c87efe8cb9affac69978c',
        'D017-2018_SENTINEL2_MSK-SC.zip': '72f273381af9ada66ef7d7402edad4ebbf3bcbab3cf6184ee503a34422fa3eb5',
        'D017-2018_SENTINEL2_TS.zip': 'ebe2638eb5a7ae1c6a882ba21c64cc42250b9809728f9ac4577b8a16270841d0',
        'D017-2018_SPOT_RGBI.zip': '4c42a009e9b47be1b0fb11a680e6bc94b3246082b90fd88c4f3fe2b2f7665979',
        'D017-2021_AERIAL_LABEL-COSIA.zip': 'cba5724234efc38060baec9d64ce85bfd3424f56054f37bbd7688de182e954fc',
        'D017-2021_AERIAL_RGBI.zip': '0192a766862d3a5f11022954e697264e3c1d889131c2f29dc67823778ea14966',
        'D017-2021_ALL_LABEL-LPIS.zip': '100976474df1142dd36b8ff7994d201fcf607455f8157ab477364ee8f6530b97',
        'D017-2021_DEM_ELEV.zip': '8f3fe39822732649d4f8ff0105f784dd8f91c691c54dc56d2f8ad7659bc537a3',
        'D017-2021_SENTINEL1-ASC_TS.zip': 'f407f528734ffbc06be6688a214c20bc9abe27d75a4a39672f07192026e008de',
        'D017-2021_SENTINEL1-DESC_TS.zip': 'd93c718ccab2af9ddfc07fcdb03ae504b35882ac4bde13ab2c30cb75097577d7',
        'D017-2021_SENTINEL2_MSK-SC.zip': 'ff23fb586f1a73ba61efc32d30d3d2b0be0bfcf923d129390ac43bda490d6177',
        'D017-2021_SENTINEL2_TS.zip': '7975e1a38721c648b577d1182eda5f94f16cf99a0e8e782439613c854cb235ed',
        'D017-2021_SPOT_RGBI.zip': '8f1291c89334c5f14f473591590dc2b69ad6f9a5ee035712abd14e059c4dc537',
        'D018-195X_AERIAL-RLT_PAN.zip': '6865be61ca4d11321e104c91d183aa2354e71cc4c54ff454920ef60b6412322e',
        'D018-2020_AERIAL_LABEL-COSIA.zip': '6d5e50c268295c7470ec14fb925fc9256093ffa0d1dd48e0e7022951f87f7229',
        'D018-2020_AERIAL_RGBI.zip': '8bc59ab7063af50e70cf53b10811b294fb54a9c12bd7947f6f5d0871e7d3c02b',
        'D018-2020_ALL_LABEL-LPIS.zip': '5379337df05826d7fb17f52314c35619e624e36a5308c19032d11c4326eb934b',
        'D018-2020_DEM_ELEV.zip': 'd3241172ff19b7d99a7166e585f1c787a13ce801f3e3847f3680dc083f4c7a0d',
        'D018-2020_SENTINEL1-ASC_TS.zip': '5ff7a279a672621394e37a54241b93cceb5282d759f6074f0ef720425c381cba',
        'D018-2020_SENTINEL1-DESC_TS.zip': '8c1a654a056f9cb0aa710a5a5d0308c65f3975d457d98c7d2e07513192312d63',
        'D018-2020_SENTINEL2_MSK-SC.zip': '567685b472fa43a9596d2ee94d058b9989ea6da9bae9b03c217d76bc75b99d53',
        'D018-2020_SENTINEL2_TS.zip': 'b576cb913a17cac017c1fb1ef03b64d5a587518b7a584b9935897c83d492ff98',
        'D018-2020_SPOT_RGBI.zip': 'a7d6744598a9d571ffc25eca150700c44bd0294850b92c1426b82459662a9cc8',
        'D020-195X_AERIAL-RLT_PAN.zip': '70380a295323c8908df63e827239c4ff83b8a54124b235d268694d81befaa81d',
        'D020-2019_AERIAL_LABEL-COSIA.zip': 'e10d7c6868ff60485db4ba1aa3ac42a426cc4b0f5a0dd73c7a66700d01a64a3e',
        'D020-2019_AERIAL_RGBI.zip': 'c738f4cfe8cb09a4b8a408c41645d8c0e7dd705a2fdfbe52838d7b8a957fc350',
        'D020-2019_ALL_LABEL-LPIS.zip': '5916825247989b7fb2ae2350d82ec9760a0d061b7fd63cd63ddfd0256a830f21',
        'D020-2019_DEM_ELEV.zip': 'a296ab6055f9dc6ce8d8d86d5b7d2e366b7bc0e05fe5f3efc5995b5fe56e899d',
        'D020-2019_SENTINEL1-ASC_TS.zip': '1f0f64060332cee816e380d0737bc150640b9c94ab62151197ee1740b856e542',
        'D020-2019_SENTINEL1-DESC_TS.zip': 'c79503aa737a8700c7c4fc04b290ec97f3f4706e517307b877306bc8d30028cc',
        'D020-2019_SENTINEL2_MSK-SC.zip': '8a04e3ce6df9024742b78f393e8d8460a54827cbefc67d3387a6696da6d1fad9',
        'D020-2019_SENTINEL2_TS.zip': 'f9041fa76a5ba221a8e51ce0c5fa9918472551d9aa5c5711238e655b3dc4bf24',
        'D020-2019_SPOT_RGBI.zip': '6c307986d0feafead6f87eaaf97301aee3030cca6b9c5d491384921f676308cc',
        'D021-195X_AERIAL-RLT_PAN.zip': '86feff16511db29db2f2fc31f48b8d827122ea059830ea1c22943fa500e43000',
        'D021-2020_AERIAL_LABEL-COSIA.zip': '444b6ae491cd2f6ad4a6fdcf1d4c68ba3fdfbab5dafaebb0d38da74cb4254549',
        'D021-2020_AERIAL_RGBI.zip': 'c124d24aefe4363b16e7d77dca21c45c76c30c52f2a15b9af57c6ddfaad22a5e',
        'D021-2020_ALL_LABEL-LPIS.zip': '594d31179afd0aedeb7c82aeb4f101694cba2cf47b47d73fe836c8821a127fd1',
        'D021-2020_DEM_ELEV.zip': '8ae1751b9502671a05efe02bb04c29fa4658ead18991c746acdc9d3aca7ff1c7',
        'D021-2020_SENTINEL1-ASC_TS.zip': '3836aca9a97bb6d61fd8de53cdfcc6d8415bcb3da8ecea8302fadeb497b84a98',
        'D021-2020_SENTINEL1-DESC_TS.zip': '38fca135d643e0cf5f8e2f33517c2db6b85f0c705288d373c29d4adceb85a103',
        'D021-2020_SENTINEL2_MSK-SC.zip': '6f46f335fc1370ba3a7d06a17e496f9886ac637b610e663351b38a208e8d1916',
        'D021-2020_SENTINEL2_TS.zip': '431023657813c53855f526fb704cbc63fc6c8cd9e5ce36f9310c377e299a48ee',
        'D021-2020_SPOT_RGBI.zip': '6f6ed41c7595cbad928e7958c409f168ed783a13d41c27c098a9e8fc2b055f35',
        'D022-195X_AERIAL-RLT_PAN.zip': '7e9bfffb5d93072bce116b1ec4a4557a61431abb136a989d162b6f307868518e',
        'D022-2021_AERIAL_LABEL-COSIA.zip': '45e1f9501400a1041fa307612a5878e34cce6571ef063e0725cca64b847adbd9',
        'D022-2021_AERIAL_RGBI.zip': 'b22c65f8cb3341c71853de7b215822b642e14b30ec741d532b7b50ff4d05f3c6',
        'D022-2021_ALL_LABEL-LPIS.zip': 'c69661f3009adbfd07cd7436374f3000f4388cfd966f583bb117a09bad548d21',
        'D022-2021_DEM_ELEV.zip': '7ca339f64ce3d402ee28bbacae66a27dcbdf8b9a1f2afdd5ba80273883d7c967',
        'D022-2021_SENTINEL1-ASC_TS.zip': '0ab56cd2576b936db77271568bc283128e0d4c24fe5ab69f31b318f167da2b8b',
        'D022-2021_SENTINEL1-DESC_TS.zip': '6957aadb5ad0099ee3b3c2235b16e457df91a98fe7b1c7de2843011f269e48ad',
        'D022-2021_SENTINEL2_MSK-SC.zip': 'f18addf4338290fe5ebd77c66cc631ac711956575e12f328bbaa677fa93b5cbc',
        'D022-2021_SENTINEL2_TS.zip': '5b84a9dde715bdf7b94bd994af04c9d71e5731ab0f60ea560fa46a6409d93314',
        'D022-2021_SPOT_RGBI.zip': '6f5216c1aec1c837c13d1f0cbd444aa55b4db4932c77955eddd08e06a94b3475',
        'D023-195X_AERIAL-RLT_PAN.zip': '2d08185fe834ec94656f23f50445a4cb57fba02b6d45824a1ffafff0e5327493',
        'D023-2020_AERIAL_LABEL-COSIA.zip': 'a92d34d95f53998fd93cea4de9be723f44842968467dd74b6380d3f518e68a8a',
        'D023-2020_AERIAL_RGBI.zip': '3cc48fc33aad92d5f1ed4a4b19d710bf9af6e80c1195ecc8818a38c65f220ced',
        'D023-2020_ALL_LABEL-LPIS.zip': 'e1aee590c9529713094ae507cf4feeea81765f181055783960c7054b298ba73f',
        'D023-2020_DEM_ELEV.zip': 'f3945f55b39499aee8fab0117544df55a2991af15df0905b719af583e11892c0',
        'D023-2020_SENTINEL1-ASC_TS.zip': '5cffd82fac43c21e88eb53549eebedd64e6549a0f1b6e97b071e94ef46ca94e0',
        'D023-2020_SENTINEL1-DESC_TS.zip': '643989824d76b0ff2879f05bc21f8a2364bd25925be9740c7bd9b3a885610b01',
        'D023-2020_SENTINEL2_MSK-SC.zip': '9d9b229bbd0854edc5159267fe9f4f2975f40d6f564104427c156d395936e92f',
        'D023-2020_SENTINEL2_TS.zip': 'c3de23f76c4c4d924c4ca26495b1475ef6a81aca5cd6b63c9fe6069cc94c1be5',
        'D023-2020_SPOT_RGBI.zip': 'ee1a4927fad8ef3e495416bc22708b52b2aaa3c2abfe696c316ed02cf249ae10',
        'D024047-195X_AERIAL-RLT_PAN.zip': '0fc5e9eb31ed12f81ba1277150dbab9f6d5d71492c82024b2eeb666aa69ff86b',
        'D024047-2021_AERIAL_LABEL-COSIA.zip': 'bc14d578dcfff821ed7e61ce67161fb3d2cf2d4cd5bbfa7fc28649d475aeb914',
        'D024047-2021_AERIAL_RGBI.zip': '8be1cd2dcba4027ebff7b22cc104fb1c51a6d1ae412df5bab29d5dec6d56ce87',
        'D024047-2021_ALL_LABEL-LPIS.zip': '756e720277cb312868d91b6caef8f5eecfba5d8649ee3b89a5bcfd19c877eaef',
        'D024047-2021_DEM_ELEV.zip': '1fb36b37b44c73c75798366e49559320e2bd8d2101d2c99f17f587c8775f49f4',
        'D024047-2021_SENTINEL1-ASC_TS.zip': '471b4581d2f9c53f42754f534788526d791f941e08ababdac080a746f3b7aad6',
        'D024047-2021_SENTINEL1-DESC_TS.zip': '4d3b10766c4dbb4a3b199eb731b9c74504503be6546c92e4b4374fb37dbda390',
        'D024047-2021_SENTINEL2_MSK-SC.zip': '897e6d00a4fc0813e9d8b8d115505f07c63f28b662800f775fd0b9178d6908b4',
        'D024047-2021_SENTINEL2_TS.zip': 'f3119e22584b12b221cbf01f3482e0f67d8e91b5e61d4387d9fdab4e26579650',
        'D024047-2021_SPOT_RGBI.zip': '274ae816c73679e785065c417470468b22a928c52badb7dd4db62dd2416e05e9',
        'D025039-195X_AERIAL-RLT_PAN.zip': '97f293bc83e310dbeb189b9d7538dfb83352e191bc4ec1cb0c9351d4e9104a2b',
        'D025039-2020_AERIAL_LABEL-COSIA.zip': '3e7ab4f7d86a4ada206afbe65c16c49df3bc733a1110346deb2a664af17cad53',
        'D025039-2020_AERIAL_RGBI.zip': '550c925ed880c4ceeceedc48dcc15f58e1ca80ceab8d38006427d046c8632fa2',
        'D025039-2020_ALL_LABEL-LPIS.zip': '5ca69ebc2da982be915f4d2bdbc7dc78b22517920302d2d2f5accb3da3954039',
        'D025039-2020_DEM_ELEV.zip': 'f599e68f9e9776508b6bb7debeddbfb8eaf4072800e80b7787f19f62466fdae2',
        'D025039-2020_SENTINEL1-ASC_TS.zip': '178b504581d555e50680fd8d8ddd33c640b83d55a710223c80b3dc87bbf92236',
        'D025039-2020_SENTINEL1-DESC_TS.zip': 'f0d4e7e0f512c1686d9f102757332fa7b7c15def9b4786bc3b63f7a095ebab7f',
        'D025039-2020_SENTINEL2_MSK-SC.zip': 'cb7fd6fa64e3132f4697c7a6f6994490e74804acdf64c1c3d0eb6c5924380814',
        'D025039-2020_SENTINEL2_TS.zip': 'eab602372e332dd49ecb103170f1d5b976c0def2d9cfdaf55c901e04a5931726',
        'D025039-2020_SPOT_RGBI.zip': '11e7947488117cefd844838d024fc256cf381d240621fbb571dfcb197cf4f3f1',
        'D026-195X_AERIAL-RLT_PAN.zip': 'a094f6d9d26e08137ba420f7cba6f92b989ab61fcd82577a6ebffc38fe36082c',
        'D026-2020_AERIAL_LABEL-COSIA.zip': '14dec3253870a62d3cdff4fe45ab5c9814f756643d2810bf12d31456df3012e1',
        'D026-2020_AERIAL_RGBI.zip': '52c60bcff21f3430f7adc42420ae7dc44d2ee52458713e2aa9a151808fad75fb',
        'D026-2020_ALL_LABEL-LPIS.zip': 'b741907e350a0077bf17d17e5fc7e1e5af736a028427642b8cada05cb4e24068',
        'D026-2020_DEM_ELEV.zip': '942de4407edcef83d1a6f616680387c1cd2e165ef2574bb6523777abbcbcb016',
        'D026-2020_SENTINEL1-ASC_TS.zip': '609abc69e960393a2551440d1a5252fb5868d9704de153d90b8a4bf901aa2bb9',
        'D026-2020_SENTINEL1-DESC_TS.zip': 'e221bf85b211a2fb92dc14d332ae18ea69177a4f2af1fcd0f34dc268a9ddaa6f',
        'D026-2020_SENTINEL2_MSK-SC.zip': '3c76e7e4988d8b1e3ebebfbbb819e2026b380ca0a0d3f1ae6f1208adaf845640',
        'D026-2020_SENTINEL2_TS.zip': '76d6bdfc18d54b8eca2665cc34c213ad7979e83941b31188773add9c47d24194',
        'D026-2020_SPOT_RGBI.zip': 'f90d877868b94625b4e2177d6385c3c731aa88cc6fae59cad78a929c10fc3873',
        'D029-195X_AERIAL-RLT_PAN.zip': '7eda768c2827915cdafdeee01b68f307a847291c771ade7c80bbd2f30afd7471',
        'D029-2021_AERIAL_LABEL-COSIA.zip': '67f3210be373c120bf40b1b9c49cb8b5c0e389993b9d2735d27025a076569db8',
        'D029-2021_AERIAL_RGBI.zip': '160d9ab7634f28f195d738c228a4086f4c3273117199afde0586784ab01fbaa3',
        'D029-2021_ALL_LABEL-LPIS.zip': '03199916d79172275bdbdce39101ef521d0346a69257e2d427415e9a5593c471',
        'D029-2021_DEM_ELEV.zip': '3242d993a31bb7ab95d993fa710cbb335aa8fe7bb37fb69649330bef26b137d0',
        'D029-2021_SENTINEL1-ASC_TS.zip': '314175cfd84cb8c15252c19daa5a409c511855f2d8562091dff61da03d1be886',
        'D029-2021_SENTINEL1-DESC_TS.zip': '34fb72a5945a38eaca2ca4590ac3a0b7fed2074477f3baa65433e96bdce975da',
        'D029-2021_SENTINEL2_MSK-SC.zip': '991ab7ba178cdeb23f1098526fdb27aaa85aa5f184f71e0fb1fcfc76d3a4608f',
        'D029-2021_SENTINEL2_TS.zip': '7bb20f9d7ab301b744231f5b73cf9acebfff3f0a3c4d95054f3dc6503b8c5f82',
        'D029-2021_SPOT_RGBI.zip': '129ff1598153c9ea53bb9b8536cdfd8025da65da1e757ef3730082f0e240024f',
        'D030-195X_AERIAL-RLT_PAN.zip': '42490367809c48d18382fb8b00104b12a61349be9d49b163f0592ef313e62bdc',
        'D030-2021_AERIAL_LABEL-COSIA.zip': 'dd55f45caf4c7d9d1e551124cded7d0ba75dd9650af198e1dfbf69fbd62ee483',
        'D030-2021_AERIAL_RGBI.zip': '6f7ada3ce0428133c0c09e06186f42144c0287640f0781c737937471714b55cb',
        'D030-2021_ALL_LABEL-LPIS.zip': 'd390b2a9955b4b193b829c2c82ed2a65050709ef891b226cc23d8daca7fbb203',
        'D030-2021_DEM_ELEV.zip': '53fd9c70bb68f0b55533efad9cff1c6e57e663528baa27df4a1933e7c10c99bf',
        'D030-2021_SENTINEL1-ASC_TS.zip': '4afeac13726390de0a362bb9ece123726902913def7ba716f071f5826f087aed',
        'D030-2021_SENTINEL1-DESC_TS.zip': 'a1fabf3d5b22ed55d7311fb307cfd6c0ef369b2926f837fdd43b68c7bd49cda7',
        'D030-2021_SENTINEL2_MSK-SC.zip': '20cdcd3f8425aaea8bf6ce7efb941f93c3217d386f8ef97b38472c4890f7b405',
        'D030-2021_SENTINEL2_TS.zip': '42baa20edf8af781786f9708d1f65d5623efae9b73bf266ad56bbb5121e8806d',
        'D030-2021_SPOT_RGBI.zip': '4ce707f1471fb762239487f2eb7801e5d87892aea7d7afae1d57c968347ab297',
        'D031-195X_AERIAL-RLT_PAN.zip': '1d38865af51acdfa651184f7ba0940914d4bacbc37337baa89ea3bb561a4cfa1',
        'D031-2019_AERIAL_LABEL-COSIA.zip': 'd15ec3e32f3c8de4b3f41c8539aad045446b894f12467388320794c10ae0f068',
        'D031-2019_AERIAL_RGBI.zip': '1e221048f46e63b06802840f49687a070aa3249856e5bd3fcdb876aa2b7a0835',
        'D031-2019_ALL_LABEL-LPIS.zip': '286de01ae7c5e745a16e3b0ada79b8b88730d556db2103ad633d0b601993a26b',
        'D031-2019_DEM_ELEV.zip': '54ef0e6359f804312924480a50167e75f4036a8788efaddea8b2df4b00a980b1',
        'D031-2019_SENTINEL1-ASC_TS.zip': '13990f6c18484d280de07afc7effc7735684ca641e24d9fb10cf5dd1bacf4e31',
        'D031-2019_SENTINEL1-DESC_TS.zip': '7cc2e26cf3756a5d1fe0e1cd5366e537c6d4ecbb47134479c87e98c693a5b314',
        'D031-2019_SENTINEL2_MSK-SC.zip': '2d36f63fe198d9863da2ec2d84531668f802369ad012a84418e4ac967f0b3b0e',
        'D031-2019_SENTINEL2_TS.zip': '402435d1533f24e384907ed654fc7c5cc57bc42e2c24bcf718d9ee438ab291d4',
        'D031-2019_SPOT_RGBI.zip': '449910920d4c25c857185ce0a69a2ec4014b8a0af84bbc2e9408693bce36c721',
        'D032-195X_AERIAL-RLT_PAN.zip': 'bfd99e695b502b9327f25470b4ed4a742bfecf4a3caef46fa3610d1a0a2992f8',
        'D032-2019_AERIAL_LABEL-COSIA.zip': '164fc942827801ed6451b5f19e0a405210353ca156017fe97181fe9a7af7db5d',
        'D032-2019_AERIAL_RGBI.zip': '35d04540230d7caf677effe888736bc09b2f5124849c9739c0d5ab0ef39994c9',
        'D032-2019_ALL_LABEL-LPIS.zip': '029916024772d5523ecd213bd3914987f35c4b7976a65b7ef9601a7f7c372d67',
        'D032-2019_DEM_ELEV.zip': 'd27da2ae2a99cf41620095c24294d7631f0ff652fa052e4ffa959cb48590bf3a',
        'D032-2019_SENTINEL1-ASC_TS.zip': 'd9cd7a9a21afb9b5991f304a8effacc361dcac732fed8c700bd8e66f75ea4dd1',
        'D032-2019_SENTINEL1-DESC_TS.zip': 'c3c7e44105231556bd24a1dd265559238e4b16e06d2052c67df7a11f47544054',
        'D032-2019_SENTINEL2_MSK-SC.zip': '79dd5c854c8ac549a2708a8cd802b3c08f5d31aa3e9c436d0016e5ad4bf47535',
        'D032-2019_SENTINEL2_TS.zip': '2703bd915e8220dc8e131455fe91767424ea1a92c13257dcbcc04e93756af075',
        'D032-2019_SPOT_RGBI.zip': 'bbea82d0d379ced1c6a7230222556721373e037b3634d90b3d4c982ce953d96f',
        'D033-195X_AERIAL-RLT_PAN.zip': 'fd44ab21de1dcde81a38cf6e107daca0760532d71f47c681a23406f9b57245be',
        'D033-2018_AERIAL_LABEL-COSIA.zip': '218a105fa729e2b9fbaca0c69819f9c77af22b12aeecad261f536830aaf88549',
        'D033-2018_AERIAL_RGBI.zip': '3b4486633458ed60419ab3c7deff321e0d6bbde7d82833c637016eaf7e66cc10',
        'D033-2018_ALL_LABEL-LPIS.zip': '27b316ce908eee952d581e04e7f375bcfc1b25786e36b0df42672be1b3074e0e',
        'D033-2018_DEM_ELEV.zip': '9a6f5d14f6c7a383cc49518195c34a7035dc0dfdd304fbe3936753a1e374faea',
        'D033-2018_SENTINEL1-ASC_TS.zip': 'd031a16feebd73184c92f926b01e473a0ede1873155ff38f58e5fdfe61599f3d',
        'D033-2018_SENTINEL1-DESC_TS.zip': '0d87ba68410a6e316ca1c688b9919d96090f8a51e908512c629d37ab35fce48b',
        'D033-2018_SENTINEL2_MSK-SC.zip': '7ee08c3482467ea93a212a0cc0b3143a14f895a3d79158e9e30e2fe98984c9af',
        'D033-2018_SENTINEL2_TS.zip': '15d3126b57583641cdc3ad525a35c01c99624c6f97a748e8735d13dd3a48182f',
        'D033-2018_SPOT_RGBI.zip': 'ec1e0141b7ec63531255315a91000d2cc7cd7b06f32105ce27ec1cd3bdb27464',
        'D033-2021_AERIAL_LABEL-COSIA.zip': 'f1e599eadc5f4581959c106494ab9733db231dfce92d8e473de80d2d3a28e45f',
        'D033-2021_AERIAL_RGBI.zip': '2ed7c040a74497b4c308a0629547240390c1ee4964f0df9d92797fe1bae4386f',
        'D033-2021_ALL_LABEL-LPIS.zip': '0a14c39387768e9b9345fded28a436ad36bfec5e28e8ed4092b91515528d66e6',
        'D033-2021_DEM_ELEV.zip': '511977efb12eec17309480120d9bebbefb7706f6b23b951568279e774285d359',
        'D033-2021_SENTINEL1-ASC_TS.zip': '375b35a89e5848fc0535b09e38c4a5d4c2c9ec92da2e05d4fb8bb8ee6b3703c4',
        'D033-2021_SENTINEL1-DESC_TS.zip': '77c6a9d3d1960c5193bcaf8d411b39f03dda9728778ba18a0416f57a8bfcdfce',
        'D033-2021_SENTINEL2_MSK-SC.zip': 'e7782db4f3ceadb34dfe3cb4ef202fc78be1b1a9bed7e314ad6d94615e2a4655',
        'D033-2021_SENTINEL2_TS.zip': '530c85c219f857d2e1cbe181a496481ae77fe363ea8506574891acaf279a77b8',
        'D033-2021_SPOT_RGBI.zip': 'dabb93a1e599ea34414b7b77a97c65d22ddb427cad51314e3aee1f0d0e934f1c',
        'D034-195X_AERIAL-RLT_PAN.zip': '914898ba6b8ad9d9f9458d74a6b0015b4f5089e5e3338fe0d59f25fc3197e998',
        'D034-2021_AERIAL_LABEL-COSIA.zip': '2f401a9b385b6b45c39f8c9c3ecb194de7c831a71d4ffe119971af1beb5cc295',
        'D034-2021_AERIAL_RGBI.zip': 'b95c812ceae0e2fb58b22f03ccdcdcf509df0019243049b6d4d3a0aa2d3233b2',
        'D034-2021_ALL_LABEL-LPIS.zip': 'b0aa68a049c26a47354b3c525242cdd9fbf44c82ec148a127f258e3bf7a82fab',
        'D034-2021_DEM_ELEV.zip': '485fa49f07444724623480f2a1ce29ee8de8d95c2b07cc783c9d5512922abf04',
        'D034-2021_SENTINEL1-ASC_TS.zip': '3f835a2e854ce6b6b7e4451431f3933219d690c032c20c6ede9d372731dc87ca',
        'D034-2021_SENTINEL1-DESC_TS.zip': '5a4f0b127209aca44b10488d73b10499e92478bc69abaa6cfba80b811accdb71',
        'D034-2021_SENTINEL2_MSK-SC.zip': '07228129951df11fd152bfa5435264f74ba522a2487a2f59d2483423c6eec083',
        'D034-2021_SENTINEL2_TS.zip': 'b0a49f152b39e02774c52e111d2596c6c0681b6f9f0e0d40c790f3db0d12912a',
        'D034-2021_SPOT_RGBI.zip': '58ee89d65192e042a12c99b1949de6d3444221a58db861570bb9dcb5ad8ef360',
        'D035-195X_AERIAL-RLT_PAN.zip': '74762f7c60a989dbf55699118e8514b2f4cbc706420fd18da36b1007104c33db',
        'D035-2020_AERIAL_LABEL-COSIA.zip': '1d0e56f2efac827166ceb66412abbf39792ef024c39bd8828384bdd523d8b946',
        'D035-2020_AERIAL_RGBI.zip': '482c432ce5669e68ffadee9ff0d5d5d998693fddb30cd2316a148d55c544b5ef',
        'D035-2020_ALL_LABEL-LPIS.zip': 'e22b871944c692c1bb68275cad8ff6bb72491414d477c6a30c24c3ce3f03c74d',
        'D035-2020_DEM_ELEV.zip': '6ead6e1c4f5ccf84ff5eba819c7306067bda14fc4302663113fae78328457762',
        'D035-2020_SENTINEL1-ASC_TS.zip': '7d35aa4866963da06f21d14708855070332b663e89e15549a85fcf3c2efe9446',
        'D035-2020_SENTINEL1-DESC_TS.zip': '926b239038efd2df94fd2d3abb9fc722a9f3a6a333b197ff348dd67a983fa216',
        'D035-2020_SENTINEL2_MSK-SC.zip': '3a9df8e6b8405dd80d44c2249ffe0d1390f51867127351dd4d153db8278d80a7',
        'D035-2020_SENTINEL2_TS.zip': '8a91e131b2387fd545327ed8c77baaa1c16404a88a457af4ce4a7053e7ff8fd0',
        'D035-2020_SPOT_RGBI.zip': '06d426f816dfc901b184a099db13be452ccd32adaa0a834c79bd992f0aff98cc',
        'D036-195X_AERIAL-RLT_PAN.zip': '77b77c84652832138190e2289004081a49d83c55c00e3219504c11910bd6a857',
        'D036-2020_AERIAL_LABEL-COSIA.zip': '439d4cb7d87cf0940b50978cd6686a6075d6db746f5d3f8eb98985a0b3cd60bf',
        'D036-2020_AERIAL_RGBI.zip': '68bcd05a826d16baa2a7df2a505deaea37e62ff09370c65b74441858c6d6f9ae',
        'D036-2020_ALL_LABEL-LPIS.zip': '0eebb6e22b37e15126b6718ff7542f94e771681539125eaad6a1c53c2863fb5e',
        'D036-2020_DEM_ELEV.zip': '66c49f2fffc2c564b469ad89ae63e667ce9034e37c35ed2028570ceb7c76c810',
        'D036-2020_SENTINEL1-ASC_TS.zip': '14799a417a11324417015a413810d0014d2696df904e7318bfffaf19a34d68a2',
        'D036-2020_SENTINEL1-DESC_TS.zip': '53752530a1fdc3d8b7c30c980c49c89306fe34da2f57a3987207af34a96f11f8',
        'D036-2020_SENTINEL2_MSK-SC.zip': 'eef5b30e0d63d1f2f3e01786017238a0af38be8e8ba7dd2c623795351b841b79',
        'D036-2020_SENTINEL2_TS.zip': '81c30d1423a7a318c5db65a44105ade84eeeeb855d58de524c1b0a1c9ffbe15e',
        'D036-2020_SPOT_RGBI.zip': '50647574fd76e47be6a396287a4f194771b3253b902c0eaa160415c69bcbfe3a',
        'D037-195X_AERIAL-RLT_PAN.zip': 'a94736e35a7c4f7e64917e230fb3f7bc47571649a38a416ae1f9ce0006450661',
        'D037-2021_AERIAL_LABEL-COSIA.zip': '7c5ea22ba40060d6a52105c13806c9f510a757e921a84a28cef4489ad93b7601',
        'D037-2021_AERIAL_RGBI.zip': '29fdef058407ebf12ad46e49b3fb251e3acd5dcc1b003c8f39675efd3f637bcf',
        'D037-2021_ALL_LABEL-LPIS.zip': 'ee60f215ce1f5ea3149b407192336b09362046c57b73c86b5c3edbd5ded7ddf9',
        'D037-2021_DEM_ELEV.zip': '8b293265ad1c48d83f31f98d02ff58b866dd0d36063cf5cb3fca4388c9cbfcb5',
        'D037-2021_SENTINEL1-ASC_TS.zip': '76721356e09d3b3690447528d678b2b02c72736fc51394bfe1f0187c9137a09f',
        'D037-2021_SENTINEL1-DESC_TS.zip': '96c30f9b9c587148db58a070edc162d1897e9ecfcae0902915861364e1b3d201',
        'D037-2021_SENTINEL2_MSK-SC.zip': '16cab6d65cc42e7b3462b6e79eaf709582d5e63a951bf3f0a27ea677ffe34fab',
        'D037-2021_SENTINEL2_TS.zip': 'e38cef855e1fcb57f58378d7c4967c4a5334aea5185dc89b76b54b330b746a93',
        'D037-2021_SPOT_RGBI.zip': 'a0db4a7b3ba7d02fede35cffecfdfe210acd5100f8ba2a89c52649151b0c87f1',
        'D038-195X_AERIAL-RLT_PAN.zip': '247ab2c8360117685a6d4f302a00672678493b305db51dbd27c638f26930831e',
        'D038-2021_AERIAL_LABEL-COSIA.zip': 'c17fd01d879a42ffd344c1525fc60bcf6fd8054c53cdf5bae60d97d9d5bda4f8',
        'D038-2021_AERIAL_RGBI.zip': '9b133a237cd1bbbc7b8b9cf556b0a63daa48bb8b536b91a1c92434687eb13cfb',
        'D038-2021_ALL_LABEL-LPIS.zip': 'e821ea446994bf9dbcb42ea8b3b414ed8d72e099a83fd6e3b87d2209cb9018cd',
        'D038-2021_DEM_ELEV.zip': '5c9afbe8c1b49b0fd8a3eec657a2fd7022f1ca4389a8b48e2bfa95dc899467ab',
        'D038-2021_SENTINEL1-ASC_TS.zip': '789b1bd98772e03e7997807f309acd7787a0aab3e0fe83c75fa609b7f1d8f341',
        'D038-2021_SENTINEL1-DESC_TS.zip': '2af48beaf1553621d3fcc11de71ecb1b9014d2e223e3b81124d48c16d3768439',
        'D038-2021_SENTINEL2_MSK-SC.zip': '9ee5d337c67cdb5cdc65b6589b9573fee2f03b4cb0a01f3fe00f8c95be1a6d28',
        'D038-2021_SENTINEL2_TS.zip': '38dff7cc18ca28a36be8d0eca0e3e58c5e966e1adcd85e527d7626b7cd93da70',
        'D038-2021_SPOT_RGBI.zip': 'b7b4eecacc51caae93e97ce0001455a0e5d018e59c70eb9545ef4cc47b5010e5',
        'D040-195X_AERIAL-RLT_PAN.zip': 'd4d5dda915b2d95975f2aa48831a5641fcc5ebecb90099bd24814d46851e3f6b',
        'D040-2021_AERIAL_LABEL-COSIA.zip': 'b51c4dc601c4341b036b4d5a2664169180d4c67b3a6a6f892ab5045452482c64',
        'D040-2021_AERIAL_RGBI.zip': '2e8bbbf1f73bea7f1ecccbc95dcdc7eb0cebfa35b646196da43acb981ae93883',
        'D040-2021_ALL_LABEL-LPIS.zip': '70e5b52bdaa80e4e1a44cb630c4034aa57f5de7a677fa17917562f40e93cad28',
        'D040-2021_DEM_ELEV.zip': 'd4b291f721175edae3ced934795aef47622e11f137072b36ce58382da60f73f4',
        'D040-2021_SENTINEL1-ASC_TS.zip': 'a51cde4a9c300d5fb4e2f68ddc58fb772f448bebfba30e02b27cd23dc4102227',
        'D040-2021_SENTINEL1-DESC_TS.zip': '8442f7c1e08a08bd8e7ee41d6bf6eb2df39cb8991e30734fc08d89c25ab6193f',
        'D040-2021_SENTINEL2_MSK-SC.zip': '16e9740b2896ecb940155fd1c89208ac60b4bbd2deb3fd1e357dc45510313305',
        'D040-2021_SENTINEL2_TS.zip': '5ad5eb76561e3e8e3c606495d51285a700ee8155733b323e70c2a54162538347',
        'D040-2021_SPOT_RGBI.zip': '2a00ab07ebc6d450ddbaf1ee8d9a0ef17c0beea8ddc4d5e8023de2000f351e2a',
        'D041-195X_AERIAL-RLT_PAN.zip': 'b4fbeb7240b1f818622e7465a5c40f45ddfeadd56b4f257f91cad51e5b61050d',
        'D041-2021_AERIAL_LABEL-COSIA.zip': 'cb06d0932851a0ed7eff624f721c2ba5403704bee01245d453ee4eb5beb5cc5c',
        'D041-2021_AERIAL_RGBI.zip': '3aaa8539c10e91e3752ed00ace84d6b6474d2385b57c7a09da8af236b6c93fe0',
        'D041-2021_ALL_LABEL-LPIS.zip': 'a707c187491aee9acf06473e8f2376e2f343771037dfab4ff847033484f879cb',
        'D041-2021_DEM_ELEV.zip': '650c6bc147aadc36b5ee0261f969afc7d6f3225adbf14526bc6fd4b7e406c067',
        'D041-2021_SENTINEL1-ASC_TS.zip': '98b0f4a814bd06316257465e8c6efac2b3264cdb300b80ef488609513e519552',
        'D041-2021_SENTINEL1-DESC_TS.zip': 'ad46a2350eee451fcc1d1ca1e1a7a19611169662fbd3fae778aff5616a980179',
        'D041-2021_SENTINEL2_MSK-SC.zip': 'ca0bed65c004d96a6bec1caf4b7c7b4c328e7436badc432e62653eeae633642f',
        'D041-2021_SENTINEL2_TS.zip': '8f6b5eb9ca162dd34265e5b112dfebfecc71c6023b2be4d281d68c1925326a91',
        'D041-2021_SPOT_RGBI.zip': 'd9849116ea7d27700daee394a416475399ee8c4c59232247815975605a4dab0b',
        'D044-195X_AERIAL-RLT_PAN.zip': '68784ab283ee8159b04a7389a3d757ac7a9ebca1f039a8b8d95a97f6b1e0dc24',
        'D044-2020_AERIAL_LABEL-COSIA.zip': 'fa61f61748e982963a136ead7414183762df061a23ac494e5bd1e38d730afc12',
        'D044-2020_AERIAL_RGBI.zip': 'bc66b868283fe56ba835a71d2a6b46a785dfd9ae365512353086e0745154b1bf',
        'D044-2020_ALL_LABEL-LPIS.zip': '2d372396b35b4ddcd57ed042afd20e213b000e6e050c34cc1bc713d1fccc9232',
        'D044-2020_DEM_ELEV.zip': '2d4587093ce32e10abfd91700c37f9452663572bc0e8655e0a85cd6ceed38423',
        'D044-2020_SENTINEL1-ASC_TS.zip': '40b3bad5bf8ba3ab5cc3ba23d0b7609510a4cee52e07f406632877033abed657',
        'D044-2020_SENTINEL1-DESC_TS.zip': 'aef51eb95f50b76287bc4e939fbdf34967174ae12233d6f72bc245abe083e77a',
        'D044-2020_SENTINEL2_MSK-SC.zip': '0b15363feb3cc047ddbdba7f557d64ee3002b84b3f8b85642b08b32e04842698',
        'D044-2020_SENTINEL2_TS.zip': '5c5f68df90e7c6e7a1000190450dee386afc95adca30cef25e97a6dcc735d56b',
        'D044-2020_SPOT_RGBI.zip': '521e6daaedba7570b80aa02c108b75b015c4a7c08d04b84c58d1edbfc6a64711',
        'D044-2022_AERIAL_LABEL-COSIA.zip': '9bac3806794bd0b417121e1b5e81c1e8cc4863a416fd06613ef18ecdafb876ad',
        'D044-2022_AERIAL_RGBI.zip': '541573ec9dfda81203226b52c90bd98a4883c5d24b2503d5c3bdb0ea136d845c',
        'D044-2022_ALL_LABEL-LPIS.zip': '5d484a27ac27d1627e030b6eb77c1169a64d155e75571c9acedbb96dab97c2f7',
        'D044-2022_DEM_ELEV.zip': '2e636aea0ff0ac316b964d44fc68d1fc7821a9630a0954a8e8c683ee6ec164be',
        'D044-2022_SENTINEL1-ASC_TS.zip': 'f851068342abb75262ec0fd14bf5e98c134c3eda2ee89907d18c7fc997a914f7',
        'D044-2022_SENTINEL1-DESC_TS.zip': 'b2d24c8f2a0eafe21917f8cd8773d8c6c994b2172ed0be3c71bb6164ecf09668',
        'D044-2022_SENTINEL2_MSK-SC.zip': 'febe347962dabf51b0ddf728e6f03e80404123bb05a4427f47dbcc31cf888fcb',
        'D044-2022_SENTINEL2_TS.zip': 'c98ec9c9fe4cb6c9ac66505449fa653fa529facf514440f0883fdb0ebda0d82c',
        'D044-2022_SPOT_RGBI.zip': '72c5c004c9d357fb78a51584d1dcdf072e88dd1c86530c28bfcd02223f9e4622',
        'D045-195X_AERIAL-RLT_PAN.zip': '5530d641dab1a40d2e057cea4e388622348e7cb480c5c8bd39771b63bc8bbf1a',
        'D045-2020_AERIAL_LABEL-COSIA.zip': '989de83774b3fad4506f1be088b61219a2920ba413d0897f1d05eba2d6d06a52',
        'D045-2020_AERIAL_RGBI.zip': '7d2222c6c2d4a971f6181aae8607f7bf845825c4c98ed4145a0ee6f4ceb13800',
        'D045-2020_ALL_LABEL-LPIS.zip': 'e43b05baacf0d3a4058e145cf71fb28e7f87656881f77080581cba3793b4afbf',
        'D045-2020_DEM_ELEV.zip': 'b4ff8ca5513fbb8b102e948791cabb9cc295b1e83b02c4b82dd23e2689a0ddbe',
        'D045-2020_SENTINEL1-ASC_TS.zip': 'eaf610eb93d1da27987fc84936b0cf23d82f85c586f57b03ca00787d49529452',
        'D045-2020_SENTINEL1-DESC_TS.zip': 'd5344678274d5d2f092f2dfc7733be5addaa9d172d194cf94508a25de3e4debe',
        'D045-2020_SENTINEL2_MSK-SC.zip': 'a57b256c4564929b25d4ee2d1497b878232d25e59c5c19dbdcab029668fd1073',
        'D045-2020_SENTINEL2_TS.zip': 'e59ad15cfedf8147a69bcade376f3f45f2ec1e9fef895461bc0a45e00bb69e0c',
        'D045-2020_SPOT_RGBI.zip': 'f0a130c9bd6c0f7eb5c83a79f336caac3e13ded3c1e9ff1c1c3c79454e55158a',
        'D046-195X_AERIAL-RLT_PAN.zip': '5ae8e01fbbae39f8438f9f3eeb8e7c94a5de5829c7508566fbfa5de4bf6e6c65',
        'D046-2019_AERIAL_LABEL-COSIA.zip': 'c65f2a3be43d8dc6e5c65a85659236a78051909177d411be5249d9879474ad2a',
        'D046-2019_AERIAL_RGBI.zip': 'a556cc514df072117e325efa0868e5a2d931b8698ec58867bde114553a6a097f',
        'D046-2019_ALL_LABEL-LPIS.zip': 'bbfa0931446528873c58e8c7acc675181c09eda81df106211d1bb32502edd14e',
        'D046-2019_DEM_ELEV.zip': 'b6fcabc97f6c374ab89b978e8efac12b759a71b288ec8608f1dcf80425b62f47',
        'D046-2019_SENTINEL1-ASC_TS.zip': 'f1832246d94e757c5ea2ef1c8ce32e93da9faf0718c81fe5113d93bf56207411',
        'D046-2019_SENTINEL1-DESC_TS.zip': '96f05c4e7c9f115def80f2b3aed0d583320e9a5b38159345cd8ed5b6ce02c66a',
        'D046-2019_SENTINEL2_MSK-SC.zip': '2db77412f9dced7ab95b2b7f2601f81a29c2fee23785463a96656cf928101014',
        'D046-2019_SENTINEL2_TS.zip': 'f905c55bb0ea7618cc606185bff8da482c1619367403d111c3147bf8876888a1',
        'D046-2019_SPOT_RGBI.zip': '5abbd854b30e70f8821cfa6b1aeabd311939d99cc4d4c50112f8f295fd172aa0',
        'D049-195X_AERIAL-RLT_PAN.zip': '575085309efec9b97be49b7cbe4a14c55fd300edc8db7106fd624e1b1d330de7',
        'D049-2020_AERIAL_LABEL-COSIA.zip': '18bc4f34f33a8bfad2b85b4ae2167cc5ee7e3dc153fbfd4b49437725e2c81567',
        'D049-2020_AERIAL_RGBI.zip': '4a4a5617d91c25c3553a5553f40ba2797aff4adf694fa2e14ae10d3d59178190',
        'D049-2020_ALL_LABEL-LPIS.zip': 'b9709fc214d2ee11d7d89fb729873369a6ca3226533e3e63fe3cef39934fb122',
        'D049-2020_DEM_ELEV.zip': '359b5a343c99f667031ca90f287fe3fa3354635c8e960b8b3625002342ccdfbf',
        'D049-2020_SENTINEL1-ASC_TS.zip': 'c957647452f2ed8fb32e52409b2399cbb39621ae81ffe5b887b86ed07bdd5f4c',
        'D049-2020_SENTINEL1-DESC_TS.zip': 'ad2fd7369c46a3e6b1fee4ad8ace6e3ba6a975976dc5e3698aa8473a700dd19e',
        'D049-2020_SENTINEL2_MSK-SC.zip': 'ca71619daaf33e7358bdcb21facf909216dcaa2ee49058eabe7d94e9d9c67581',
        'D049-2020_SENTINEL2_TS.zip': '1aa5172226a39466497f7126f82fbc387b721f3ccc328f36fe75c454146700a5',
        'D049-2020_SPOT_RGBI.zip': '6d4fd158876df975af0f6d4add025c857a8447644ab06dceb4cc6979ec8b3728',
        'D051-195X_AERIAL-RLT_PAN.zip': '83b1aae5a8527cb08a2ffbd70db0b7437df1c9a19e7940ae60878f24b5f46ed5',
        'D051-2019_AERIAL_LABEL-COSIA.zip': '4bb24cbc4272997ff81675087c25328fec0f05e1afcbdc072b182c5f284b33cb',
        'D051-2019_AERIAL_RGBI.zip': '9d632a3768197ccc6f6c6c01203984021467d4bcab8d93582d924c3edd9d2ae3',
        'D051-2019_ALL_LABEL-LPIS.zip': '88959278dfc657da5bd0c44efc612c55b3278ed22ced03f1a5431f63ec8266c0',
        'D051-2019_DEM_ELEV.zip': '7772af276991fbd2841d42a78f288374213d4e432f86cb70d236682ac4bcb071',
        'D051-2019_SENTINEL1-ASC_TS.zip': 'c929da80227e97363e4d2ae14c664cdd282e6496878b7814a49a72c3e3a6b7a3',
        'D051-2019_SENTINEL1-DESC_TS.zip': '451edf3d0e096fb17fbcf155a0f03eeefd377eae3b44649f16a237f34d6595b6',
        'D051-2019_SENTINEL2_MSK-SC.zip': '559f7fbd4580e122623b1273710845826dea3f6786d9771cb67321c717af8587',
        'D051-2019_SENTINEL2_TS.zip': '98b1cea06cb881c7af6d8207b6662fa70ef061192bdaefaded79a57085bf1dfd',
        'D051-2019_SPOT_RGBI.zip': 'bb49b7d9df4cb2716808e89a822e72e2ed046b30c6e6eaf3eb746aa99edc2376',
        'D052-195X_AERIAL-RLT_PAN.zip': '3c81a8b236e9acb6a39ef2c9037cfbc2d00069a697676a99a4cd00caa37918c7',
        'D052-2019_AERIAL_LABEL-COSIA.zip': '4018e67afa6b5d64289a07c0dcba975d82da5932889db45b63b5e0b239f2db70',
        'D052-2019_AERIAL_RGBI.zip': '3557c07b81b78b1871dc1eaf07e5c43c0fe06b62ee6007042d46634073d14996',
        'D052-2019_ALL_LABEL-LPIS.zip': 'badcf931be0a7c29c09abbe59e61a391c044f760f793c10d39d954238ef05891',
        'D052-2019_DEM_ELEV.zip': 'a404454bedbd6437df2ec851e82868f52d1407a49c16ca304205073381fc2672',
        'D052-2019_SENTINEL1-ASC_TS.zip': '758048bcae222dd8dcdfd567e8dd7ddea59db87abe770fdaf91fea158de57342',
        'D052-2019_SENTINEL1-DESC_TS.zip': 'aa76e12c15097c7f3bab848b541fcd5e5ea635f821f6526e4a24945a1ef99c1c',
        'D052-2019_SENTINEL2_MSK-SC.zip': '544a86e85d25493839bb19e6fe5a5ed723fd3a11bce868694ec658005e0a7a12',
        'D052-2019_SENTINEL2_TS.zip': 'da5a1651eeab387303cd0edd54dec104c6fc37c58f43b4f925ca5e3372a2a5f3',
        'D052-2019_SPOT_RGBI.zip': 'b16907fdd14578e9c06ac675bc7c1664606821bcf70758c313d4c4505c64c9b3',
        'D054057-195X_AERIAL-RLT_PAN.zip': 'cf341b37accb517a51162ae6bc4a851bab1355e22f4f91f94d8d576146510833',
        'D054057-2018_AERIAL_LABEL-COSIA.zip': 'ff79e3c1f7a09bf10ec4ef4650712228fb69e0c58b5ea9da4042be69f1d3d616',
        'D054057-2018_AERIAL_RGBI.zip': 'd50b7bbe9e79038f13c8755d40366535cd0b8b253d9bd3c9add111f2f83d3a82',
        'D054057-2018_ALL_LABEL-LPIS.zip': '21af40d498b467779ecdb9038b8be94c43d219f04c9fdd5595d018def9820eb2',
        'D054057-2018_DEM_ELEV.zip': 'f2f4eee9d052f5eb1558169e7ade2a07075758d1f3bb8d782c0660c1ebe37439',
        'D054057-2018_SENTINEL1-ASC_TS.zip': '0991a0a77e025e6357ed51f4ebc9d56300839325cc669e9927b1650125ba9bd5',
        'D054057-2018_SENTINEL1-DESC_TS.zip': '96395fc7866b7f833baecf6373ae2874929c10dbe8f64393ae872f41c3fd2dae',
        'D054057-2018_SENTINEL2_MSK-SC.zip': '143f2cd8b58d83ffc9646558d116d86ff383dca7986edb3e5a08797a1b96a921',
        'D054057-2018_SENTINEL2_TS.zip': '9e5e7ec5d033147df024d5cf70305d539fc1cbf143f255ba581c2fd3368a5e0a',
        'D054057-2018_SPOT_RGBI.zip': '9ba2bc6c5f7aa7d68c6b11e7637cecc41ae75cc97fbb289604da15cda9abbe10',
        'D055-195X_AERIAL-RLT_PAN.zip': '80025defea29b4e0e89f9c6fb62bdbea033ecc60b653987fa5b97dc18f72a7aa',
        'D055-2018_AERIAL_LABEL-COSIA.zip': '49606605f620f139641d65d1f6f006b4503fe4f853a25426c1da7432a9fd5d1b',
        'D055-2018_AERIAL_RGBI.zip': '703ffb2d968971a19394500f3a8f87f851e7fa23b2793a440fc56552a48dbbb5',
        'D055-2018_ALL_LABEL-LPIS.zip': '0e4935f1592b8b6afde784acfb30f7bf65acc02009ae8ceb0286608df893d37d',
        'D055-2018_DEM_ELEV.zip': 'cb6818ca337d64a72825045ad7c47267faa5f167196e04235f2fac75cf2074fc',
        'D055-2018_SENTINEL1-ASC_TS.zip': 'd5c6bfadf582a19d4e6bd3ca0fdb8f5e06f0a38e64eb8bbff1fc53cc6dd04fe4',
        'D055-2018_SENTINEL1-DESC_TS.zip': '362d00142a8eaadb08f7cd053bad1e4c5eb7266c701bf2c188ffff2039f5c9dc',
        'D055-2018_SENTINEL2_MSK-SC.zip': 'fc46fd26705ba6dfe78381e8c167e2f8a8fbd8409509e9f8fe1d2eae0e990ccb',
        'D055-2018_SENTINEL2_TS.zip': '1b9d005b16864f0096e1cfa7036616fd983015e331ea84055d667f7fa7b19806',
        'D055-2018_SPOT_RGBI.zip': '0b143bbc4ef1c8207e855343037a5407c463867b341efd436f9000e6d40f8f7c',
        'D056-195X_AERIAL-RLT_PAN.zip': 'd0c3fe0a8386e35a23286c7b794f381b9856f56a21df90df9e4e7278a00a5a1f',
        'D056-2019_AERIAL_LABEL-COSIA.zip': '05edade849635a31ad6a898b070aef44be9dbaec312c2e8cf8ecf1a719410204',
        'D056-2019_AERIAL_RGBI.zip': '24f068768a7f40d20675a7b60c7e91cc3195349fac18b62b8b77ea5bb26c9464',
        'D056-2019_ALL_LABEL-LPIS.zip': '9cdc6513c3258259f4f8ec6f6ab9cba2333018d6dfb5cbc2e91cba4bed9ebb1d',
        'D056-2019_DEM_ELEV.zip': '4beb6ba3e7a0bd7023c62e500dbb8a910e044c80309941c8d4707c8e5d95ac04',
        'D056-2019_SENTINEL1-ASC_TS.zip': 'b223b9501a27e173ffb122a7e7fb6c840ccbab1cbcef5e472e7202ba8ad7e5ca',
        'D056-2019_SENTINEL1-DESC_TS.zip': 'd6fe0dfce7155441669d0c19e35b03f9334134b98158562d572b7b21e3fd2922',
        'D056-2019_SENTINEL2_MSK-SC.zip': '625b3c5f9cda45f610483fb189cbfd3dad5a827577b0e952f1b35b9c5ad596f0',
        'D056-2019_SENTINEL2_TS.zip': '62fd78bd5edcd458e21e67d95e1ad5f9e526fa20a00b3600b91a25e93c6e6e7a',
        'D056-2019_SPOT_RGBI.zip': '8ddb0b3a428bcec9dd1eb12316c1463dcd1ec41bea9002f462e1847f643f905a',
        'D058-195X_AERIAL-RLT_PAN.zip': 'c44a6fc3b06e69cd2d4e72f39e24e1a519f5683e3d2d252358612f27430c4fb6',
        'D058-2020_AERIAL_LABEL-COSIA.zip': '4a2ca016c501a28b05f4a3c6f1fbb8b76eb60be93234729a9e92fa1f32be5e1b',
        'D058-2020_AERIAL_RGBI.zip': 'a928ab86805057a8612840e20c4551c6dde146ebe3704ad61b2958cf244564d3',
        'D058-2020_ALL_LABEL-LPIS.zip': 'a754450e6f0250a2b983996bab64ce7dcb39299be096f4659f0634d774152195',
        'D058-2020_DEM_ELEV.zip': '97c398eb9f0071d73dc20aa3bb8b84a44235bd1d6f29d00977cf4bac450dad33',
        'D058-2020_SENTINEL1-ASC_TS.zip': 'bdb98e4c2a423e318451aba954fce7d3197f47a1c6b76d137c57fec9d46a7a6f',
        'D058-2020_SENTINEL1-DESC_TS.zip': 'ca185aefa2d27fbfec19789b00a0129741464ecca0d5ccc09b2c55e19d5e9ff3',
        'D058-2020_SENTINEL2_MSK-SC.zip': 'fb7b92e68bb8f54428372a768a95a28eea9aee5f6e68a3718a15788283459335',
        'D058-2020_SENTINEL2_TS.zip': 'db5460d12b727af43006d04539796f83bd041d8646cde87cf242f3c99adb5722',
        'D058-2020_SPOT_RGBI.zip': 'a7b43dd995ccc7f924c7ab3fdc4f0da7aa3bc90c9f8c49acb012ab690bfea277',
        'D059062-195X_AERIAL-RLT_PAN.zip': 'dd4e9e64ee9669e771fc9974f4b524fb1779eebfcdef08fc82001e479d39e39f',
        'D059062-2021_AERIAL_LABEL-COSIA.zip': '86706535add0ed6fec0895fca17fe4136a9ec12e21084d01c65f1c41cae51578',
        'D059062-2021_AERIAL_RGBI.zip': 'e0d92bd486c8e41001af68b5a6b33bed04a18f02ff3567a78952b7941fb1e81f',
        'D059062-2021_ALL_LABEL-LPIS.zip': '84e72cc7d5ce5f19cd9aed1534aab65e353ceee9e005c4c1c393db4a86bb282f',
        'D059062-2021_DEM_ELEV.zip': '7e5316a2bc23c3a3ae7fabb538f421f2ccae0071c01e8d5342df85508b959a95',
        'D059062-2021_SENTINEL1-ASC_TS.zip': '84c83fa768284b8176313dc0d8c53fc72023514d0cd17fa9068f6ac9192ea205',
        'D059062-2021_SENTINEL1-DESC_TS.zip': '29d12f5caf247aa2043e16569ddb22ed0ee0cb605fbac1dacd788eef8f4c160c',
        'D059062-2021_SENTINEL2_MSK-SC.zip': '50c5b57c2cb59c2e78147aa9b9c7778ab37e251575f5b93aecdc623fd7049f23',
        'D059062-2021_SENTINEL2_TS.zip': '5fff19bef67cf5d1b853a9a4b1f4078971801323f71937d74551b543c58c68af',
        'D059062-2021_SPOT_RGBI.zip': '4d8c97ca73e5e91a821afe4397fe9fc94a127e30f23a03a0f0ec200c0ae5b1d9',
        'D060-195X_AERIAL-RLT_PAN.zip': 'b3b6bd9b9712ad6d941dee77fa52da1348b7554291026b6dd42a118f3fa8c6c4',
        'D060-2021_AERIAL_LABEL-COSIA.zip': '821083fae655fb42f23a5692c9efcac9123766775408e6cb23a0c4790b802267',
        'D060-2021_AERIAL_RGBI.zip': 'b2390eefe528e32408b5aa29fbe5a22feb9a64ba08b75eef5bbbb9b8a01fd0a2',
        'D060-2021_ALL_LABEL-LPIS.zip': '9ad412a331a84d20d7df03eb1a7fa17ec01a7e6a650eb4c23ecd1635de5a4849',
        'D060-2021_DEM_ELEV.zip': '9fb68e425fabf0a8c9c790b697ff5f14744d5e4c84045796e65919b96a4873ac',
        'D060-2021_SENTINEL1-ASC_TS.zip': '75c8b4a93ec076a81429d84f83c1f3e702f33fef18c4c345aa96a718fcce431d',
        'D060-2021_SENTINEL1-DESC_TS.zip': '384d84116eaf01f9376acd51283ee1e5c3ed9e7a60294caf3c37a6c2ea096b12',
        'D060-2021_SENTINEL2_MSK-SC.zip': 'c449e94028c3067d63969a60a6a2aa3f06bfbe84c3ee86b2bee3a56571b02691',
        'D060-2021_SENTINEL2_TS.zip': '509de9290a83685b5669ca8d30b6aef3794544516107b9ca41557dbad81fb1fc',
        'D060-2021_SPOT_RGBI.zip': '1458ccc62854e55baaec5dc818e74e944884db248a3c984581966ee97b2e8a34',
        'D061-195X_AERIAL-RLT_PAN.zip': 'c464aa403b0aff2179f37125ab92e7cfa44b6250ecf040c71a0e5a7d292bed77',
        'D061-2020_AERIAL_LABEL-COSIA.zip': 'b2dae522296f765f68f4907e150909802f464efdf8fa69843777bb7e8669eb2d',
        'D061-2020_AERIAL_RGBI.zip': '41e530ed0063c314be9c2338d03a501af4acdee67c33b9c969c7c2a30cd05579',
        'D061-2020_ALL_LABEL-LPIS.zip': 'c0bab7940e25fb10a1736f5df94c20e4f683c2f454ea9a4a3136240205ea5c61',
        'D061-2020_DEM_ELEV.zip': '35c20459140d31ce835907f8c3e5654aeb10e9200c4b689844ff31b648c0fdd9',
        'D061-2020_SENTINEL1-ASC_TS.zip': '3f6b3b64508deaac2f55b0dc52bb4c482a7ee4f5e9285f1ad3412fa7f6dea203',
        'D061-2020_SENTINEL1-DESC_TS.zip': 'b4a5159957efd5a589f93b3e3c107a404f37b991d37e8d0173b430078771e242',
        'D061-2020_SENTINEL2_MSK-SC.zip': 'eda800690b8342004fb212dcc0fb12e4ffe3437e2736c28178a8f13bf23afafc',
        'D061-2020_SENTINEL2_TS.zip': 'e8ab1dcd59731690af24c2ea51a017abc4ed347f54e587f261e675957c7fac3c',
        'D061-2020_SPOT_RGBI.zip': '808afb397b17b24e06debccb4a74b130465926b9197ad29611c425e8754736ab',
        'D063-195X_AERIAL-RLT_PAN.zip': '2eb7461b955e716a6587e1f55e807f17556942a9cfe74be1a03241415d875f94',
        'D063-2019_AERIAL_LABEL-COSIA.zip': '286dfe50f26d8db7b22c2079dbb0b58e0e564af989b7a1bb08ab1c90760f449a',
        'D063-2019_AERIAL_RGBI.zip': 'c93bb8f506af37e3909fedda5be03739fcc7ddcc3e1a1e6c0f99eddb9db264d3',
        'D063-2019_ALL_LABEL-LPIS.zip': '6dd8168c2b7481c4cb0c6fe1d2b6fccbe8e45714c623cea532410d19e8b1720a',
        'D063-2019_DEM_ELEV.zip': '5ac1b4fe167b47db8921754bc9a9fb46fc44ccac77695cdd7197832d8e728136',
        'D063-2019_SENTINEL1-ASC_TS.zip': '1e4dcc61feebb88fc2175b11af362321f41825ac67053c328be5b7b5ebc009ab',
        'D063-2019_SENTINEL1-DESC_TS.zip': '849abd02c8f49e00e07bf500a5d9493d0410ca0c99c34aaecdc1e30d40d02a67',
        'D063-2019_SENTINEL2_MSK-SC.zip': 'ff24b4fa72086f0f2d0516f13b440cbdccc8bab417adddf6828815edb005cb3c',
        'D063-2019_SENTINEL2_TS.zip': '1e6bf8a6a23b5268a240e86d43b0e31c6b8aeae639d5cf58802f6e5adb8142a0',
        'D063-2019_SPOT_RGBI.zip': 'e83d2a42f925807865733832a6fe9789522a3e033f38a6d6b6bdb1f44c6d9c78',
        'D064-195X_AERIAL-RLT_PAN.zip': '762c4a8c827dc1a75dc74fe18f17f7edf5b91a2d784422826303e69746a5c55d',
        'D064-2021_AERIAL_LABEL-COSIA.zip': 'e8ad896053e52921543f748b7d278d894726d4d627f043aaa9981941357733f6',
        'D064-2021_AERIAL_RGBI.zip': '3f698532ecbea8355dc5fa6cb761c9ccaef1479adea110c384b5eda95e2c0a13',
        'D064-2021_ALL_LABEL-LPIS.zip': '0b7ab9c3e1789a2b48d694d119d4594e4d54929166416c00c257e8e34f09d9ee',
        'D064-2021_DEM_ELEV.zip': '727375bc83fd3c8f6b7f96c24ae3698179e1fe5af7218ac7dbbcdf76db1d3d29',
        'D064-2021_SENTINEL1-ASC_TS.zip': 'f6714abf189ab1a609c5a562d64c3e943091cd1a7ef3b5f4dd8a77b045b21a30',
        'D064-2021_SENTINEL1-DESC_TS.zip': 'ad2d79d71210c6dd05341bc7120b16dff8e7ae1ba67f0002a90caac6b8f55546',
        'D064-2021_SENTINEL2_MSK-SC.zip': 'c29da5096b07db092d331416e90116c40027a8a18d962af806ea2fd3bd89b2ad',
        'D064-2021_SENTINEL2_TS.zip': 'c44e90c7c247e6bd98cf909edd443c7f2d13114566e20e6a93f8a277ad17fdc2',
        'D064-2021_SPOT_RGBI.zip': '9065219b570bb627b60c4b7e5a681c5c218da4b7765c93cbe919cbc94abda699',
        'D065-195X_AERIAL-RLT_PAN.zip': '913a2e0124f7f2ac5d5fcc70318e652163d7b3ce6686aba537360259784fa6e4',
        'D065-2019_AERIAL_LABEL-COSIA.zip': 'da0e9e872b9cf8b0a73617caac8ca80241a796b75a5486c193f2dccb6ffe2760',
        'D065-2019_AERIAL_RGBI.zip': '1fe2e5d3f73b03e69f35b58cbae488cb3c6ac3a6b2447746883c260c56f1b7ba',
        'D065-2019_ALL_LABEL-LPIS.zip': '6ac995af23fcada88371ec1e2efe4ca471ad4de0b9d736685879229649640fa0',
        'D065-2019_DEM_ELEV.zip': '42bd65eef147103837dcd93e87d2be22d77aab8b91f483721daed017545a547e',
        'D065-2019_SENTINEL1-ASC_TS.zip': '5b27e2a044888637f4a668f2d24174cc43c3447c15182cedea4b5a3ba2756778',
        'D065-2019_SENTINEL1-DESC_TS.zip': 'c209e84fbad8993a1782452a3d4401bf2c9a54eb73fcb308dc28266ca7c3284e',
        'D065-2019_SENTINEL2_MSK-SC.zip': '084a83dc1d9bd28ca8641cffeca180750d4c3eb5a4973c2ce8cbd1f1e9981ced',
        'D065-2019_SENTINEL2_TS.zip': 'afa998ec3647385d15e0b568582b09a10b4187cf6d60c830cd9b774def8abc08',
        'D065-2019_SPOT_RGBI.zip': '26d85de3ec4757cee5419ef6a9f42d860e959ce90c75f84ffd4240992614a78a',
        'D066-195X_AERIAL-RLT_PAN.zip': '35694a8db9195654152c6bad6157a43d1c7af4379f324b6581317cef34287d69',
        'D066-2021_AERIAL_LABEL-COSIA.zip': '23573349c7386292895d1037305a3125624111f25d0fbbba18948abbae056585',
        'D066-2021_AERIAL_RGBI.zip': '239e67c669dd2142cae9827c06f76d6283da9d662f4ac2a50a7acc7028b4589e',
        'D066-2021_ALL_LABEL-LPIS.zip': '333938b1179e5d124915958490952b70db4aea34a18e9c796c2dff95170dbce4',
        'D066-2021_DEM_ELEV.zip': 'cb7d7aae53f93b8e87d20e909368495d78570e6a63ba3e23d14a08a779ab3f63',
        'D066-2021_SENTINEL1-ASC_TS.zip': '4c5f7c824b64f4bca80efce87bbe00428db7ba34b67544413f4a3a4debf7b7dc',
        'D066-2021_SENTINEL1-DESC_TS.zip': 'ca5193777b7a4dc9588d9f0ce0c675dcdc4cd42b54ac022b8bec5f2fae0c9eae',
        'D066-2021_SENTINEL2_MSK-SC.zip': '7624aaaf183977982fadd76f2378a6fc35d2fb81acab1d5bbd8b3c848fa58ec1',
        'D066-2021_SENTINEL2_TS.zip': 'c6519c689871f4e5de131ad7773589437bc6795d8c0fd2dc2d587d747cd8add3',
        'D066-2021_SPOT_RGBI.zip': 'd817eb264de5eecd0bce8a5ba76c9a3063cbab2078b4936c6b20fcbd601c1941',
        'D067-195X_AERIAL-RLT_PAN.zip': '1175b055084a6db33aaad73ad18c597c535a5f61d9f2492b3a8764a1de4b0665',
        'D067-2021_AERIAL_LABEL-COSIA.zip': 'c26d9204cd93b8485a77e767de9c6690e55860b80256ca2173db8e2b25c78620',
        'D067-2021_AERIAL_RGBI.zip': '8d8884fa0ad64d1434836d6fd3e1904ffa51ed377d402760e19e3cfee3cb6de3',
        'D067-2021_ALL_LABEL-LPIS.zip': 'dd4e44555460a4b1d21e8e73031e611a98a59756a7113d86748a7f24fec01a6a',
        'D067-2021_DEM_ELEV.zip': '31d25903eaa43778fa68597a27810c6edf76eacf46000b37b61590d7f2a976f0',
        'D067-2021_SENTINEL1-ASC_TS.zip': '9ddf7ed0f4b20753161c627b35cfa5ec45c6dd847e7579fcbe85dc357f9715bf',
        'D067-2021_SENTINEL1-DESC_TS.zip': '05906ee759ca54e30abd91ed3a4bd4655c6d726348c25d6426992b3db8ac2339',
        'D067-2021_SENTINEL2_MSK-SC.zip': '76950c4fb2ed38b27774b0ece122e971ae689c61ef36ad45ab051661a09a52f8',
        'D067-2021_SENTINEL2_TS.zip': 'fb789bef5fcc2fb8e0914305fcf1888e464b5b5208c3b792cab891da3b7a2198',
        'D067-2021_SPOT_RGBI.zip': '64590ac59ccbf19aab2938ff92e25a0f2b43677e77171527fa2be2c8f833b727',
        'D068-195X_AERIAL-RLT_PAN.zip': '60abb6ee4c265f507e91f93f25cbdd22c180fbe7e94ce882b50db8aa2b1d9da5',
        'D068-2021_AERIAL_LABEL-COSIA.zip': '71d515636b2f63797e2428b08a4f98b1434fb2e67d720cd9bf6d761df8456b7e',
        'D068-2021_AERIAL_RGBI.zip': 'eea78a7be6b4a63f8368c5c7ca029bfc454ee89041acb62e026824499cd89c97',
        'D068-2021_ALL_LABEL-LPIS.zip': '04dfda15cd07d3447d1c90922e92bcc62c125ba547fa3622f3902cfaa6549273',
        'D068-2021_DEM_ELEV.zip': '84ec2c775f8fa3cfc2cc4e14467defb01561f26eeda0f24b5aee535b7670edbe',
        'D068-2021_SENTINEL1-ASC_TS.zip': '88617173239899cb9634937775a3a1ff42fed05941a08faee443955fcb283b05',
        'D068-2021_SENTINEL1-DESC_TS.zip': 'd0c56d2d698cc8c69ba32f7bbb0120120a8e31e9c879646079f7b8d613464785',
        'D068-2021_SENTINEL2_MSK-SC.zip': '630df7d3236b5b39b21b9cef21f515b57b691bbbc8cb10fcd7b3c38ac8ac02c9',
        'D068-2021_SENTINEL2_TS.zip': '6c44a1fdbf0ff97f69808dae26e1c5035d99187971c6f1ba1f36c97e18e5444b',
        'D068-2021_SPOT_RGBI.zip': 'e9a4031e87d7268c2dae09e4ed62287b91967c0f6b109229e850c872e8933ca5',
        'D069-195X_AERIAL-RLT_PAN.zip': '1895968e453c9248ded7cf76627e41ba186b345e3753c597304aba38415f6943',
        'D069-2020_AERIAL_LABEL-COSIA.zip': '8a4b76509fd30d525cda0a3bb678ae11f308bcb9086402b1a035d6fedc0e70ed',
        'D069-2020_AERIAL_RGBI.zip': '7a1c214a39f3a00c68a5c3e0ae803a298998d8257c2db7c3249658d3791b336a',
        'D069-2020_ALL_LABEL-LPIS.zip': '202ca40ad74d44485dddca4d62b530ccd3edfdbc3aa923f3a2abe05f31a331fe',
        'D069-2020_DEM_ELEV.zip': '7e9019584f66d843904f897bedfa5a25a7a2310ccf5ecfae97812b8aa189389b',
        'D069-2020_SENTINEL1-ASC_TS.zip': '91382b48425101e5463f00252904469992e25fccdfdcde987e1b256fcf6cd45e',
        'D069-2020_SENTINEL1-DESC_TS.zip': '61d423db9a64a5b5d25968b28d86f45b2da11057e7a60494e52b33465271b87a',
        'D069-2020_SENTINEL2_MSK-SC.zip': 'fc725d9ee06919af9f8a69f26877fa52f363b578fd4df30478819c63d4c79b87',
        'D069-2020_SENTINEL2_TS.zip': 'f1bf007642ba81b287fc68a27657d1c451403ceae699316b1b240849160ba0bc',
        'D069-2020_SPOT_RGBI.zip': '22c2529af96dbe08b8761f7da5e02e7cf89dd50eb26becb9dfb5f6d4b82f2033',
        'D070-195X_AERIAL-RLT_PAN.zip': '67e8f6f7c46307030898737252716efebf7fe2388c710ddca21727cdd431f229',
        'D070-2020_AERIAL_LABEL-COSIA.zip': '4cd6a0610b2ca0fcbf148027c841824a4e6db17df5593a4bafa56acbef962851',
        'D070-2020_AERIAL_RGBI.zip': '81a183af66a5f739ae658bf6ef3beb85db8ac85e44b98ea7c52a8a2a8137fe05',
        'D070-2020_ALL_LABEL-LPIS.zip': '3d0d057bb86e1b291707da9bd388b0c48398cb07814f93430062fa675433d575',
        'D070-2020_DEM_ELEV.zip': '1aa86d3e059eed222a97b5ec07afe13f33ad4dca16f4b4acb1de475a027a971f',
        'D070-2020_SENTINEL1-ASC_TS.zip': '35df65d2126aa359987dd4ff2e3c51806902d0167057981db2245693c0d5526f',
        'D070-2020_SENTINEL1-DESC_TS.zip': '01a724822f0c74bbf814dc4d602513fa8b1c0f3a7269299fb924cbdec81cf120',
        'D070-2020_SENTINEL2_MSK-SC.zip': 'eed4eb267e2121a48929024323fa75ba365f7682a0d69fcf066011153f8b7abd',
        'D070-2020_SENTINEL2_TS.zip': 'dfd47e40fa5893e505f3702a87ef7c4ab3d101e957f3845346051d536a2ee94b',
        'D070-2020_SPOT_RGBI.zip': 'b704ff54f87487a7afa27007004c2aa58def1225e78a7331a6fb083a45a6594c',
        'D071-195X_AERIAL-RLT_PAN.zip': 'e687f38134ab10c575ccd056d157a6662667537170a75807d64d2ef96db6e870',
        'D071-2020_AERIAL_LABEL-COSIA.zip': '1b42160ce40ba4a38cb53adc2b7c16de42da4087d70818391209c46a950a0130',
        'D071-2020_AERIAL_RGBI.zip': 'aab555dc4233c8ac756888a1e6c4f99f463ba8a9d0a0a77dbe57b9c703001842',
        'D071-2020_ALL_LABEL-LPIS.zip': 'e201b58ec33a795e1717d62fbb01c5dc52682791c21c0fc0ac28dc4d83b6e4f4',
        'D071-2020_DEM_ELEV.zip': 'b188b29e505498db25c958314a1c9ea738d9dbe58b458fe90ad02e9866462ae6',
        'D071-2020_SENTINEL1-ASC_TS.zip': 'aca08b315e1c73090755ab75f87c010321da49ff15e712c18a5120854e790a30',
        'D071-2020_SENTINEL1-DESC_TS.zip': '7e83335adb24871e0aa085875d1052b1ec84932ce250227f7bf6bca83fe02aeb',
        'D071-2020_SENTINEL2_MSK-SC.zip': 'fdbb417bfacd6cd69b6e635f2ae0aa1db443a63dbaef673041811c420c13dc2b',
        'D071-2020_SENTINEL2_TS.zip': 'bc549a241f15e59e67cbb2f934da405933fe4653e27e358df5f3ee03747f3f39',
        'D071-2020_SPOT_RGBI.zip': '7a118d7f857df46e87b42614d903fa6db37ffbe23c57880f356e6aebf930acaa',
        'D072-195X_AERIAL-RLT_PAN.zip': '42c2518e646da945c06207dac550e3809881df8dcfbba75c8c3ce2639c670336',
        'D072-2019_AERIAL_LABEL-COSIA.zip': 'fd68909d9b7d6abcec2c43d1c27f9125983fc2b24390f5366fafbc7c22f9ccd6',
        'D072-2019_AERIAL_RGBI.zip': '0f9ad113bb413a1294529e439e659ad7eee2b8d34e7ffb61f9e341a52dce4342',
        'D072-2019_ALL_LABEL-LPIS.zip': 'cc093a6d37b8d259b542569000bbbdd619ead4f36d230367fd27dec574911261',
        'D072-2019_DEM_ELEV.zip': 'a3759a61195b4cc683e88cbe604ec2a3197159738d7ce889ba6beb23f2504f16',
        'D072-2019_SENTINEL1-ASC_TS.zip': 'f79e19f81cd4c12f212cea5d71cdca5340e2a8dddfeb256a84ada3014fd3a63b',
        'D072-2019_SENTINEL1-DESC_TS.zip': 'e31cc40524401afb30fa8f101e2c0bd740807daa61a42761c4a31ffea632a3d3',
        'D072-2019_SENTINEL2_MSK-SC.zip': 'c9aab7778200c36c885a1f70da138bcc36b5a483bed5979aff4ef081cc2e58f6',
        'D072-2019_SENTINEL2_TS.zip': '7682a1c17d7b7fd936a18d2fd1f091f8cd05c0acd84b3e0f59fc9113310bc5ff',
        'D072-2019_SPOT_RGBI.zip': '322cc08a14fd2921d33a6ec5b2f889f13e0bbcd76bf747ede3ef3b25141bb9a4',
        'D073-195X_AERIAL-RLT_PAN.zip': 'f9d9f2e1bc10663d6177aae78833800197334907aa60a984b2ce20f7e28348ff',
        'D073-2022_AERIAL_LABEL-COSIA.zip': 'e97f3bef720db422d12b1cbb757f221ff9b5bb297efc45596117b30f4517c884',
        'D073-2022_AERIAL_RGBI.zip': 'd08bb41f882955a99189e33d961f5c373277731bbc15af178a85aaa5f6a4de16',
        'D073-2022_ALL_LABEL-LPIS.zip': '578420ef4b78073a3a076adfa9c521c49a72cfb252f251c56fd5fe3606338394',
        'D073-2022_DEM_ELEV.zip': 'd06474827d02e11b82f28112cd18973cab13393b2dc0ba9e0159024e84fc8ab6',
        'D073-2022_SENTINEL1-ASC_TS.zip': 'da5c68bfb2a311f2eeca22fd0d86dc4d3128526b901a99e5031a012757e36b5d',
        'D073-2022_SENTINEL1-DESC_TS.zip': '76310c29d794277270d1ab394212bdf846e5c5adef7cd38207ecc0f545b66c5c',
        'D073-2022_SENTINEL2_MSK-SC.zip': '8db885d9ad2462da057f5543cd01a2b833744206e519785e494e288b43637711',
        'D073-2022_SENTINEL2_TS.zip': 'f39c044e46e79d7e1646412b2979eeab2f31b01895a88aac61923713833769ef',
        'D073-2022_SPOT_RGBI.zip': '4a58d01d2a1f139fd070f9d79b389c1599c686d07916bd1f43bff4c43e81e261',
        'D074-195X_AERIAL-RLT_PAN.zip': 'e12b10b4813b3ed6e881e0d82d6d19abc1e35befb5925a1c9c49fef5fa4751df',
        'D074-2020_AERIAL_LABEL-COSIA.zip': '472737fa5e93a08e072441330eec97ed87f47732d67fa681503561e7e2508f56',
        'D074-2020_AERIAL_RGBI.zip': '2e3b6ce781fbd18af09e6b6eb8fef3c7a6f017fc22307b36fbe3e6c5f2881ad3',
        'D074-2020_ALL_LABEL-LPIS.zip': 'd5769a57c34e43c32e8dddb9849611afb9b11a76da8baacf395914a78b001138',
        'D074-2020_DEM_ELEV.zip': '95673850b5a9334d16dc200ca2a43cb144c9e2d1ae9664f267d6234934864b84',
        'D074-2020_SENTINEL1-ASC_TS.zip': '973b869b6ed27b02e50209c6739b1c53a98d68d70c6104f57733cc42c4e075c5',
        'D074-2020_SENTINEL1-DESC_TS.zip': '78a206174edcf3e4ee90dfb04a2be3fb449f79f89ee36ab62e5c0c215823fb6d',
        'D074-2020_SENTINEL2_MSK-SC.zip': '7d7e3dfdf132cf4301721f2187a2e5b4242070f80b214d24eca7abb6a7a77bd1',
        'D074-2020_SENTINEL2_TS.zip': '3574339155ad214297140c86bab8831cb2ba1302a2557937748df3f5454f4002',
        'D074-2020_SPOT_RGBI.zip': '9485186db355b51ef84b0f98d53646a446308c310b7137a3b38da38792e6dc05',
        'D075-195X_AERIAL-RLT_PAN.zip': '546fb02afb695c08cbb0628aac2a25efc195076bc7a1103cb5da5e247e4b04bb',
        'D075-2021_AERIAL_LABEL-COSIA.zip': 'b56de5c671f5cc78e08bd8ae689abe0bf7266fb32e04fba526cadc48801730ec',
        'D075-2021_AERIAL_RGBI.zip': 'df142615e415ce885157e29b85280feed27eedec1b20625de668fa4a44e1d22a',
        'D075-2021_ALL_LABEL-LPIS.zip': 'b519cd50f2d5aa5689398fff2e88ee8e139d7c6d59ccc7d5651cf1867a623de0',
        'D075-2021_DEM_ELEV.zip': '38f553c3b18e7accba6df5537956249a6f90a0e8fce4198754ae6ca60c3998bc',
        'D075-2021_SENTINEL1-ASC_TS.zip': 'eb8d837cde1cd46b1d8b6a8c448263afc8337545df17969ab8a43e4dce21504b',
        'D075-2021_SENTINEL1-DESC_TS.zip': '99f8b38677d36611c735321d6783e850f82c2678aad295b8add33e6185b76418',
        'D075-2021_SENTINEL2_MSK-SC.zip': '4f8d804bbff58279a49a8ec7c63889789de021111473a50caa7def602bfd1878',
        'D075-2021_SENTINEL2_TS.zip': '4dce695e999f3830aaba3be382babb64d617c290eded5b3b7c19585f9229d146',
        'D075-2021_SPOT_RGBI.zip': 'd0a001ac6f49f8d5d169daca35fb0ad226bc3049b994a19b737f4cce037987bf',
        'D076-195X_AERIAL-RLT_PAN.zip': '21c6f5a1036429b14e4fc2b7455e18b86abaf188bedd0e35e85a29fdc44b8907',
        'D076-2019_AERIAL_LABEL-COSIA.zip': 'c4bde21cfdda0ea8ddeb8cf367162fbabd13e0175bc7b259626698e070fe670c',
        'D076-2019_AERIAL_RGBI.zip': 'efed348b38e651c225dd8a4961a1d62c5fe5f5db16cbd0ffb570f41efe8b893b',
        'D076-2019_ALL_LABEL-LPIS.zip': 'b0c9963efc87b64ff3ef4d2a7e6552900f625d68e5acd4f182c5f41c574e706a',
        'D076-2019_DEM_ELEV.zip': '4e58ddc3eedbe9e89fb5decbd52179fe54eba07447fb3e25a5cb1ef238046f49',
        'D076-2019_SENTINEL1-ASC_TS.zip': 'd110599a1e378758fbb112294e2597badc749f7ca8b7b649613d5a0c970623c3',
        'D076-2019_SENTINEL1-DESC_TS.zip': '189db189d7d1486d6121ce219468edbd5485e4c926b8e5ae3ec7e9192792996f',
        'D076-2019_SENTINEL2_MSK-SC.zip': 'ae8704c4c939a19b09fc6fab559490739144cf63fe04517df65d4b477ff0442d',
        'D076-2019_SENTINEL2_TS.zip': 'd81bf7b981c4ec30eebd222e4e61dbf80c633e3f9f6fe0735fc347f1fb6fe5b3',
        'D076-2019_SPOT_RGBI.zip': '0785b4033f77cfe5ed2396ac8cec4eb510a9d31fc7ea3925b77e8b462b8e91fc',
        'D077-195X_AERIAL-RLT_PAN.zip': 'ef0ed53a661acfc26b3812f0da65ef3ff4ead58619ec5917d133e9499f20d037',
        'D077-2021_AERIAL_LABEL-COSIA.zip': 'd1afda87d656619c6775b263bedce4d1a94f63ae9f97a69f4e8dbcd0c10c50fd',
        'D077-2021_AERIAL_RGBI.zip': '05f908b250e2b1a8d1cde82f3842e7614c39e65a86ea0a99569f1b121ba45f00',
        'D077-2021_ALL_LABEL-LPIS.zip': '29a377f5b883e34f97cf584e0035fd4e3f4b6c81d06d088ab1d4b3e956d0c3ef',
        'D077-2021_DEM_ELEV.zip': '0143786df00fa76fc04e2d7f0fe6f0d59aa6283f771dea60aa509ea32a099227',
        'D077-2021_SENTINEL1-ASC_TS.zip': '11e97d163ffeda97a482f2621cb7b304a3a0a844c8896341c5afcc159770ac4a',
        'D077-2021_SENTINEL1-DESC_TS.zip': '373cdc027578d3d53c64ae67dd5f10246ccc21bb0209a9b5b531edc03fc81bbd',
        'D077-2021_SENTINEL2_MSK-SC.zip': 'ba16a8e65ce866a9bbd53e9f48c41f9a96e5266086435f8746bac0776f4630b0',
        'D077-2021_SENTINEL2_TS.zip': 'c64dd7e12ef21eb27e3e1ed78f26ddd4d62ef8c29ddc16e544e8340fe5325f44',
        'D077-2021_SPOT_RGBI.zip': '9249c60778255209c3c6392ec0b87b7729cf1ff3b28fe8083e07e35878fb33af',
        'D078-195X_AERIAL-RLT_PAN.zip': 'c2b0b866021e333445e75cd3efc37a10bc44fe8a7923dbea9cfc74b7fae02e08',
        'D078-2021_AERIAL_LABEL-COSIA.zip': '2d9de780f42ba2dd35732578534dc236ed5a1dd39540b73a7a137c236f2bd71f',
        'D078-2021_AERIAL_RGBI.zip': 'da8758d4290f67bd85a654144ed94b7ac215c0f151e4a66b2ccbd4a4bc5722de',
        'D078-2021_ALL_LABEL-LPIS.zip': '0751bdbcec0e2b0937161e902a6d63b9f8922b7c331cc3959ba9bd1abe9e97a8',
        'D078-2021_DEM_ELEV.zip': '765206309ff40a213705e27b6b26e57442fc2281f6d0a27371b45812b8aea300',
        'D078-2021_SENTINEL1-ASC_TS.zip': '5e68c04c98cbf20c28991357d1e8177a59ddc2e2a67018f7fee1b99a3140f75b',
        'D078-2021_SENTINEL1-DESC_TS.zip': 'b3ade867c4ae513501f9c23bcd5ae8e8d6c8197a512fe44a02086710a0360a14',
        'D078-2021_SENTINEL2_MSK-SC.zip': '5f8ae8ada64a85050fe79674b6b6cb6ad16086d940a67db4ddec9469704dc450',
        'D078-2021_SENTINEL2_TS.zip': '656228ccae32780f3f867e54706a481166dde636ff8c2bd3a84a52ab429a4b50',
        'D078-2021_SPOT_RGBI.zip': 'a3aa8a37b57d67728586839fd35409d84c625a7d3a05f30835d99501f633c451',
        'D080-195X_AERIAL-RLT_PAN.zip': '0d2b360b8823244f8a84a255bae7ddc6d7c6b930b86915ac69e3ad307dc5a55a',
        'D080-2017_AERIAL_LABEL-COSIA.zip': '34cf926ae10860268308c19e1cefa99ba73ce360391fd85996f5c6b700d12b8e',
        'D080-2017_AERIAL_RGBI.zip': '7d7519a944e6b5ddbcc97b2165b7c16ba47b385d08d35f1377fa8c1ccc7aad94',
        'D080-2017_ALL_LABEL-LPIS.zip': 'aecc69ae9c8bcd5b965496599c2c2fa1e9042b78e9bb838150ae2466f386fcab',
        'D080-2017_DEM_ELEV.zip': 'f3050a0f17e0cba4d0be47c45c06c6f3a0845bcac04ba4054fbb162b30bb2172',
        'D080-2017_SENTINEL1-ASC_TS.zip': 'cd648dfa1f8e57cd96cd4bb4fc1dc8edf5be24f2a219020ca79d9eb826aeae68',
        'D080-2017_SENTINEL1-DESC_TS.zip': '866a7e5f7fe04dc896a0a9731dbfb8f2ad7776ad11a88809ad84ca112ebc6e90',
        'D080-2017_SENTINEL2_MSK-SC.zip': '0608effb5c3666da707fab545f021498edd4cad580d6f49f9afe2daeccc8c213',
        'D080-2017_SENTINEL2_TS.zip': '7b5be7f6469b552b44ca16fee9e2168f06aa86d4f4f86ec0ba78049d91fb08e8',
        'D080-2017_SPOT_RGBI.zip': '0ef4d114a1f95f65ac3acd7e043b83c6f153a8297d45cbdc3cb32325f54258d7',
        'D080-2021_AERIAL_LABEL-COSIA.zip': 'efad47db7c12bf85ee604ad6d8b7dca7004b0cff306f4590239e879918a24b72',
        'D080-2021_AERIAL_RGBI.zip': 'c2af03b3862965af607f75268cd23178e98bd4c07a70b2df8a16030507580d71',
        'D080-2021_ALL_LABEL-LPIS.zip': '43e2e9d3497c6621029ab1623507ca2e9d6ca81cdcbdcff4944dcbef971ce5c0',
        'D080-2021_DEM_ELEV.zip': '77fc3fa912dc1c31a6f4f35d236ac0af5cb1b334c48f63248e3d9168fee8903c',
        'D080-2021_SENTINEL1-ASC_TS.zip': 'b6916012258975a727cac158d10e89bccc2373cf7bf4c75f7fa50339ae7371a2',
        'D080-2021_SENTINEL1-DESC_TS.zip': '19d3f675f53a7e761c08ade5b26d4445b511cac694b9b958585a75621527ac88',
        'D080-2021_SENTINEL2_MSK-SC.zip': 'c9aca27ebeb9b7c982761c52e508aab7f744fb2925b2c464a6b702a3c2e317ac',
        'D080-2021_SENTINEL2_TS.zip': 'abb14992f2b0ba26ee541a2552d0dba7dcc58c905ed85c13af3e26243c0f07a9',
        'D080-2021_SPOT_RGBI.zip': 'db5d3a14da4142e1e60d7c813d5fa48d937f63b95d7bf7584256ddcc47d0d662',
        'D081-195X_AERIAL-RLT_PAN.zip': 'bd6c1c45b15e4685f68c199026a8ad5a0648271ad1fe208a0e19d1b01e8dd98d',
        'D081-2020_AERIAL_LABEL-COSIA.zip': '3be853fdc7bcf096aa7a80e54a782a4230b0a7584568239793bac1c1098d3475',
        'D081-2020_AERIAL_RGBI.zip': '15903fbd1e08cb90916d1db26632b3693481b47cf52723c0044ced580f86d1b0',
        'D081-2020_ALL_LABEL-LPIS.zip': 'f030496d22697e862f7a2bbe0c4aa06aeed0296a62ec2df78f4cda5973ec5f82',
        'D081-2020_DEM_ELEV.zip': '71e1473ddcf4b544e44e81b3cbb4bb52f6b7f9efcbdf2ae3a88865e3cd2badb7',
        'D081-2020_SENTINEL1-ASC_TS.zip': '1fcd067b67ce5ef7b654830f49c0c70a607faac140f1a4fcd77144fe6d391f23',
        'D081-2020_SENTINEL1-DESC_TS.zip': '86a9163ed8bd288846d39527fecb196837aaa82bd6b1dab4d53f5f50f756049e',
        'D081-2020_SENTINEL2_MSK-SC.zip': '6613ecb6d89a12fddd2d212e39f74be0d009c6b71848a742af9e85e8c55eec7b',
        'D081-2020_SENTINEL2_TS.zip': '02c73d7faf807891660944df476243e5ac7d7bdc1fee60324bb5af2eae222c5b',
        'D081-2020_SPOT_RGBI.zip': '5de9af268e3d031a4c81d589430d4088bc524b1b33764852bb1b0caf88527a43',
        'D083-195X_AERIAL-RLT_PAN.zip': '582cd0971e9d31b2addc27a8b81f0d69af754e5404f4c110d955adf80527ed10',
        'D083-2020_AERIAL_LABEL-COSIA.zip': '77469a2b5da5c9e2988f38d68d5b171bc05bf3160d3a352c74e48cab31e53fc9',
        'D083-2020_AERIAL_RGBI.zip': '63c5d492120b52a8c5c637014c593ebe9fce6b74d460f87a22ef3f92152f1e7b',
        'D083-2020_ALL_LABEL-LPIS.zip': '42e80b5d9216737cda7158ab492ff66d99ccd2ac474f49b2a08c5eeb4504b2ae',
        'D083-2020_DEM_ELEV.zip': 'f14defad6a48fe04c6b9a71468a5fa10b7e4d38a693766c5ac52f31c827f77a7',
        'D083-2020_SENTINEL1-ASC_TS.zip': '6612b26a003403d874cf729ce00903f7ff4e861fac5d17a85465b74c83746208',
        'D083-2020_SENTINEL1-DESC_TS.zip': 'de566f8c7fc36fe589e686b0d3db523adaf220e1678c3c0be452c384a5eb7765',
        'D083-2020_SENTINEL2_MSK-SC.zip': '6f06b37589da6ebacc78ba9eb847542c7038ababa619c01855b5abcdff407769',
        'D083-2020_SENTINEL2_TS.zip': '3de13e9db9ce5bf4b908dbc141e9e44068b695b0672440a07c4983636114d4d6',
        'D083-2020_SPOT_RGBI.zip': '7cc577629eac1a37b26419afde089a5bb101d4f4fb776e35d8cd703011561990',
        'D084-195X_AERIAL-RLT_PAN.zip': '85d387ef47641def0c9ef9b78e7e2a559f3c2e227dd550fc6c9d7ec69f8e0d6d',
        'D084-2021_AERIAL_LABEL-COSIA.zip': 'dab4e256517a314ae49229f560295f4e283ac16ff55ea07ecd0eaaa6533d7ab4',
        'D084-2021_AERIAL_RGBI.zip': 'a4f3404c9fb76dcefbce8602dab0351f85253a7036e0317189bf153f3d8db59a',
        'D084-2021_ALL_LABEL-LPIS.zip': '0a62d885bd949a4ae5b4116e29888532676aca5fd18575cf85902e46295689df',
        'D084-2021_DEM_ELEV.zip': 'cc0c4c7983a92ab516e4dfd998450e79013001f162423121923aa3cd157f48e2',
        'D084-2021_SENTINEL1-ASC_TS.zip': 'dd3b3a55eedd090937fa8496b0254d98209ae3eb7d701f084d941b7f957c9fcd',
        'D084-2021_SENTINEL1-DESC_TS.zip': '5a27c4edd08817a08ec6d525f1bf9d3580cc298d64af02a1dde5e69db6194f23',
        'D084-2021_SENTINEL2_MSK-SC.zip': '5a9f727d7aa222b2cc402052594813ab464371f952e32a34472cb93482d6cce1',
        'D084-2021_SENTINEL2_TS.zip': '76fad88e192f212f3bb76fac6116eade91a67abac42d98eec40525d28e42c2a8',
        'D084-2021_SPOT_RGBI.zip': '1175e296757131e1c68c7d837ae91d79f38d20c62169c1b7fef0e8295e9b4124',
        'D085-195X_AERIAL-RLT_PAN.zip': '8f8e5fe267f419a09e7b8ef10de38bbc5a715f8b07fb1f5d1d3c958b180467b9',
        'D085-2019_AERIAL_LABEL-COSIA.zip': '72fdc37a78c05539ded08dc449efbe2277e20ce214dc2addf86e9d4b172226db',
        'D085-2019_AERIAL_RGBI.zip': '82447cff44118b93bb795a3fb9f872c3e5c0cc4613b0dcbcfe6e7fc3bb289682',
        'D085-2019_ALL_LABEL-LPIS.zip': '4fcdff6a82f64158d0074ede82ab8a2bca4d778fd2923405a0966b736150c8f8',
        'D085-2019_DEM_ELEV.zip': 'af697bee9f8fc9757e72c70014ee168fce57aa960190d866811a5960af31ecf6',
        'D085-2019_SENTINEL1-ASC_TS.zip': 'fbcbe7a2632abde4d5e81f71e6281645853c28b28236b2bf15d9357b49c3c723',
        'D085-2019_SENTINEL1-DESC_TS.zip': 'f78846f156c338e3fa6d6a96559302cdfa886b5fea46204482e9322be5152567',
        'D085-2019_SENTINEL2_MSK-SC.zip': '2600b74840df332b980bd30888b502159b85b3833e612f38363e236566c30221',
        'D085-2019_SENTINEL2_TS.zip': 'a1d1921810c8a3cadd044aaffa7f6f760bc99795b3805817da8887a0a4fd62f7',
        'D085-2019_SPOT_RGBI.zip': '501bfeb3b5fcbcdef208fd5c6127953d9c19a5e1d4682041ebd0d376ebad8d82',
        'D086-195X_AERIAL-RLT_PAN.zip': 'ddb397e9213e9a662b06e48b33e5f5282bf8d6b557b9e4e9f2119d5afdb25fc2',
        'D086-2020_AERIAL_LABEL-COSIA.zip': 'b5aa846904c5de76b305abd239684082c967bf89b928930aa6146414acebeeee',
        'D086-2020_AERIAL_RGBI.zip': '16efbd2afee6ec174ec2e2f2c73ccecabbebf7c10a7e74b2ca66545e9519f12c',
        'D086-2020_ALL_LABEL-LPIS.zip': '58231d06e4adc6564e378214b76415011f1bc1f2eafef18e08e5e5fc3e0cfd58',
        'D086-2020_DEM_ELEV.zip': '453f5f49e42cd60051ab87255ac8dffd779115d256c60455f2e843f638097261',
        'D086-2020_SENTINEL1-ASC_TS.zip': '961e1db817cb5f2578063f2218807d96aee99e8ba55ced0fbef87b621a2de17b',
        'D086-2020_SENTINEL1-DESC_TS.zip': '322ff1f5660b087527f32dc7b2abcf8f4bd742804128ba1926f5f4565c248328',
        'D086-2020_SENTINEL2_MSK-SC.zip': 'bd54e2b6dec0e6ce387840f167b0ebda26daa430ed797329f883419b207bad8b',
        'D086-2020_SENTINEL2_TS.zip': '4a0866dd8935b41336adb4a442a33f9c2a938757c9bda980fa4cd87df6fa818d',
        'D086-2020_SPOT_RGBI.zip': '10127d041683a4cb237e3c03133c0d17db02348633dd065b4a3d9f8ff3481e3a',
        'D091-195X_AERIAL-RLT_PAN.zip': '3df39e5f6b7e85b5b8440e459d30c4ec032bde2c9d16f812d47a22d6ce56d3ff',
        'D091-2021_AERIAL_LABEL-COSIA.zip': 'dcb16c57ed316e215f599675618572976f0eeb35679b0b15197969964d3e5e52',
        'D091-2021_AERIAL_RGBI.zip': 'c9774152596a85180dca902e9f6ef07ffd279a3412e1adcd9bd030a3914be1ea',
        'D091-2021_ALL_LABEL-LPIS.zip': '7f5bc8f86f71f9b0fd06158b0dc24befed473ab37d87cdb5f2c6d21885391ef3',
        'D091-2021_DEM_ELEV.zip': '09a22af95dc993e92b24556a99c81b3ad3ea1663df08a1f03f81d8b6b10fd628',
        'D091-2021_SENTINEL1-ASC_TS.zip': '45b163262b498083586dde0048f84bb2b18b2874004ce648d410658304e49b87',
        'D091-2021_SENTINEL1-DESC_TS.zip': 'f3e62c094ca511b8466d85a4421272676e4dce9241767052efc5afaf2bf94163',
        'D091-2021_SENTINEL2_MSK-SC.zip': '3f7109bd76e9e57a35491bf2931cfed677788912ea3fbb926097a47d298bf1f1',
        'D091-2021_SENTINEL2_TS.zip': '3770fdaabc5f5fed577e9f2a87b502d88084c381a42938eabd47917b105eff72',
        'D091-2021_SPOT_RGBI.zip': 'f11a9e8b9d1457eb457104af10e05a3753430fc11b9408b1b9ad9933321418af',
        'GLOBAL_ALL_MTD.zip': '8e52b53afba43138343140c926fbad95cdfd2682b3e8e7fffb153ea58211bc59',
    }

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
            'split_toy',
        ] = 'split_1',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
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
            checksum: If True, check the SHA-256 of downloaded files (may be slow).
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
        self.checksum = checksum
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

        if zip_path.exists():
            if self.checksum and not check_integrity(
                zip_path, sha256=self.sha256s[zip_path.name]
            ):
                raise RuntimeError('Dataset found, but corrupted.')
        elif self.download:
            sha256 = self.sha256s[zip_path.name] if self.checksum else None
            download_url(
                f'{self.download_link}/GLOBAL_ALL_MTD.zip', str(root), sha256=sha256
            )
        else:
            raise DatasetNotFoundError(self)
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

                    if zip_path.is_file():
                        if self.checksum and not check_integrity(
                            zip_path, sha256=self.sha256s[zip_path.name]
                        ):
                            raise RuntimeError('Dataset found, but corrupted.')
                    elif self.download:
                        sha256 = self.sha256s[zip_path.name] if self.checksum else None
                        download_url(
                            f'{self.download_link}/{dir_name}.zip',
                            str(self.root),
                            sha256=sha256,
                        )
                    else:
                        raise DatasetNotFoundError(self)

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

        rgb_indices = [
            self.sentinel2_ts_bands.index(band) for band in self.sentinel2_ts_rgb_bands
        ]
        rgb_image = last_timepoint[rgb_indices].transpose(1, 2, 0)
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
        vv = quantile_normalization(data[-1, 0])
        ax.imshow(vv.numpy(), cmap='gray')
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
        dataset_type: Literal[
            'land_cover', 'crop_type', 'crop_type_2', 'crop_type_3'
        ] = 'land_cover',
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
            dataset_type: Type of labels to use. Choose ``land_cover`` for
                19-class COSIA labels or ``crop_type`` for 23-class LPIS crop
                classification labels (baseline) or ``crop_type_2`` for 31-class
                LPIS crop classification labels or ``crop_type_3`` for 46-class
                LPIS crop classification labels. ``crop_type_2`` and
                ``crop_type_3`` are deeper levels of the LPIS crop classification
                labels.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            ValueError: If *dataset_type* is not ``land_cover``, ``crop_type``,
                ``crop_type_2``, or ``crop_type_3``.
            ValueError: If no data modality is enabled.
            ValueError: If an invalid band name is provided.

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
