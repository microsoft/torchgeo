# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SatlasPretrain dataset."""

import os
from collections.abc import Callable, Iterable
from typing import ClassVar, TypedDict

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, extract_archive, which


class _Task(TypedDict, total=False):
    BackgroundInvalid: bool
    categories: list[str]
    colors: list[list[int]]
    type: str


# https://github.com/allenai/satlas/blob/main/satlas/model/dataset.py
TASKS: dict[str, _Task] = {
    'polyline_bin_segment': {
        'type': 'bin_segment',
        'categories': [
            'airport_runway',
            'airport_taxiway',
            'raceway',
            'road',
            'railway',
            'river',
        ],
        'colors': [
            [255, 255, 255],  # (white) airport_runway
            [192, 192, 192],  # (light grey) airport_taxiway
            [160, 82, 45],  # (sienna) raceway
            [255, 255, 255],  # (white) road
            [144, 238, 144],  # (light green) railway
            [0, 0, 255],  # (blue) river
        ],
    },
    'bin_segment': {
        'type': 'bin_segment',
        'categories': [
            'aquafarm',
            'lock',
            'dam',
            'solar_farm',
            'power_plant',
            'gas_station',
            'park',
            'parking_garage',
            'parking_lot',
            'landfill',
            'quarry',
            'stadium',
            'airport',
            'airport_runway',
            'airport_taxiway',
            'airport_apron',
            'airport_hangar',
            'airstrip',
            'airport_terminal',
            'ski_resort',
            'theme_park',
            'storage_tank',
            'silo',
            'track',
            'raceway',
            'wastewater_plant',
            'road',
            'railway',
            'river',
            'water_park',
            'pier',
            'water_tower',
            'street_lamp',
            'traffic_signals',
            'power_tower',
            'power_substation',
            'building',
            'bridge',
            'road_motorway',
            'road_trunk',
            'road_primary',
            'road_secondary',
            'road_tertiary',
            'road_residential',
            'road_service',
            'road_track',
            'road_pedestrian',
        ],
        'colors': [
            [32, 178, 170],  # (light sea green) aquafarm
            [0, 255, 255],  # (cyan) lock
            [173, 216, 230],  # (light blue) dam
            [255, 0, 255],  # (magenta) solar farm
            [255, 165, 0],  # (orange) power plant
            [128, 128, 0],  # (olive) gas station
            [0, 255, 0],  # (green) park
            [47, 79, 79],  # (dark slate gray) parking garage
            [128, 0, 0],  # (maroon) parking lot
            [165, 42, 42],  # (brown) landfill
            [128, 128, 128],  # (grey) quarry
            [255, 215, 0],  # (gold) stadium
            [255, 105, 180],  # (pink) airport
            [255, 255, 255],  # (white) airport_runway
            [192, 192, 192],  # (light grey) airport_taxiway
            [128, 0, 128],  # (purple) airport_apron
            [0, 128, 0],  # (dark green) airport_hangar
            [248, 248, 255],  # (ghost white) airstrip
            [240, 230, 140],  # (khaki) airport_terminal
            [192, 192, 192],  # (silver) ski_resort
            [0, 96, 0],  # (dark green) theme_park
            [95, 158, 160],  # (cadet blue) storage_tank
            [205, 133, 63],  # (peru) silo
            [154, 205, 50],  # (yellow green) track
            [160, 82, 45],  # (sienna) raceway
            [218, 112, 214],  # (orchid) wastewater_plant
            [255, 255, 255],  # (white) road
            [144, 238, 144],  # (light green) railway
            [0, 0, 255],  # (blue) river
            [255, 240, 245],  # (lavender blush) water_park
            [65, 105, 225],  # (royal blue) pier
            [238, 130, 238],  # (violet) water_tower
            [75, 0, 130],  # (indigo) street_lamp
            [233, 150, 122],  # (dark salmon) traffic_signals
            [255, 255, 0],  # (yellow) power_tower
            [255, 255, 0],  # (yellow) power_substation
            [255, 0, 0],  # (red) building
            [64, 64, 64],  # (dark grey) bridge
            [255, 255, 255],  # (white) road_motorway
            [255, 255, 255],  # (white) road_trunk
            [255, 255, 255],  # (white) road_primary
            [255, 255, 255],  # (white) road_secondary
            [255, 255, 255],  # (white) road_tertiary
            [255, 255, 255],  # (white) road_residential
            [255, 255, 255],  # (white) road_service
            [255, 255, 255],  # (white) road_track
            [255, 255, 255],  # (white) road_pedestrian
        ],
    },
    'land_cover': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'background',
            'water',
            'developed',
            'tree',
            'shrub',
            'grass',
            'crop',
            'bare',
            'snow',
            'wetland',
            'mangroves',
            'moss',
        ],
        'colors': [
            [0, 0, 0],  # unknown
            [0, 0, 255],  # (blue) water
            [255, 0, 0],  # (red) developed
            [0, 192, 0],  # (dark green) tree
            [200, 170, 120],  # (brown) shrub
            [0, 255, 0],  # (green) grass
            [255, 255, 0],  # (yellow) crop
            [128, 128, 128],  # (grey) bare
            [255, 255, 255],  # (white) snow
            [0, 255, 255],  # (cyan) wetland
            [255, 0, 255],  # (pink) mangroves
            [128, 0, 128],  # (purple) moss
        ],
    },
    'tree_cover': {'type': 'regress', 'BackgroundInvalid': True},
    'crop_type': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': [
            'invalid',
            'rice',
            'grape',
            'corn',
            'sugarcane',
            'tea',
            'hop',
            'wheat',
            'soy',
            'barley',
            'oats',
            'rye',
            'cassava',
            'potato',
            'sunflower',
            'asparagus',
            'coffee',
        ],
        'colors': [
            [0, 0, 0],  # unknown
            [0, 0, 255],  # (blue) rice
            [255, 0, 0],  # (red) grape
            [255, 255, 0],  # (yellow) corn
            [0, 255, 0],  # (green) sugarcane
            [128, 0, 128],  # (purple) tea
            [255, 0, 255],  # (pink) hop
            [0, 128, 0],  # (dark green) wheat
            [255, 255, 255],  # (white) soy
            [128, 128, 128],  # (grey) barley
            [165, 42, 42],  # (brown) oats
            [0, 255, 255],  # (cyan) rye
            [128, 0, 0],  # (maroon) cassava
            [173, 216, 230],  # (light blue) potato
            [128, 128, 0],  # (olive) sunflower
            [0, 128, 0],  # (dark green) asparagus
            [92, 64, 51],  # (dark brown) coffee
        ],
    },
    'point': {
        'type': 'detect',
        'categories': [
            'background',
            'wind_turbine',
            'lighthouse',
            'mineshaft',
            'aerialway_pylon',
            'helipad',
            'fountain',
            'toll_booth',
            'chimney',
            'communications_tower',
            'flagpole',
            'petroleum_well',
            'water_tower',
            'offshore_wind_turbine',
            'offshore_platform',
            'power_tower',
        ],
        'colors': [
            [0, 0, 0],
            [0, 255, 255],  # (cyan) wind_turbine
            [0, 255, 0],  # (green) lighthouse
            [255, 255, 0],  # (yellow) mineshaft
            [0, 0, 255],  # (blue) pylon
            [173, 216, 230],  # (light blue) helipad
            [128, 0, 128],  # (purple) fountain
            [255, 255, 255],  # (white) toll_booth
            [0, 128, 0],  # (dark green) chimney
            [128, 128, 128],  # (grey) communications_tower
            [165, 42, 42],  # (brown) flagpole
            [128, 0, 0],  # (maroon) petroleum_well
            [255, 165, 0],  # (orange) water_tower
            [255, 255, 0],  # (yellow) offshore_wind_turbine
            [255, 0, 0],  # (red) offshore_platform
            [255, 0, 255],  # (magenta) power_tower
        ],
    },
    'rooftop_solar_panel': {
        'type': 'detect',
        'categories': ['background', 'rooftop_solar_panel'],
        'colors': [
            [0, 0, 0],
            [255, 255, 0],  # (yellow) rooftop_solar_panel
        ],
    },
    'building': {
        'type': 'instance',
        'categories': ['background', 'ms_building'],
        'colors': [
            [0, 0, 0],
            [255, 255, 0],  # (yellow) building
        ],
    },
    'polygon': {
        'type': 'instance',
        'categories': [
            'background',
            'aquafarm',
            'lock',
            'dam',
            'solar_farm',
            'power_plant',
            'gas_station',
            'park',
            'parking_garage',
            'parking_lot',
            'landfill',
            'quarry',
            'stadium',
            'airport',
            'airport_apron',
            'airport_hangar',
            'airport_terminal',
            'ski_resort',
            'theme_park',
            'storage_tank',
            'silo',
            'track',
            'wastewater_plant',
            'power_substation',
            'pier',
            'crop',
            'water_park',
        ],
        'colors': [
            [0, 0, 0],
            [255, 255, 0],  # (yellow) aquafarm
            [0, 255, 255],  # (cyan) lock
            [0, 255, 0],  # (green) dam
            [0, 0, 255],  # (blue) solar_farm
            [255, 0, 0],  # (red) power_plant
            [128, 0, 128],  # (purple) gas_station
            [255, 255, 255],  # (white) park
            [0, 128, 0],  # (dark green) parking_garage
            [128, 128, 128],  # (grey) parking_lot
            [165, 42, 42],  # (brown) landfill
            [128, 0, 0],  # (maroon) quarry
            [255, 165, 0],  # (orange) stadium
            [255, 105, 180],  # (pink) airport
            [192, 192, 192],  # (silver) airport_apron
            [173, 216, 230],  # (light blue) airport_hangar
            [32, 178, 170],  # (light sea green) airport_terminal
            [255, 0, 255],  # (magenta) ski_resort
            [128, 128, 0],  # (olive) theme_park
            [47, 79, 79],  # (dark slate gray) storage_tank
            [255, 215, 0],  # (gold) silo
            [192, 192, 192],  # (light grey) track
            [240, 230, 140],  # (khaki) wastewater_plant
            [154, 205, 50],  # (yellow green) power_substation
            [255, 165, 0],  # (orange) pier
            [0, 192, 0],  # (middle green) crop
            [0, 192, 0],  # (middle green) water_park
        ],
    },
    'wildfire': {
        'type': 'bin_segment',
        'categories': ['fire_retardant', 'burned'],
        'colors': [
            [255, 0, 0],  # (red) fire retardant
            [128, 128, 128],  # (grey) burned area
        ],
    },
    'smoke': {'type': 'classification', 'categories': ['no', 'partial', 'yes']},
    'snow': {'type': 'classification', 'categories': ['no', 'partial', 'yes']},
    'dem': {'type': 'regress', 'BackgroundInvalid': True},
    'airplane': {
        'type': 'detect',
        'categories': ['background', 'airplane'],
        'colors': [
            [0, 0, 0],  # (black) background
            [255, 0, 0],  # (red) airplane
        ],
    },
    'vessel': {
        'type': 'detect',
        'categories': ['background', 'vessel'],
        'colors': [
            [0, 0, 0],  # (black) background
            [255, 0, 0],  # (red) vessel
        ],
    },
    'water_event': {
        'type': 'segment',
        'BackgroundInvalid': True,
        'categories': ['invalid', 'background', 'water_event'],
        'colors': [
            [0, 0, 0],  # (black) invalid
            [0, 255, 0],  # (green) background
            [0, 0, 255],  # (blue) water_event
        ],
    },
    'park_sport': {
        'type': 'classification',
        'categories': [
            'american_football',
            'badminton',
            'baseball',
            'basketball',
            'cricket',
            'rugby',
            'soccer',
            'tennis',
            'volleyball',
        ],
    },
    'park_type': {
        'type': 'classification',
        'categories': ['park', 'pitch', 'golf_course', 'cemetery'],
    },
    'power_plant_type': {
        'type': 'classification',
        'categories': ['oil', 'nuclear', 'coal', 'gas'],
    },
    'quarry_resource': {
        'type': 'classification',
        'categories': ['sand', 'gravel', 'clay', 'coal', 'peat'],
    },
    'track_sport': {
        'type': 'classification',
        'categories': ['running', 'cycling', 'horse'],
    },
    'road_type': {
        'type': 'classification',
        'categories': [
            'motorway',
            'trunk',
            'primary',
            'secondary',
            'tertiary',
            'residential',
            'service',
            'track',
            'pedestrian',
        ],
    },
    'cloud': {
        'type': 'bin_segment',
        'categories': ['background', 'cloud', 'shadow'],
        'colors': [
            [0, 255, 0],  # (green) not clouds or shadows
            [255, 255, 255],  # (white) clouds
            [128, 128, 128],  # (grey) shadows
        ],
        'BackgroundInvalid': True,
    },
    'flood': {
        'type': 'bin_segment',
        'categories': ['background', 'water'],
        'colors': [
            [0, 255, 0],  # (green) background
            [0, 0, 255],  # (blue) water
        ],
        'BackgroundInvalid': True,
    },
}


class SatlasPretrain(NonGeoDataset):
    """SatlasPretrain dataset.

    `SatlasPretrain <https://satlas-pretrain.allen.ai/>`_ is a large-scale pre-training
    dataset for tasks that involve understanding satellite images. Regularly-updated
    satellite data is publicly available for much of the Earth through sources such as
    Sentinel-2 and NAIP, and can inform numerous applications from tackling illegal
    deforestation to monitoring marine infrastructure. However, developing automatic
    computer vision systems to parse these images requires a huge amount of manual
    labeling of training data. By combining over 30 TB of satellite images with 137
    label categories, SatlasPretrain serves as an effective pre-training dataset that
    greatly reduces the effort needed to develop robust models for downstream satellite
    image applications.

    Reference implementation:

    * https://github.com/allenai/satlas/blob/main/satlas/model/dataset.py

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.48550/arXiv.2211.15660

    .. versionadded:: 0.7

    .. note::
       This dataset requires the following additional library to be installed:

       * `AWS CLI <https://aws.amazon.com/cli/>`_: to download the dataset from AWS.
    """

    # https://github.com/allenai/satlas/blob/main/satlaspretrain_urls.txt
    url = 's3://ai2-public-datasets/satlas/'
    tarballs: ClassVar[dict[str, tuple[str, ...]]] = {
        'landsat': ('satlas-dataset-v1-landsat.tar',),
        'naip': (
            'satlas-dataset-v1-naip-2011.tar',
            'satlas-dataset-v1-naip-2012.tar',
            'satlas-dataset-v1-naip-2013.tar',
            'satlas-dataset-v1-naip-2014.tar',
            'satlas-dataset-v1-naip-2015.tar',
            'satlas-dataset-v1-naip-2016.tar',
            'satlas-dataset-v1-naip-2017.tar',
            'satlas-dataset-v1-naip-2018.tar',
            'satlas-dataset-v1-naip-2019.tar',
            'satlas-dataset-v1-naip-2020.tar',
        ),
        'sentinel1': ('satlas-dataset-v1-sentinel1-new.tar',),
        'sentinel2': (
            'satlas-dataset-v1-sentinel2-a.tar',
            'satlas-dataset-v1-sentinel2-b.tar',
        ),
        'static': ('satlas-dataset-v1-labels-static.tar',),
        'dynamic': ('satlas-dataset-v1-labels-dynamic.tar',),
        'metadata': ('satlas-dataset-v1-metadata.tar',),
    }
    md5s: ClassVar[dict[str, tuple[str, ...]]] = {
        'landsat': ('89ea5e8974826c071908392827780a06',),
        'naip': (
            '523736842994861054f04b97c4d90bfb',
            '636b9a3b08be0e40d098cb7b5e655b57',
            '69e2b1052b1d2d465322a24cf7207a16',
            '38999aea424d403ad60e1398443636aa',
            '97f4855072a8a406a4bfbe94c5f7311c',
            '9ba3c626b23e6d26749a323eaedc7c0a',
            'e4aba3d198dedfe1524a9338e85794aa',
            '74191a36d841b0b9b5d5cbae9a92ad71',
            '55b110cc6f734bf88793306d49f1c415',
            '97fc8414334987c59593d574f112a77e',
        ),
        'sentinel1': ('3d88a0a10df6ab0aa50db2ba4c475048',),
        'sentinel2': (
            '7e1c6a1e322807fb11df8c0c062545ca',
            '6636b8ecf2fff1d6723ecfef55a4876d',
        ),
        'static': ('4e38c2573bc78cf1f0d7267e432cb42c',),
        'dynamic': ('4503ae687948e7d2cb7ade0083f77a8a',),
        'metadata': ('6b9ac5a4f9a1ee88a271d28f12854607',),
    }

    # NOTE: 'tci' is RGB (b04-b02), not BGR (b02-b04)
    bands: ClassVar[dict[str, tuple[str, ...]]] = {
        'landsat': tuple(f'b{i}' for i in range(1, 12)),
        'naip': ('tci', 'ir'),
        'sentinel1': ('vh', 'vv'),
        'sentinel2': ('tci', 'b05', 'b06', 'b07', 'b08', 'b11', 'b12'),
    }

    chip_size = 512

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train_lowres',
        good_images: str = 'good_images_lowres_all',
        images: Iterable[str] = ('sentinel1', 'sentinel2', 'landsat'),
        labels: Iterable[str] = ('land_cover',),
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SatlasPretrain instance.

        Args:
            root: Root directory where dataset can be found.
            split: Metadata split to load.
            good_images: Metadata mapping between col/row and directory.
            images: List of image products.
            labels: List of label products.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If *images* is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(images) <= set(self.bands.keys())

        self.root = root
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.split = pd.read_json(os.path.join(root, 'metadata', f'{split}.json'))
        self.good_images = pd.read_json(
            os.path.join(root, 'metadata', f'{good_images}.json')
        )
        self.split.columns = ['col', 'row']
        self.good_images.columns = ['col', 'row', 'directory']
        self.good_images = self.good_images.groupby(['col', 'row'])

    def __len__(self) -> int:
        """Return the number of locations in the dataset.

        Returns:
            Length of the dataset
        """
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and label at that index.
        """
        col, row = self.split.iloc[index]
        directories = self.good_images.get_group((col, row))['directory']

        sample: dict[str, Tensor] = {}

        for image in self.images:
            self._load_image(sample, image, col, row, directories)

        for label in self.labels:
            self._load_label(sample, label, col, row)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(
        self,
        sample: dict[str, Tensor],
        image: str,
        col: int,
        row: int,
        directories: pd.Series,
    ) -> None:
        """Load a single image.

        Args:
            sample: Dataset sample to populate.
            image: Image product.
            col: Web Mercator column.
            row: Web Mercator row.
            directories: Directories that may contain the image.
        """
        # Moved in PIL 9.1.0
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR  # type: ignore[attr-defined]

        # Find directories that match image product
        good_directories: list[str] = []
        for directory in directories:
            path = os.path.join(self.root, image, directory)
            if os.path.isdir(path):
                good_directories.append(directory)

        # Choose a random timestamp
        idx = torch.randint(len(good_directories), (1,))
        directory = good_directories[idx]

        # Load all bands
        channels = []
        for band in self.bands[image]:
            path = os.path.join(self.root, image, directory, band, f'{col}_{row}.png')
            with Image.open(path) as img:
                img = img.resize((self.chip_size, self.chip_size), resample=resample)
                array = np.atleast_3d(np.array(img, dtype=np.float32))
                channels.append(torch.tensor(array))
        raster = rearrange(torch.cat(channels, dim=-1), 'h w c -> c h w')
        sample[f'image_{image}'] = raster

    def _load_label(
        self, sample: dict[str, Tensor], label: str, col: int, row: int
    ) -> None:
        """Load a single label.

        Args:
            sample: Dataset sample to populate.
            label: Label product.
            col: Web Mercator column.
            row: Web Mercator row.
        """
        path = os.path.join(self.root, 'static', f'{col}_{row}', f'{label}.png')
        if os.path.isfile(path):
            with Image.open(path) as img:
                raster = torch.tensor(np.array(img, dtype=np.int64))
        else:
            raster = torch.zeros(self.chip_size, self.chip_size, dtype=torch.long)
        sample[f'mask_{label}'] = raster

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        products = [*self.images, 'metadata']
        if self.labels:
            products.append('static')

        for product in products:
            # Check if the extracted directory already exists
            if os.path.isdir(os.path.join(self.root, product)):
                continue

            tarballs = self.tarballs[product]
            md5s = self.md5s[product]
            for tarball, md5 in zip(tarballs, md5s):
                path = os.path.join(self.root, tarball)

                # Check if the tarball has already been downloaded
                if os.path.isfile(path):
                    extract_archive(path)
                    continue

                # Check if the user requested to download the dataset
                if not self.download:
                    raise DatasetNotFoundError(self)

                # Download and extract the tarball
                aws = which('aws')
                aws('s3', 'cp', self.url + tarball, self.root)
                check_integrity(path, md5 if self.checksum else None)
                extract_archive(path)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        images = []
        titles = []
        for key, value in sample.items():
            match key.split('_', 1):
                case ['image', 'landsat']:
                    images.append(rearrange(value[[3, 2, 1]], 'c h w -> h w c') / 255)
                    titles.append('Landsat 8/9')
                case ['image', 'naip']:
                    images.append(rearrange(value[:3], 'c h w -> h w c') / 255)
                    titles.append('NAIP')
                case ['image', 'sentinel1']:
                    images.extend([value[0] / 255, value[1] / 255])
                    titles.extend(['Sentinel-1 VH', 'Sentinel-1 VV'])
                case ['image', 'sentinel2']:
                    images.append(rearrange(value[:3], 'c h w -> h w c') / 255)
                    titles.append('Sentinel-2')
                case ['mask' | 'prediction', label]:
                    cmap = torch.tensor(TASKS[label]['colors'])
                    images.append(cmap[value])
                    titles.append(label.replace('_', ' ').capitalize())

        fig, ax = plt.subplots(ncols=len(images), squeeze=False)
        for i, (image, title) in enumerate(zip(images, titles)):
            ax[0, i].imshow(image)
            ax[0, i].axis('off')

            if show_titles:
                ax[0, i].set_title(title)

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
