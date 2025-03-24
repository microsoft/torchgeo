# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Flood-S1 dataset."""

import glob
import json
import os
import re
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import rasterio as rio
import torch
from matplotlib.colors import ListedColormap
from pyproj import Transformer
from torch import Tensor

from ..utils import Path, disambiguate_timestamp
from .base import CopernicusBenchBase


class CopernicusBenchFloodS1(CopernicusBenchBase):
    """Copernicus-Bench Flood-S1 dataset.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://arxiv.org/abs/2311.12056

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_flood_s1/flood_s1.zip'
    md5 = 'f4337fee5e90203c6d0c3efeb0b97b8a'
    zipfile = 'flood_s1.zip'
    directory = 'flood_s1'
    filename = 'grid_dict_{}.json'
    filename_regex = r'.{18}_(?P<date>\d{8})'
    date_format = '%Y%m%d'
    all_bands = ('VV', 'VH')
    rgb_bands = ('VV', 'VH')
    cmap = ListedColormap(['black', 'cyan', 'magenta'])
    classes = ('No Water', 'Permanent Waters', 'Floods')

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        mode: Literal[1, 2] = 1,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusBenchBase instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: Number of pre-flood images, 1 or 2.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.mode = mode
        self.bands = bands or self.all_bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        filepath = os.path.join(root, self.directory, self.filename.format(split))
        with open(filepath) as f:
            self.metadata = json.load(f)
        self.files = sorted(self.metadata.keys())

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        key = self.files[index]
        path = self.metadata[key]['path']
        directory = os.path.join(self.root, self.directory, 'data', path)
        mask_path = glob.glob(os.path.join(directory, 'MK0_MLU*.tif'))[0]
        sample = self._load_image(directory) | self._load_mask(mask_path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> dict[str, Tensor]:
        """Load an image and metadata.

        Args:
            path: File path to load.

        Returns:
            An image sample.
        """
        images = []
        times = []
        ptypes = ['SL1', 'MS1']
        if self.mode == 2:
            ptypes.insert(0, 'SL2')

        for ptype in ptypes:
            image = []
            for band in self.bands:
                # Band (every band)
                filepath = glob.glob(os.path.join(path, f'{ptype}_I{band}_*.tif'))[0]
                with rio.open(filepath) as f:
                    image.append(f.read(1).astype(np.float32))

            # Image (every ptype)
            images.append(image)

            # Time (every ptype)
            if match := re.match(self.filename_regex, os.path.basename(filepath)):
                if 'date' in match.groupdict():
                    date_str = match.group('date')
                    mint, maxt = disambiguate_timestamp(date_str, self.date_format)
                    time = (mint + maxt) / 2
                    times.append(time)

        # Location (only once)
        with rio.open(filepath) as f:
            x = (f.bounds.left + f.bounds.right) / 2
            y = (f.bounds.bottom + f.bounds.top) / 2
            transformer = Transformer.from_crs(f.crs, 'epsg:4326', always_xy=True)
            lon, lat = transformer.transform(x, y)

        return {
            'image': torch.tensor(np.array(images)),
            'lat': torch.tensor(lat),
            'lon': torch.tensor(lon),
            'time': torch.tensor(times),
        }
