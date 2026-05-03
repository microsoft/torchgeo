# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TimeSen2Crop dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url, extract_archive


class TimeSen2Crop(NonGeoDataset):
    """TimeSen2Crop dataset.

    The `TimeSen2Crop <https://zenodo.org/records/4715631>`__ dataset is a
    pixel-based dataset of more than one million Sentinel-2 time series labeled
    with 16 crop types. Imagery covers Austria during the agronomic year from
    September 2017 to August 2018, with one tile reacquired in 2019.

    Dataset features:

    * 1.13M single-pixel time series across 16 Sentinel-2 tiles
    * 9 spectral bands per acquisition (B2, B3, B4, B5, B6, B7, B8A, B11, B12)
    * variable number of acquisitions T per tile (each tile has its own dates)
    * pixel-quality flag per acquisition (clear, cloud, shadow, snow)

    Dataset format:

    * one ZIP archive on Zenodo
    * after extraction: ``Dataset/<tile>/<class>/<i>.csv`` plus a per-tile
      ``Dataset/<tile>/dates.csv``
    * each sample CSV is a (T, 10) matrix: 9 spectral bands plus a trailing
      condition column (0 clear, 1 cloud, 2 shadow, 3 snow)

    Dataset classes:

    0. Legumes
    1. Grassland
    2. Maize
    3. Potato
    4. Sunflower
    5. Soy
    6. Winter Barley
    7. Winter Caraway
    8. Rye
    9. Rapeseed
    10. Beet
    11. Spring Cereals
    12. Winter Wheat
    13. Winter Triticale
    14. Permanent Plantation
    15. Other Crops

    Note that ``33UVP`` and ``2019_33UVP`` cover the same MGRS tile in
    different years; treat them as one location when constructing
    geography-disjoint train/test splits.

    Variable-length sequences: each tile has its own number of acquisitions
    ``T``, so a :class:`~torch.utils.data.DataLoader` that draws from more
    than one tile cannot use the default collate (it calls ``torch.stack``,
    which requires identical shapes and raises
    ``RuntimeError: stack expects each tensor to be equal size``). Two
    options:

    1. Pass ``pad_to=<int>`` to right-pad every sample to a fixed length.
       Padded time steps are marked in ``mask`` with
       :attr:`PADDING_VALUE` so loss masking is easy.
    2. Pass a custom ``collate_fn`` to the DataLoader, for example using
       :func:`torch.nn.utils.rnn.pad_sequence`, which only pads to the
       longest sequence in each batch.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/JSTARS.2021.3073965

    .. versionadded:: 0.10
    """

    url = 'https://zenodo.org/records/4715631/files/TimeSen2Crop.zip'
    filename = 'TimeSen2Crop.zip'
    md5 = 'b5b7aad3fef192e78252e11c9a0e5cb8'
    extracted_dirname = 'TimeSen2Crop'
    cache_dirname = 'cache'

    all_bands = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12')
    rgb_bands = ('B4', 'B3', 'B2')

    classes = (
        'Legumes',
        'Grassland',
        'Maize',
        'Potato',
        'Sunflower',
        'Soy',
        'Winter Barley',
        'Winter Caraway',
        'Rye',
        'Rapeseed',
        'Beet',
        'Spring Cereals',
        'Winter Wheat',
        'Winter Triticale',
        'Permanent Plantation',
        'Other Crops',
    )

    valid_tiles: ClassVar[tuple[str, ...]] = (
        '2019_33UVP',
        '32TNT',
        '32TPT',
        '32TQT',
        '33TUM',
        '33TUN',
        '33TVM',
        '33TVN',
        '33TWM',
        '33TWN',
        '33TXN',
        '33UUP',
        '33UVP',
        '33UWP',
        '33UWQ',
        '33UXP',
    )

    #: Mask value used for time steps that fall in the right-padded region
    #: when ``pad_to`` is set, distinct from the four condition codes (0..3).
    PADDING_VALUE: ClassVar[int] = 4

    def __init__(
        self,
        root: Path = 'data',
        tiles: Sequence[str] = valid_tiles,
        bands: Sequence[str] = all_bands,
        pad_to: int | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new TimeSen2Crop dataset instance.

        Args:
            root: root directory where dataset can be found.
            tiles: subset of Sentinel-2 tiles to load. Each tile has its own
                number of acquisitions; mixing tiles requires ``pad_to`` to use
                the default DataLoader collation.
            bands: subset of spectral bands to load, in output order.
            pad_to: if set, right-pad the time axis with zeros to this length
                and mark padded steps in the mask with ``PADDING_VALUE``. Must
                be at least as large as the longest time series among the
                selected tiles.
            transforms: a function/transform that takes an input sample and
                returns a transformed version.
            download: if True, download dataset and store it in the root
                directory.
            checksum: if True, check the MD5 of the downloaded archive (slow).

        Raises:
            AssertionError: If a tile or band name is not recognized, or if
                ``pad_to`` is smaller than the longest selected tile.
            DatasetNotFoundError: If dataset is not found and ``download`` is
                False.
        """
        for tile in tiles:
            assert tile in self.valid_tiles, (
                f'Invalid tile {tile!r}. Valid tiles: {list(self.valid_tiles)}.'
            )
        for band in bands:
            assert band in self.all_bands, (
                f'Invalid band {band!r}. Valid bands: {list(self.all_bands)}.'
            )

        self.root = root
        self.tiles = tuple(tiles)
        self.bands = tuple(bands)
        self.band_indices = [self.all_bands.index(b) for b in self.bands]
        self.pad_to = pad_to
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.tile_dates: dict[str, list[str]] = {}
        self.index: list[tuple[str, int, int]] = []
        self._tile_T: dict[str, int] = {}
        for tile in self.tiles:
            cache_path = os.path.join(self.root, self.cache_dirname, f'{tile}.npz')
            with np.load(cache_path, allow_pickle=False) as f:
                dates = f['dates'].astype(str)
                self.tile_dates[tile] = list(dates)
                self._tile_T[tile] = int(dates.shape[0])
                for class_id in range(len(self.classes)):
                    count_key = f'count_{class_id}'
                    if count_key not in f.files:
                        continue
                    n = int(f[count_key])
                    for i in range(n):
                        self.index.append((tile, class_id, i))

        if self.pad_to is not None:
            max_T = max(self._tile_T[t] for t in self.tiles)
            assert self.pad_to >= max_T, (
                f'pad_to={self.pad_to} is smaller than the longest selected '
                f'tile time series ({max_T}).'
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.index)

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at the given flat index.

        Args:
            index: index of the sample.

        Returns:
            Dictionary with ``image`` (T, B), ``mask`` (T,), and ``label`` ().
        """
        tile, class_id, sample_idx = self.index[index]
        cache_path = os.path.join(self.root, self.cache_dirname, f'{tile}.npz')
        with np.load(cache_path, allow_pickle=False) as f:
            bands = f[f'bands_{class_id}'][sample_idx]
            mask = f[f'mask_{class_id}'][sample_idx]

        bands = bands[:, self.band_indices]
        T = bands.shape[0]

        if self.pad_to is not None and T < self.pad_to:
            pad = self.pad_to - T
            bands = np.concatenate(
                [bands, np.zeros((pad, bands.shape[1]), dtype=bands.dtype)], axis=0
            )
            mask = np.concatenate(
                [mask, np.full((pad,), self.PADDING_VALUE, dtype=mask.dtype)], axis=0
            )

        sample: Sample = {
            'image': torch.from_numpy(bands.astype(np.float32, copy=False)),
            'mask': torch.from_numpy(mask.astype(np.int64, copy=False)),
            'label': torch.tensor(class_id, dtype=torch.long),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset, building the cache if needed."""
        cache_dir = os.path.join(self.root, self.cache_dirname)
        if all(
            os.path.exists(os.path.join(cache_dir, f'{tile}.npz'))
            for tile in self.tiles
        ):
            return

        extracted = os.path.join(self.root, self.extracted_dirname)
        if os.path.isdir(extracted):
            self._build_cache()
            return

        zip_path = os.path.join(self.root, self.filename)
        if os.path.exists(zip_path):
            extract_archive(zip_path)
            self._build_cache()
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()
        extract_archive(os.path.join(self.root, self.filename))
        self._build_cache()

    def _download(self) -> None:
        """Download the dataset archive."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _build_cache(self) -> None:
        """Stack each (tile, class)'s CSVs into a per-tile npz cache.

        Within a tile, every sample shares the same number of acquisitions
        (the tile's own ``dates.csv``), so all samples of one class can be
        stacked into a single (N, T, 10) array. The cache turns ~1.1M small
        file reads at training time into one mmap-backed slice.
        """
        cache_dir = os.path.join(self.root, self.cache_dirname)
        os.makedirs(cache_dir, exist_ok=True)

        for tile in self.valid_tiles:
            tile_dir = os.path.join(self.root, self.extracted_dirname, tile)
            if not os.path.isdir(tile_dir):
                continue

            cache_path = os.path.join(cache_dir, f'{tile}.npz')
            if os.path.exists(cache_path):
                continue

            dates_path = os.path.join(tile_dir, 'dates.csv')
            with open(dates_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            # Drop a leading header row (e.g. ``acquisition_date``) if present.
            if lines and not lines[0].isdigit():
                lines = lines[1:]
            arrays: dict[str, np.ndarray] = {'dates': np.asarray(lines)}

            for entry in os.scandir(tile_dir):
                # Class subfolders are integer-named (``0`` .. ``15``).
                if not (entry.is_dir() and entry.name.isdigit()):
                    continue
                class_id = int(entry.name)
                if class_id >= len(self.classes):
                    continue

                csv_paths = sorted(
                    p.path for p in os.scandir(entry.path) if p.name.endswith('.csv')
                )
                if not csv_paths:
                    # Some (tile, class) cells legitimately have zero samples.
                    continue

                bands_stack: list[np.ndarray] = []
                mask_stack: list[np.ndarray] = []
                for csv_path in csv_paths:
                    # Per-sample CSVs include a header row.
                    arr = np.loadtxt(
                        csv_path, delimiter=',', dtype=np.float32, skiprows=1
                    )
                    bands_stack.append(arr[:, : len(self.all_bands)])
                    mask_stack.append(arr[:, len(self.all_bands)].astype(np.uint8))

                arrays[f'bands_{class_id}'] = np.stack(bands_stack, axis=0)
                arrays[f'mask_{class_id}'] = np.stack(mask_stack, axis=0)
                arrays[f'count_{class_id}'] = np.asarray(len(csv_paths))

            # ty narrows **kwargs against the typed `allow_pickle` parameter,
            # which is unrelated to the array kwargs we actually pass.
            np.savez(cache_path, **arrays)  # ty: ignore[invalid-argument-type]

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample as a temporal RGB strip.

        Args:
            sample: a sample returned by :meth:`__getitem__`.
            show_titles: whether to draw the class label as a title.
            suptitle: optional suptitle.

        Returns:
            A matplotlib Figure.

        Raises:
            RGBBandsMissingError: if any of B4, B3, B2 are missing from
                ``self.bands``.
        """
        try:
            rgb_indices = [self.bands.index(b) for b in self.rgb_bands]
        except ValueError as exc:
            raise RGBBandsMissingError() from exc

        image = sample['image'].numpy()
        rgb = np.clip(image[:, rgb_indices] / 3000.0, 0, 1)

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.imshow(rgb[None, ...], aspect='auto')
        ax.set_yticks([])
        ax.set_xlabel('Acquisition step')

        if show_titles:
            label_idx = int(sample['label'])
            ax.set_title(f'Crop: {self.classes[label_idx]}')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
