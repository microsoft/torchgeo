# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Pretrain dataset."""

import io
import os
import random
import re
import tarfile
from collections.abc import Iterator, Sequence
from typing import ClassVar

import requests
import torch
import torch.distributed as dist
import torch.utils.data
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import IterableDataset

from ..utils import Sample, quantile_normalization


def _expand_urls(urls: str | Sequence[str]) -> list[str]:
    """Expand brace notation into a list of shard paths or URLs.

    Args:
        urls: One or more shard paths or URLs, optionally containing numeric
            brace notation such as ``example-{000000..000009}.tar``.

    Returns:
        The expanded list of shard paths or URLs.
    """
    if not isinstance(urls, str):
        return [expanded for url in urls for expanded in _expand_urls(url)]

    match = re.search(r'\{(\d+)\.\.(\d+)\}', urls)
    if match is None:
        return [urls]

    start, stop = match.group(1), match.group(2)
    expanded = []
    for i in range(int(start), int(stop) + 1):
        shard = urls[: match.start()] + str(i).zfill(len(start)) + urls[match.end() :]
        expanded.extend(_expand_urls(shard))
    return expanded


class CopernicusPretrain(IterableDataset[Sample]):
    """Copernicus-Pretrain dataset.

    Copernicus-Pretrain is an extension of the SSL4EO-S12 dataset to all major Sentinel
    missions (S1-S5P). The images are organized into ~310K regional grids (0.25°x0.25°,
    consistent with ERA5), densely covering the whole land surface and near-land ocean
    with time series from eight distinct Sentinel modalities.

    This dataset streams samples directly from its sharded tar archives, which can be
    local files or remote http(s) URLs. It is a PyTorch IterableDataset that is
    compatible with :class:`torch.utils.data.DataLoader`, and shards are automatically
    split across distributed ranks and DataLoader workers.

    The full dataset has a varying number of modalities, S1/2 local patches, and
    timestamps for different grids. It also contains metadata including the filenames
    all images are derived from. For simplicity, the current dataset class provides
    a minimum example:

    - only use grids with all modalities (220k)
    - sample one local patch for S1 and S2
    - sample one timestamp for each modality

    Therefore, each sample contains 8 tensors (S1, S2, S3, S5P NO2/CO/SO2/O3, DEM).

    Example:

    .. code-block:: python

       dataset = CopernicusPretrain(
           urls='data/example-{000000..000009}.tar', shardshuffle=True, resampled=True
       )

       # Check the first sample
       sample = next(iter(dataset))
       s1 = sample['s1_grd.pth']
       s2 = sample['s2_toa.pth']
       s3 = sample['s3_olci.pth']
       s5p_co = sample['s5p_co.pth']
       s5p_no2 = sample['s5p_no2.pth']
       s5p_o3 = sample['s5p_o3.pth']
       s5p_so2 = sample['s5p_so2.pth']
       dem = sample['dem.pth']

       # Create a DataLoader for distributed training on 2 GPUs
       dataloader = DataLoader(dataset, batch_size=10, num_workers=2)
       # A resampled dataset is infinite size, so limit the epoch length
       # Total number of samples / (batch size * world size)
       number_of_batches = 1000 // (10 * 2)

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.7

    .. versionchanged:: 0.10
       *webdataset* is no longer required to load this dataset. The constructor now
       accepts explicit *urls*, *shardshuffle*, *resampled*, and *shuffle_buffer*
       arguments instead of forwarding all arguments to ``webdataset.WebDataset``.
    """

    url_dict: ClassVar[dict[str, str]] = {
        # grids with all modalities
        '220k_aligned': 'https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/ssl4eo_s_220k_aligned/example-{000000..002255}.tar',
        # remaining grids (with at least one modality)
        '220k_310k_union': 'https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/ssl4eo_s_220k_310k_union/example-{002256..003210}.tar',
        # 100 example grids
        '100_example': 'https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/example_100_grids/example_100_webdataset/example-{000000..000009}.tar',
    }

    def __init__(
        self,
        urls: str | Sequence[str],
        shardshuffle: bool = False,
        resampled: bool = False,
        shuffle_buffer: int = 10,
    ) -> None:
        """Initialize a new CopernicusPretrain instance.

        Args:
            urls: One or more shard paths or http(s) URLs, optionally containing
                numeric brace notation such as ``example-{000000..000009}.tar``.
            shardshuffle: Shuffle the order of shards each epoch.
            resampled: Yield an infinite stream of samples by sampling shards
                with replacement instead of iterating over each shard once.
            shuffle_buffer: Size of the buffer used to shuffle samples.
        """
        self.urls = _expand_urls(urls)
        self.shardshuffle = shardshuffle
        self.resampled = resampled
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over images in the dataset.

        Yields:
            sample of images

        .. versionchanged:: 0.10
           Removed *json* metadata.
        """
        samples = (
            sample
            for shard in self._iter_shards()
            for sample in self._iter_samples(shard)
        )
        for sample in self._shuffle_samples(samples):
            if self._has_all_modalities(sample):
                sample = self._sample_one_local_patch(sample)
                sample = self._sample_one_time_stamp(sample)
                yield sample

    def _iter_shards(self) -> Iterator[str]:
        """Yield the shards that this rank and DataLoader worker should read.

        If *resampled*, shards are sampled indefinitely with replacement.
        Otherwise, shards are split across distributed ranks and DataLoader
        workers, and each assigned shard is yielded once.

        Yields:
            Shard paths or URLs.
        """
        shards = self.urls
        if self.resampled:
            while True:
                yield random.choice(shards)
        else:
            if dist.is_available() and dist.is_initialized():
                shards = shards[dist.get_rank() :: dist.get_world_size()]
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                shards = shards[worker_info.id :: worker_info.num_workers]
            if self.shardshuffle:
                shards = random.sample(shards, k=len(shards))
            yield from shards

    def _open_shard(self, shard: str) -> tarfile.TarFile:
        """Open a local or remote tar shard for sequential streaming.

        Args:
            shard: Local path or http(s) URL of a tar shard.

        Returns:
            The tar archive opened in streaming mode.
        """
        if shard.startswith(('http://', 'https://')):
            response = requests.get(shard, stream=True, timeout=30)
            response.raise_for_status()
            return tarfile.open(fileobj=response.raw, mode='r|*')
        return tarfile.open(shard, mode='r|*')

    def _iter_samples(self, shard: str) -> Iterator[Sample]:
        """Sequentially read samples from a single tar shard.

        Files are grouped into samples by their prefix up to the first dot,
        and ``.pth`` entries are decoded into tensors. All other entries,
        including *json* metadata, are skipped.

        Args:
            shard: Local path or http(s) URL of a tar shard.

        Yields:
            sample of images
        """
        with self._open_shard(shard) as tar:
            key = None
            sample: Sample = {}
            for member in (m for m in tar if m.isfile()):
                prefix, _, field = os.path.basename(member.name).partition('.')
                if prefix != key:
                    if sample:
                        yield sample
                    key = prefix
                    sample = {}
                if field.endswith('.pth'):
                    data = tar.extractfile(member)
                    assert data is not None
                    sample[field] = torch.load(
                        io.BytesIO(data.read()), weights_only=True
                    )
            if sample:
                yield sample

    def _shuffle_samples(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        """Shuffle a stream of samples using a fixed-size buffer.

        Args:
            samples: Stream of samples to shuffle.

        Yields:
            sample of images
        """
        buffer: list[Sample] = []
        for sample in samples:
            buffer.append(sample)
            if len(buffer) >= self.shuffle_buffer:
                yield buffer.pop(random.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(random.randrange(len(buffer)))

    def _has_all_modalities(self, sample: Sample) -> bool:
        """Selection function: filter samples with all required modalities.

        Args:
            sample: A single sample from the dataset.

        Returns:
            True if all modalities are present in the sample, else False.
        """
        required_keys = [
            's1_grd.pth',
            's2_toa.pth',
            's3_olci.pth',
            's5p_co.pth',
            's5p_no2.pth',
            's5p_o3.pth',
            's5p_so2.pth',
            'dem.pth',
        ]
        return all(key in sample for key in required_keys)

    def _sample_one_local_patch(self, sample: Sample) -> Sample:
        """Mapping function: randomly select one local patch for S1 and S2.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only a single patch for S1 and S2.
        """
        s1, s2 = sample['s1_grd.pth'], sample['s2_toa.pth']
        idx = random.randint(0, s1.shape[0] - 1)
        sample['s1_grd.pth'], sample['s2_toa.pth'] = s1[idx], s2[idx]
        return sample

    def _sample_one_time_stamp(self, sample: Sample) -> Sample:
        """Mapping function: randomly select one timestamp for all modalities.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only a single timestamp.
        """
        for key in sample:
            if key.endswith('.pth') and key != 'dem.pth':
                idx = random.randint(0, sample[key].shape[0] - 1)
                sample[key] = sample[key][idx]

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__iter__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        fig, ax = plt.subplots(nrows=2, ncols=4)

        image = sample['s1_grd.pth']
        vv = image[0]
        vh = image[1]
        image = torch.stack([vv, vh, (vv + vh) / 2], dim=-1)
        image = quantile_normalization(image)
        ax[0, 0].imshow(image)
        ax[0, 0].axis('off')

        rgb_bands = [3, 2, 1]
        image = sample['s2_toa.pth'][rgb_bands].float()
        image = rearrange(image, 'c h w -> h w c')
        image = quantile_normalization(image)
        ax[0, 1].imshow(image)
        ax[0, 1].axis('off')

        rgb_bands = [7, 5, 3]
        image = sample['s3_olci.pth'][rgb_bands]
        image = rearrange(image, 'c h w -> h w c')
        image = quantile_normalization(image)
        ax[0, 2].imshow(image)
        ax[0, 2].axis('off')

        image = sample['dem.pth']
        ax[0, 3].imshow(image, cmap='terrain')
        ax[0, 3].axis('off')

        image = sample['s5p_co.pth'][0]
        ax[1, 0].imshow(image, cmap='Wistia')
        ax[1, 0].axis('off')

        image = sample['s5p_no2.pth'][0]
        ax[1, 1].imshow(image, cmap='Wistia')
        ax[1, 1].axis('off')

        image = sample['s5p_o3.pth'][0]
        ax[1, 2].imshow(image, cmap='Wistia')
        ax[1, 2].axis('off')

        image = sample['s5p_so2.pth'][0]
        ax[1, 3].imshow(image, cmap='Wistia')
        ax[1, 3].axis('off')

        if show_titles:
            ax[0, 0].set_title('S1 GRD')
            ax[0, 1].set_title('S2 TOA')
            ax[0, 2].set_title('S3 OLCI')
            ax[0, 3].set_title('DEM')
            ax[1, 0].set_title('S5P CO')
            ax[1, 1].set_title('S5P NO$_2$')
            ax[1, 2].set_title('S5P O$_3$')
            ax[1, 3].set_title('S5P SO$_2$')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        return fig
