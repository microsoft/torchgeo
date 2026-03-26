# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""BioMassters datamodule."""

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
from matplotlib.figure import Figure
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import BioMassters
from ..datasets.geo import NonGeoDataset
from ..datasets.utils import Sample, pad_across_batches
from .geo import NonGeoDataModule


class _BioMasstersSpatioTemporalRegressionDataset(NonGeoDataset):
    """Dataset adapter for spatiotemporal pixelwise regression.

    The underlying :class:`~torchgeo.datasets.BioMassters` dataset returns
    sensor-specific keys such as ``image_S1`` and ``image_S2`` and uses ``label``
    for the AGB target. That interface is convenient for the dataset itself, but it
    does not match the batch contract expected by spatiotemporal pixelwise regression
    trainers, which operate on a single fused ``image`` time series, a per-pixel
    ``mask`` target, and optional sequence ``length`` metadata.

    This wrapper is therefore responsible for:

    * fusing the selected sensors into one ``(T, C, H, W)`` tensor;
    * inserting zero-filled channels for missing sensor acquisitions so the channel
      layout stays consistent across timesteps; and
    * renaming the regression target to ``mask`` while preserving the true sequence
      length for padded batching.
    """

    channel_counts: ClassVar[dict[str, int]] = {'S1': 4, 'S2': 11}

    def __init__(
        self,
        root: str = 'data',
        split: Literal['train', 'test'] = 'train',
        sensors: Sequence[Literal['S1', 'S2']] = ('S1', 'S2'),
        download: bool = False,
    ) -> None:
        """Initialize a new dataset adapter instance.

        Args:
            root: Root directory where the dataset can be found.
            split: Dataset split to use.
            sensors: Sensors to include in the fused time series.
            download: Unused placeholder for datamodule compatibility.
        """
        del download

        self.root = root
        self.split = split
        self.sensors = tuple(sensors)
        self.dataset = BioMassters(
            root=root,
            split=split,
            sensors=self.sensors,
            as_time_series=True,
        )
        self.sample_groups = [
            sample_df.copy()
            for _, sample_df in self.dataset.df.groupby('num_index', sort=True)
        ]

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Sample dictionary with fused image, mask, and length.
        """
        sample_df = self.sample_groups[index]
        months = sorted(sample_df['num_month'].unique())

        frames = []
        for month in months:
            month_df = sample_df[sample_df['num_month'] == month]
            sensor_images: dict[str, Tensor] = {}
            for sensor in self.sensors:
                sensor_df = month_df[month_df['satellite'] == sensor]
                if sensor_df.empty:
                    continue

                image = self.dataset._load_input(sensor_df['filename'].tolist())
                sensor_images[sensor] = image.squeeze(0)

            reference = next(iter(sensor_images.values()))
            _, height, width = reference.shape

            channels = []
            for sensor in self.sensors:
                if sensor in sensor_images:
                    channels.append(sensor_images[sensor])
                else:
                    channels.append(
                        torch.zeros(
                            self.channel_counts[sensor],
                            height,
                            width,
                            dtype=reference.dtype,
                        )
                    )

            frames.append(torch.cat(channels, dim=0))

        sample: Sample = {
            'image': torch.stack(frames, dim=0),
            'length': torch.tensor(len(frames), dtype=torch.long),
        }
        if self.split == 'train':
            sample['mask'] = self.dataset._load_target(
                sample_df['corresponding_agbm'].iloc[0]
            )

        return sample

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.sample_groups)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset."""
        image = sample['image']
        if image.ndim != 4:
            msg = 'Expected image tensor with shape (T, C, H, W).'
            raise ValueError(msg)

        plot_sample: Sample = {}
        channel_start = 0
        for sensor in self.sensors:
            channel_end = channel_start + self.channel_counts[sensor]
            plot_sample[f'image_{sensor}'] = image[:, channel_start:channel_end]
            channel_start = channel_end

        if 'mask' in sample:
            plot_sample['label'] = sample['mask']
        if 'prediction' in sample:
            plot_sample['prediction'] = sample['prediction']

        return self.dataset.plot(
            plot_sample, show_titles=show_titles, suptitle=suptitle
        )


def _pad_regression_time_series(
    batch: Sequence[Sample], padding_length: int, padding_value: float = 0.0
) -> Sample:
    """Collate variable-length regression time series and preserve lengths."""
    collated = pad_across_batches(
        batch, padding_length=padding_length, padding_value=padding_value
    )
    collated['length'] = torch.tensor(
        [min(int(sample['length']), padding_length) for sample in batch],
        dtype=torch.long,
    )
    return collated


class BioMasstersDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BioMassters dataset.

    This datamodule adapts BioMassters to a fused spatiotemporal regression format
    with samples shaped like ``{'image': (T, C, H, W), 'mask': (1, H, W)}``.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        padding_length: int = 12,
        sensors: Sequence[Literal['S1', 'S2']] = ('S1', 'S2'),
        **kwargs: Any,
    ) -> None:
        """Initialize a new BioMasstersDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the labeled train split used for validation.
            test_split_pct: Percentage of the labeled train split used for testing.
            padding_length: Padding length of the time series.
            sensors: Sensors to include in the fused time series.
            **kwargs: Additional keyword arguments passed to the dataset adapter.
        """
        super().__init__(
            _BioMasstersSpatioTemporalRegressionDataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sensors=sensors,
            **kwargs,
        )
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.padding_length = padding_length
        self.collate_fn = lambda batch: _pad_regression_time_series(
            batch, padding_length=self.padding_length
        )
        self.aug = lambda batch: batch

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate', 'test']:
            self.dataset = self.dataset_class(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [
                    1 - self.val_split_pct - self.test_split_pct,
                    self.val_split_pct,
                    self.test_split_pct,
                ],
                generator,
            )

        if stage in ['predict']:
            self.predict_dataset = self.dataset_class(split='test', **self.kwargs)
