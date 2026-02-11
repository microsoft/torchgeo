# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""EarthEmbeddings datasets."""

from collections.abc import Callable

import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class EarthEmbeddings(NonGeoDataset):
    """EarthEmbeddings dataset.

    `EarthEmbeddings <https://huggingface.co/datasets/ML4Sustain/EarthEmbeddings>`__
    are pre-computed embeddings of uniformly sampled MajorTOM-Core-S2L2A imagery
    using SatCLIP, FarSLIP, DINOv2, SigLIP models. These embeddings power the
    `EarthEmbeddingExplorer <https://huggingface.co/spaces/ML4RS-Anonymous/EarthEmbeddingExplorer>`__
    application, which allows users to search for satellite images using text queries,
    image uploads, or geographic locations.

    If you use this dataset in your research, please cite the following paper:

    * A tutorial paper to be uploaded to arXiv soon.

    .. versionadded:: 0.9
    """

    def __init__(
        self, root: Path = 'data', transforms: Callable[[Sample], Sample] | None = None
    ) -> None:
        """Initialize a new EarthEmbeddings instance.

        Args:
            root: Root directory where dataset can be found.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.transforms = transforms

        try:
            self.data = pd.read_parquet(root)
        except (FileNotFoundError, ValueError):
            raise DatasetNotFoundError(self)

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and label at that index.
        """
        row = self.data.iloc[index]
        t = pd.Timestamp(row['timestamp'])

        sample = {
            'embedding': torch.tensor(row['embedding']),
            'x': torch.tensor(row['centre_lon']),
            'y': torch.tensor(row['centre_lat']),
            't': torch.tensor(t.timestamp()),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(self, sample: Sample, show_titles: bool = True) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        fig, ax = plt.subplots()
        ax.plot(sample['embedding'])

        if show_titles:
            x = sample['x'].item()
            y = sample['y'].item()
            t = pd.Timestamp.fromtimestamp(sample['t'].item())
            ax.set_title(rf'{y:0.3f}°N, {x:0.3f}°W, {t}')

        fig.tight_layout()
        return fig
