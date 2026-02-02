# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Clay Embeddings dataset."""

from collections.abc import Callable

import geopandas as gpd
import pandas as pd
import shapely
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class ClayEmbeddings(NonGeoDataset):
    """Clay Embeddings dataset.

    Supports both:

    * `Clay v0 embeddings <https://source.coop/clay/clay-model-v0-embeddings>`_
    * `Clay v1.5 embeddings <https://source.coop/clay/lgnd-clay-v1-5-sentinel-2-l2a>`_

    See https://clay-foundation.github.io/model/ for details.

    .. versionadded:: 0.9
    """

    def __init__(
        self, root: Path = 'data', transforms: Callable[[Sample], Sample] | None = None
    ) -> None:
        """Initialize a new ClayEmbeddings instance.

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
            self.data = gpd.read_parquet(root)
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
        centroid = shapely.centroid(row['geometry'])
        key = 'date' if 'date' in row else 'datetime'
        date = pd.Timestamp(row[key])
        key = 'embedding' if 'embedding' in row else 'embeddings'

        sample = {
            'embedding': torch.tensor(row[key]),
            'x': torch.tensor(centroid.x),
            'y': torch.tensor(centroid.y),
            't': torch.tensor(date.timestamp()),
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
