# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Earth Index Embeddings dataset."""

from collections.abc import Callable

import geopandas as gpd
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class EarthIndexEmbeddings(NonGeoDataset):
    """Earth Index Embeddings dataset.

    `Earth Index Embeddings <https://source.coop/earthgenome/earthindexembeddings>`__
    are a global embedding product generated from Earth Index v2 Sentinel-2 mosaics. The
    embeddings are generated using the `SoftCon <https://github.com/zhu-xlab/softcon>`__
    model from `Zhu XLabs <https://www.asg.ed.tum.de/sipeo/home/>`__ and result in an
    embedding of length 384. Each embedding captures a 320 square meter patch of the
    Earth, gridded using a MajorTom-based grid. These embeddings, their IDs and
    centroids are encoded in geoparquet. The GeoParquet is named similarly to the
    imagery and references the original MGRS/UTM tile which the imagery covered.

    .. versionadded:: 0.9
    """

    def __init__(
        self, root: Path = 'data', transforms: Callable[[Sample], Sample] | None = None
    ) -> None:
        """Initialize a new EarthIndexEmbeddings instance.

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

        sample = {
            'embedding': torch.tensor(row['embedding']),
            'x': torch.tensor(row['geometry'].x),
            'y': torch.tensor(row['geometry'].y),
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
            ax.set_title(rf'{y:0.3f}°N, {x:0.3f}°W')

        fig.tight_layout()
        return fig
