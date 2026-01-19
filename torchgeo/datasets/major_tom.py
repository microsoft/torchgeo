# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Major TOM datasets."""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.wkb
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class MajorTOMEmbeddings(NonGeoDataset):
    """Major TOM Embeddings dataset.

    `Major TOM <https://huggingface.co/Major-TOM>`__ (Terrestrial Observation Metaset)
    is a standard for curating, sharing and combining large-scale EO datasets. This
    data loader provides access to the official embedding datasets created using
    Major TOM Core and several existing foundation models.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2412.05600

    .. versionadded:: 0.9
    """

    def __init__(
        self, root: Path = 'data', transforms: Callable[[Sample], Sample] | None = None
    ) -> None:
        """Initialize a new MajorTOMEmbeddings instance.

        Args:
            root: Root directory where dataset parquet files can be found.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.transforms = transforms

        try:
            path = Path(root)
            # If it's a directory, let read_parquet handle it (it usually does), 
            # or you can manually glob if read_parquet fails on non-standard folder structures.
            self.data = gpd.read_parquet(path)
        except (FileNotFoundError, ValueError, OSError):
            # Fallback: manually find all .parquet files and concatenate them
            # This is useful if the folder structure isn't a clean PyArrow dataset
            if path.is_dir():
                files = sorted(path.glob('*.parquet'))
                if not files:
                    raise DatasetNotFoundError(self)
                # Read all and concat
                dfs = [gpd.read_parquet(f) for f in files]
                self.data = pd.concat(dfs, ignore_index=True)
            else:
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
            Data and label at that index. All items are Tensors.
        """
        row = self.data.iloc[index]
        t = pd.Timestamp(row['timestamp'])

        sample = {
            'embedding': torch.tensor(row['embedding']),
            'x': torch.tensor(row['centre_lon']),
            'y': torch.tensor(row['centre_lat']),
            't': torch.tensor(t.timestamp())
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
        
        # Plotting the embedding
        ax.plot(sample['embedding'])
        ax.set_xlabel('embedding dimension')

        if show_titles:
            x_raw = sample['x'].item()
            y_raw = sample['y'].item()
            t = pd.Timestamp.fromtimestamp(sample['t'].item())

            # Determine Cardinal Directions
            lat_dir = 'N' if y_raw >= 0 else 'S'
            lon_dir = 'E' if x_raw >= 0 else 'W'

            ax.set_title(f'{abs(y_raw):0.3f}°{lat_dir}, {abs(x_raw):0.3f}°{lon_dir}, {t}')

        fig.tight_layout()
        return fig
