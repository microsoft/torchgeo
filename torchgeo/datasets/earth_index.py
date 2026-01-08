# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Earth Index Embeddings dataset."""

import einops
import geopandas as gpd
import pandas as pd
import rasterio
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .geo import VectorDataset
from .utils import GeoSlice, Sample, lazy_import


class EarthIndexEmbeddings(VectorDataset):
    """Earth Index Embeddings dataset.

    `Earth Index Embeddings <https://source.coop/earthgenome/earthindexembeddings>`__
    are a global embedding product generated from Earth Index v2 Sentinel-2 mosaics. The
    embeddings are generated using the `SoftCon <https://github.com/zhu-xlab/softcon>`__
    model from `Zhu XLabs <https://www.asg.ed.tum.de/sipeo/home/>`__ and result in an
    embedding of length 384. Each embedding captures a 320 square meter patch of the
    Earth, gridded using a MajorTom-based grid. These embeddings, their IDs and
    centroids are encoded in geoparquet. The GeoParquet is named similarly to the
    imagery and references the original MGRS/UTM tile which the imagery covered.

    .. note::
       This dataset requires the following additional library to be installed:

       * `geocube <https://pypi.org/project/geocube/>`_: to rasterize the dataset.

    .. versionadded:: 0.9
    """

    filename_regex = r"""
        ^(?P<mgrs>\d{2}[A-Z]{3})
        _(?P<start>\d{4}-\d{2}-\d{2})
        _(?P<stop>\d{4}-\d{2}-\d{2})
        \.
    """
    date_format = '%Y-%m-%d'
    is_image = True

    def __getitem__(self, query: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
        x, y, t = self._disambiguate_slice(query)
        interval = pd.Interval(t.start, t.stop)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.iloc[:: t.step]
        index = index.cx[x.start : x.stop, y.start : y.stop]

        if index.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        df = pd.concat([gpd.read_parquet(f) for f in index.filepath])
        geocube = lazy_import('geocube.api.core')
        ds = geocube.make_geocube(
            df, measurements=['embedding'], output_crs=self.crs, resolution=self.res
        )

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample: Sample = {
            'bounds': self._slice_to_tensor(query),
            'image': torch.from_numpy(ds['embedding'].values),
            'transform': torch.tensor(transform),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        .. warning::
           Visualizations are generated using PCA on each image *individually*, and
           are thus not comparable across images. The plot method is provided for
           visualization purposes only and should not be used to draw conclusions.

        Args:
            sample: a sample returned by :meth:`VectorDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        _, h, w = sample['image'].shape
        A = einops.rearrange(sample['image'], 'c h w -> (h w) c')

        # Use PCA to project embeddings from 384D to 3D space
        _, _, V = torch.pca_lowrank(A, q=3)
        B = A @ V

        B -= B.min(dim=0, keepdim=True)[0]
        B /= B.max(dim=0, keepdim=True)[0]
        image = einops.rearrange(B, '(h w) c -> h w c', h=h, w=w)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title('Embedding')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
