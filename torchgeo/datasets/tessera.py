# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tessera embeddings dataset."""

import einops
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .geo import RasterDataset
from .utils import Sample


class TesseraEmbeddings(RasterDataset):
    """Tessera embeddings dataset.

    This is a data loader for geospatial embeddings from the `Tessera foundation model
    <https://github.com/ucam-eo/tessera>`__, which processes Sentinel-1 and Sentinel-2
    satellite imagery to generate 128-channel representation maps at 10m resolution.
    These embeddings compress a full year of temporal-spectral features into dense
    representations optimized for downstream geospatial analysis tasks.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2506.20380

    .. note::
       The dataset can be downloaded using the
       `geotessera <https://github.com/ucam-eo/geotessera>`__ library. Be sure to use
       ``--format tiff`` to download GeoTIFF files compatible with TorchGeo.

    .. versionadded:: 0.9
    """

    filename_glob = 'grid_*.tiff'
    filename_regex = """
        ^grid
        _(?P<lon>[0-9.-]+)
        _(?P<lat>[0-9.-]+)
        _(?P<date>[0-9]{4})
        .tiff$
    """
    date_format = '%Y'
    all_bands = tuple(map(str, range(128)))

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        .. warning::
           Visualizations are generated using PCA on each image *individually*, and
           are thus not comparable across images. The plot method is provided for
           visualization purposes only and should not be used to draw conclusions.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        _, h, w = sample['image'].shape
        A = einops.rearrange(sample['image'], 'c h w -> (h w) c')

        # Use PCA to project embeddings from 128D to 3D space
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
