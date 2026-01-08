# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Earth Index Embeddings dataset."""

import einops
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .geo import VectorDataset
from .utils import Sample


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
