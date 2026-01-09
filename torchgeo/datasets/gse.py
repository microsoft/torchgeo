# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Google Satellite Embedding dataset."""

import pathlib
from datetime import datetime

import einops
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .geo import RasterDataset
from .utils import Path, Sample, disambiguate_timestamp


class GoogleSatelliteEmbedding(RasterDataset):
    """Google Satellite Embedding dataset.

    The `Google Satellite Embedding dataset
    <https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL>`__
    is a global, analysis-ready collection of learned geospatial `embeddings
    <https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space>`__.
    Each 10-meter pixel in this dataset is a 64-dimensional representation,
    or "`embedding vector <https://developers.google.com/machine-learning/glossary#embedding-vector>`__",
    that encodes temporal trajectories of surface conditions at and around that pixel
    as measured by various Earth observation instruments and datasets, over a single
    calendar year.

    The dataset covers terrestrial land surfaces and shallow waters, including
    intertidal and reef zones, inland waterways, and coastal waterways.
    Coverage at the poles is limited by satellite orbits and instrument coverage.

    The embeddings are unit-length, meaning they have a magnitude of 1 and do not
    require any additional normalization, and are distributed across the unit sphere,
    making them well-suited for use with clustering algorithms and tree-based
    classifiers. The embedding space is also consistent across years, and embeddings
    from different years can be used for condition change detection by considering the
    dot product or angle between two embedding vectors. Furthermore, the embeddings
    are designed to be linearly composable, i.e., they can be aggregated to produce
    embeddings at coarser spatial resolutions or transformed with vector arithmetic,
    and still retain their semantic meaning and distance relationships.

    The Satellite Embedding dataset was produced by `AlphaEarth Foundations
    <https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/>`__,
    a geospatial embedding model that assimilates multiple datastreams including
    optical, radar, LiDAR, and other sources.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2507.22291

    .. note::
       The dataset can be downloaded from a number of locations:

       * `Google Cloud Storage <https://console.cloud.google.com/storage/browser/alphaearth_foundations>`__: 2017--2024, requires a billing project
       * `Source Cooperative <https://source.coop/tge-labs/aef>`__: 2018--2024
       * `Hugging Face <https://huggingface.co/datasets/Major-TOM/Core-AlphaEarth-Embeddings>`__: subset matching Major TOM

    .. versionadded:: 0.9
    """

    # https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL#bands
    all_bands = tuple(f'A{n:02}' for n in range(64))

    def _filepath_to_timestamp(self, filepath: Path) -> tuple[datetime, datetime]:
        """Extract minimum and maximum timestamps from the filepath.

        Args:
            filepath: Full path to the file.

        Returns:
            (mint, maxt) tuple.
        """
        # Example file paths:
        #
        # * GCS/SC: 2024/10N/x086q72fv2f9q1x4a-0000000000-0000000000.tiff
        # * HF:     2024/U/1/L/7/471U_587L.tif
        date_format = '%Y'
        for part in pathlib.Path(filepath).parts[::-1]:
            try:
                return disambiguate_timestamp(part, date_format)
            except ValueError:
                pass

        return self.mint, self.maxt

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

        # Use PCA to project embeddings from 64D to 3D space
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
