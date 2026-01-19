# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Embed dataset."""

import os
from collections.abc import Callable, Iterable

import einops
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyproj import CRS

from ..errors import DatasetNotFoundError
from ..geo import RasterDataset
from ..utils import Path, Sample, download_url


class CopernicusEmbed(RasterDataset):
    """Copernicus-Embed dataset.

    `Copernicus-Embed
    <https://github.com/zhu-xlab/Copernicus-FM/tree/main/Copernicus-Embed-025deg>`__
    is an embedding dataset that gives each 0.25x0.25 grid one embedding vector,
    aggregated over all available modalities from the whole Copernicus-Pretrain dataset
    (721x1440x768, filling empty ocean grids with 0). This dataset can be seen as a
    semantic representation product that integrates various sources of satellite
    observations at an extremely high compression ratio. It also makes it very
    convenient to link Earth's surface to the atmosphere (e.g., as improved static
    variables adding to ERA5), unlocking new possibilities in the development of
    weather/climate foundation models.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.9
    """

    filename_glob = 'embed_map_*'

    url = 'https://hf.co/datasets/torchgeo/copernicus_embed/resolve/435b4a7bdce6f6fdbf4272f9d6e54f2604f35fdb/embed_map_310k.tif'
    md5 = '63de14ab9f5eeffb785066f3013a40b4'

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusEmbed instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        if self.files:
            return

        if self.download:
            assert isinstance(self.paths, str | os.PathLike)
            download_url(self.url, self.paths, md5=self.md5 if self.checksum else None)
        else:
            raise DatasetNotFoundError(self)

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

        # Use PCA to project embeddings from 768D to 3D space
        valid = A.sum(dim=1) != 0
        invalid = A.sum(dim=1) == 0
        _, _, V = torch.pca_lowrank(A[valid], q=3)
        B = A @ V

        B -= B[valid].min(dim=0, keepdim=True)[0]
        B /= B[valid].max(dim=0, keepdim=True)[0]
        B[invalid] = 1
        image = einops.rearrange(B, '(h w) c -> h w c', h=h, w=w)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title('Embedding')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
