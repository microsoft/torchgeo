# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Embedded Seamless Data."""

from collections.abc import Sequence

import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from .geo import RasterDataset
from .utils import GeoSlice, Sample


class ESDQuantizer:
    """Decode ESD-encoded quantized indices into continuous embedding vectors.

    The ESDQuantizerDecoder converts integer quantization indices produced by an
    ESD quantizer into continuous values in the range [-1, 1], representing
    multi-level embeddings of the original input. This enables downstream tasks,
    such as visualization, machine learning, or spatial analysis, to operate
    directly on decoded embeddings without reconstructing the full raw data.

    Key points:

    * Factorized decoding: Each index is split into multiple levels according
      to the quantizer configuration.
    * Continuous mapping: Level indices are rescaled and centered to [-1, 1],
      preserving relative distances in embedding space.
    * Fully vectorized: The decoding is performed on entire tensors at once,
      avoiding slow per-pixel loops and enabling GPU acceleration.
    * Flexible input: Supports arbitrary batch sizes and spatial dimensions
      (..., H, W).

    Usage:

    .. code-block:: python

        decoder = ESDQuantizerDecoder()
        decoded = decoder.apply_transform(torch.from_numpy(ESD_codes.astype(np.int32)))

    .. note::
        The output retains the channel dimension corresponding to embedding levels.
        Users can further convert embeddings to visualizations or aggregate them
        for downstream tasks.

    """

    def __init__(self, levels: Sequence[int] = (8, 8, 8, 5, 5, 5)) -> None:
        """Initialize the quantization levels for the embedding.

        Args:
            levels: Sequence of integers specifying the number of quantization
                levels for each embedding dimension.
        """
        levels_tensor = torch.tensor(levels, dtype=torch.int32)
        self._levels = levels_tensor

        basis_tensor = torch.cumprod(
            torch.tensor([1, *levels[:-1]], dtype=torch.int32), dim=0
        )
        self._basis = basis_tensor

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Convert embedding indices to normalized continuous codes.

        Args:
            indices: A tensor of integer indices representing quantized embeddings.
                Shape can be arbitrary (...,).

        Returns:
            Tensor of the same shape as `indices` with an additional channel
            dimension for embedding levels. Values are normalized to [-1, 1].
        """
        indices = rearrange(indices, '... -> ... 1')
        level_indices = (indices // self._basis) % self._levels

        half = self._levels // 2
        codes = (level_indices - half) / half

        return codes

    def quantize(self, input: Tensor) -> Tensor:
        """Quantize the input tensor using predefined levels.

        Args:
            input: A tensor containing integer embedding indices. Shape can be
                arbitrary (...,).

        Returns:
            Tensor of quantized codes with normalized values in [-1, 1], where
            the last dimension represents embedding levels. Channels are moved
            to -3 position for compatibility with expected output layout.
        """
        return self.indices_to_codes(input).movedim(-1, -3)


class EmbeddedSeamlessData(RasterDataset):
    """Embedded Seamless Data (ESD).

    The `Embedded Seamless Data (ESD) <https://arxiv.org/abs/2601.11183>`__
    is a global, analysis-ready Earth embedding dataset at 30-meter resolution,
    designed to overcome the computational and storage challenges of planetary-scale
    Earth system science. By transforming multi-sensor satellite observations into
    compact, quantized latent vectors, ESD reduces the original data volume (~1 PB for
    a full year of global land surfaces) to approximately 2.4 TB, enabling decadal-scale
    analysis on standard workstations.

    Key features:

    * **Longitudinal Consistency**: Provides a continuous record from 2000 to 2024,
      harmonized from Landsat 5, 7, 8, 9, MODIS Terra and NASADEM imagery.
    * **High Reconstructive Fidelity**: Achieves a Mean Absolute Error (MAE) of 0.013
      across six spectral bands, ensuring the embeddings retain physically meaningful
      surface information.
    * **Semantic Intelligence**: Captures complex land surface patterns, outperforming
      raw sensor fusion data for land-cover classification (global accuracy 79.74%).
    * **Implicit Denoising**: Filters transient noise such as clouds and shadows via
      the ESDNet architecture, producing clean signals suitable for temporal and
      environmental monitoring.
    * **Few-Shot Proficiency**: Supports robust learning with minimal labeled data,
      ideal for regions with scarce ground-truth measurements.
    * **Compact and Vectorized**: Each 30-meter pixel is represented by a
      high-dimensional embedding vector, which can be aggregated, compared, or analyzed
      efficiently without reconstructing raw imagery.

    The dataset covers terrestrial land surfaces, shallow waters, intertidal and reef
    zones, inland waterways, and coastal regions. Polar coverage is limited by satellite
    orbits and sensor availability.

    Produced by the ESDNet framework, ESD provides an ultra-lightweight, globally
    consistent representation of surface conditions, enabling flexible, high-resolution
    analysis of land surface dynamics over decades.

    If you use this dataset in your research, please refer to:

    * Paper: https://arxiv.org/abs/2601.11183
    * Code: https://github.com/shuangchencc/ESD
    * Dataset: https://data-starcloud.pcl.ac.cn/iearthdata/64

    .. versionadded:: 0.9
    """

    # SDC30_EBD_V001_02VMN_2024.tif
    filename_glob = 'SDC30_EBD_*'
    filename_regex = r'.*_(?P<date>\d{4})'
    date_format = '%Y'

    quantizer = ESDQuantizer()

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        sample = super().__getitem__(index)
        sample['image'] = self.quantizer.quantize(sample['image'])
        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: Sample dict containing 'image' tensor and metadata.
            show_titles: Whether to show titles on subplots.
            suptitle: Optional figure suptitle.

        Returns:
            a matplotlib Figure with the rendered sample
        """
        vectors = sample['image']
        _months, _channels, H, W = vectors.shape

        # Compute valid mask: any non-zero pixel across channels
        valid_mask = ~torch.isclose(
            vectors, torch.tensor(0.0, device=vectors.device), atol=1e-6
        )
        valid_mask = (
            valid_mask[:12].any(dim=1).any(dim=0)
        )  # combine first 12 months, shape (H, W)

        # Reduce channels to RGB using mean over selected channels
        R = (vectors[:, 5].mean(dim=0) + 1) / 2  # normalize to [0,1]
        G = (vectors[:, 1].mean(dim=0) + 1) / 2
        B = (vectors[:, 2].mean(dim=0) + 1) / 2

        # Clamp to [0,1] and convert to uint8
        disp_img = torch.zeros(H, W, 4, dtype=torch.uint8, device='cpu')
        disp_img[..., 0] = (R.clamp(0, 1) * 255).to(torch.uint8).cpu()
        disp_img[..., 1] = (G.clamp(0, 1) * 255).to(torch.uint8).cpu()
        disp_img[..., 2] = (B.clamp(0, 1) * 255).to(torch.uint8).cpu()
        disp_img[..., 3] = valid_mask.to(torch.uint8) * 255  # alpha channel

        # Plot
        fig, ax = plt.subplots()
        ax.imshow(disp_img.cpu().numpy())
        ax.axis('off')

        if show_titles:
            ax.set_title('ESD Embedding Visualization')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
