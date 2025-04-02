# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Pretrain dataset."""

import random
from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.utils.data import IterableDataset

from ..utils import lazy_import, percentile_normalization


class CopernicusPretrain(IterableDataset[dict[str, Any]]):
    """Copernicus-Pretrain dataset.

    Copernicus-Pretrain is an extension of the SSL4EO-S12 dataset to all major Sentinel
    missions (S1-S5P). The images are organized into ~310K regional grids (0.25°x0.25°,
    consistent with ERA5), densely covering the whole land surface and near-land ocean
    with time series from eight distinct Sentinel modalities.

    This dataset class uses WebDataset for efficient data loading in distributed
    environments, which returns a PyTorch IterableDataset that is compatible with
    Pytorch DataLoader. Note: it is recommended to further use webdataset.WebLoader
    (a wrapper on DataLoader) for more features in data loading.

    The full dataset has varying number of modalities, S1/2 local patches, and
    timestamps for different grids. For simplicity, the current dataset class provides
    a minimum example:

    - only use grids with all modalities (220k)
    - sample one local patch for S1 and S2
    - sample one timestamp for each modality

    Therefore, each sample contains 8 tensors (S1, S2, S3, S5P NO2/CO/SO2/O3, DEM) and
    a JSON metadata.

    Example:

    .. code-block:: python

       dataset = CopernicusPretrain(
           urls='data/example-{000000..000009}.tar', shardshuffle=True, resampled=True
       )

       # Check the first sample
       sample = next(iter(dataset))
       s1 = sample['s1_grd.pth']
       s2 = sample['s2_toa.pth']
       s3 = sample['s3_olci.pth']
       s5p_co = sample['s5p_co.pth']
       s5p_no2 = sample['s5p_no2.pth']
       s5p_o3 = sample['s5p_o3.pth']
       s5p_so2 = sample['s5p_so2.pth']
       dem = sample['dem.pth']

       # Create a DataLoader for distributed training on 2 GPUs
       dataset = dataset.dataset.batched(10) # batch size
       dataloader = webdataset.WebLoader(
           dataset, batch_size=None, num_workers=2
       )
       # Unbatch, shuffle, and rebatch to mix samples from different workers
       dataloader = dataloader.unbatched().shuffle(100).batched(10)
       # A resampled dataset is infinite size, but we can recreate a fixed epoch length
       # Total number of samples / (batch size * world size)
       number_of_batches = 1000 // (10 * 2)
       dataloader = dataloader.with_epoch(number_of_batches)

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. note::

       This dataset requires the following additional library to be installed:

       * `<https://pypi.org/project/webdataset/>`_ to load the dataset.

    .. versionadded:: 0.7
    """

    url_dict: ClassVar[dict[str, str]] = {
        # grids with all modalities
        '220k_aligned': 'https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/ssl4eo_s_220k_aligned/example-{000000..002255}.tar',
        # remaining grids (with at least one modality)
        '220k_310k_union': 'https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/ssl4eo_s_220k_310k_union/example-{002256..003210}.tar',
        # 100 example grids
        '100_example': 'https://hf.co/datasets/wangyi111/Copernicus-Pretrain/resolve/d17e1098bd4fef52e7994805658434ce7e5800fc/example_100_grids/example_100_webdataset/example-{000000..000009}.tar',
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a new CopernicusPretrain instance.

        Args:
            *args: Arguments passed to WebDataset base class.
            **kwargs: Keyword arguments passed to WebDataset base class.
        """
        wds = lazy_import('webdataset')

        self.dataset = (
            wds.WebDataset(*args, **kwargs)
            .shuffle(10)  # shuffle individual samples before batching
            .decode()  # decode binary data
            .select(self._has_all_modalities)  # select samples with all modalities
            .map(self._sample_one_local_patch)  # sample one local patch for S1 and S2
            .map(self._sample_one_time_stamp)  # sample one timestamp for all modalities
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over images and metadata in the dataset.

        Returns:
            sample of images and metadata
        """
        return iter(self.dataset)

    def _has_all_modalities(self, sample: dict[str, Any]) -> bool:
        """Selection function: filter samples with all required modalities.

        Args:
            sample: A single sample from the dataset.

        Returns:
            True if all modalities are present in the sample, else False.
        """
        required_keys = [
            's1_grd.pth',
            's2_toa.pth',
            's3_olci.pth',
            's5p_co.pth',
            's5p_no2.pth',
            's5p_o3.pth',
            's5p_so2.pth',
            'dem.pth',
            'json',
        ]
        return all(key in sample for key in required_keys)

    def _sample_one_local_patch(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Mapping function: randomly select one local patch for S1 and S2.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only a single patch for S1 and S2.
        """
        s1, s2 = sample['s1_grd.pth'], sample['s2_toa.pth']
        meta_s1, meta_s2 = sample['json']['s1_grd'], sample['json']['s2_toa']

        idx = random.randint(0, s1.shape[0] - 1)
        sample['s1_grd.pth'], sample['s2_toa.pth'] = s1[idx], s2[idx]
        sample['json']['s1_grd'], sample['json']['s2_toa'] = meta_s1[idx], meta_s2[idx]
        return sample

    def _sample_one_time_stamp(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Mapping function: randomly select one timestamp for all modalities.

        Args:
            sample: A single sample from the dataset.

        Returns:
            The same sample with only a single timestamp.
        """
        for key in sample:
            if key.endswith('.pth') and key != 'dem.pth':
                idx = random.randint(0, sample[key].shape[0] - 1)
                sample[key] = sample[key][idx]
                sample['json'][key.replace('.pth', '')] = sample['json'][
                    key.replace('.pth', '')
                ][idx]

        sample['json']['dem'] = sample['json']['dem'][0]
        return sample

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__iter__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        fig, ax = plt.subplots(nrows=2, ncols=4)

        image = sample['s1_grd.pth'].numpy()
        vv = image[0]
        vh = image[1]
        image = np.stack([vv, vh, (vv + vh) / 2], axis=-1)
        image = percentile_normalization(image)
        ax[0, 0].imshow(image)
        ax[0, 0].axis('off')

        rgb_bands = [3, 2, 1]
        image = sample['s2_toa.pth'].numpy()[rgb_bands]
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)
        ax[0, 1].imshow(image)
        ax[0, 1].axis('off')

        rgb_bands = [7, 5, 3]
        image = sample['s3_olci.pth'].numpy()[rgb_bands]
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)
        ax[0, 2].imshow(image)
        ax[0, 2].axis('off')

        image = sample['dem.pth'].numpy()[0]
        ax[0, 3].imshow(image, cmap='terrain')
        ax[0, 3].axis('off')

        image = sample['s5p_co.pth'].numpy()[0]
        ax[1, 0].imshow(image, cmap='Wistia')
        ax[1, 0].axis('off')

        image = sample['s5p_no2.pth'].numpy()[0]
        ax[1, 1].imshow(image, cmap='Wistia')
        ax[1, 1].axis('off')

        image = sample['s5p_o3.pth'].numpy()[0]
        ax[1, 2].imshow(image, cmap='Wistia')
        ax[1, 2].axis('off')

        image = sample['s5p_so2.pth'].numpy()[0]
        ax[1, 3].imshow(image, cmap='Wistia')
        ax[1, 3].axis('off')

        if show_titles:
            ax[0, 0].set_title('S1 GRD')
            ax[0, 1].set_title('S2 TOA')
            ax[0, 2].set_title('S3 OLCI')
            ax[0, 3].set_title('DEM')
            ax[1, 0].set_title('S5P CO')
            ax[1, 1].set_title('S5P NO$_2$')
            ax[1, 2].set_title('S5P O$_3$')
            ax[1, 3].set_title('S5P SO$_2$')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        return fig
