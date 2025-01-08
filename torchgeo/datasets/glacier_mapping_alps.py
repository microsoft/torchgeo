# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""GlacierMappingAlps Dataset."""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_and_extract_archive, download_url, extract_archive


class GlacierMappingAlps(NonGeoDataset):
    r"""A Multi-modal Dataset for Glacier Mapping (Segmentation) in the European Alps.

    The dataset consists of Sentinel-2 images from 2015 (mainly), 2016 and 2017, and binary segmentation masks for
    glaciers, based on an inventory built by glaciology experts
    (`Paul et al. 2020 <https://doi.org/10.1594/PANGAEA.909133>`_).

    Given that glacier ice is not always visible in the images, due to seasonal snow, shadow/cloud cover and, most
    importantly, debris cover, the dataset also includes additional features that can help in the segmentation task.

    Dataset features:

    * Sentinel-2 images (all bands, including cloud and shadow masks which can be used for loss masking)
    * glacier mask (0: no glacier, 1: glacier)
    * debris mask (0: no debris, 1: debris) based on a mix of three publications
      (`Scherler et al. 2018 <https://doi.org/10.5880/GFZ.3.3.2018.005>`_,
      `Herreid & Pellicciotti 2020 <https://doi.org/10.5281/zenodo.3866466>`_,
      `Linsbauer et al. 2021 <https://doi.glamos.ch/data/inventory/inventory_sgi2016_r2020.html>`_)
    * DEM (Copernicus GLO-30) + five derived features (using `xDEM <https://github.com/GlacioHack/xdem>`_): slope,
      aspect, terrain ruggedness index, planform and profile curvatures
    * dh/dt (surface elevation change) map over 2010-2015 (`Hugonnet et al. 2021 <https://doi.org/10.6096/13>`_)
    * v (surface velocity) map over 2015 (`ITS_LIVE <https://its-live.jpl.nasa.gov/>`_)

    Other specifications:

    * temporal coverage: one acquisition per glacier, from either 2015 (mainly), 2016, or 2017
    * spatial coverage: only glaciers larger than 0.1 km\ :sup:`2`\  are considered (n=1593, after manual QC), totalling
      ~1685 km\ :sup:`2`\  which represents ~93% of the total inventory area for this region
    * 2251 patches sampled with overlap from the 1593 glaciers;
      or 11440 for the `large` version, obtained with an increased sampling overlap
    * the dataset download size is 5.8 GB (11 GB when unarchived);
      or 29.5 GB (52 GB when unarchived) for the `large` version
    * the dataset is provided at 10m GSD (after bilinearly resampling some of the Sentinel-2 bands and the additional
      features which come at a lower resolution)
    * the dataset provides fixed training, validation, and test geographical splits (70-10-20, by glacier area)
    * five different splits are provided, according to a five-fold cross-validation scheme
    * all the features/masks are stacked and provided as NetCDF files (one or more per glacier), structured as
      `data/{glacier_id}/{glacier_id}_{patch_number}_{center_x}_{center_y}.nc`
    * data is projected and geocoded in local UTM zones

    For more details check also: https://huggingface.co/datasets/dcodrut/dl4gam_alps

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.22541/essoar.173557607.70204641/v1

    .. note::

        This dataset requires the following additional library to be installed:

        * `xarray <https://docs.xarray.dev/en/stable/getting-started-guide/installing.html>`_
        * `netcdf4 <https://unidata.github.io/netcdf4-python/>`_

    .. versionadded:: 0.7
    """

    r_url = 'https://huggingface.co/datasets/dcodrut/dl4gam_alps/resolve/7d20ca8a2b30c5518e086ffaa5ce37e6a66c42c1/data'
    download_metadata: ClassVar[dict[str, dict[str, str]]] = {
        'dataset_small': {
            'url': f'{r_url}/patches/inv_r_128_s_128.tar.gz',
            'checksum': '3e69c47c6ff5106cd4ffaa6bb2caaaef',
        },
        'dataset_large': {
            'url': f'{r_url}/patches/inv_r_128_s_64.tar.gz',
            'checksum': '06e85a6a9e3dc6b3cdb07f928e832bc8',
        },
        'splits_csv': {
            'url': f'{r_url}/map_all_splits_all_folds.csv',
            'checksum': '862355c5c3482271dd171d31c70551b3',
        },
    }

    extra_features_all = (
        'dem',  # Digital Elevation Model
        'slope',
        'aspect',
        'planform_curvature',
        'profile_curvature',
        'terrain_ruggedness_index',  # DEM-based features
        'dhdt',  # surface elevation change
        'v',  # surface velocity
    )

    rgb_bands = ('B4', 'B3', 'B2')
    all_bands = (
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B10',
        'B11',
        'B12',
    )
    rgb_nir_swir_bands = ('B4', 'B3', 'B2', 'B8', 'B11')  # the subset used in our paper

    def __init__(
        self,
        root: str | os.PathLike[str] = 'data',
        split: str = 'train',
        cv_iter: int = 1,
        version: str = 'small',
        bands: Sequence[str] = rgb_nir_swir_bands,
        extra_features: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            cv_iter: one of 1, 2, 3, 4, 5 (for the five-fold geographical cross-validation scheme)
            version: one of "small" or "large" (controls the sampling overlap)
            bands: the Sentinel-2 bands to use as input (default: RGB + NIR + SWIR)
            extra_features: additional features to include (default: None; see the class attribute for the available)
            transforms: a function/transform that takes input sample and its target as entry and returns a transformed
             version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = Path(root)
        self.split = split
        self.cv_iter = cv_iter
        self.version = version
        self.bands = bands
        self.extra_features = extra_features
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        # sanity checks
        assert split in (
            'train',
            'val',
            'test',
        ), f'Split {split} not found in the available splits'
        assert cv_iter in range(
            1, 6
        ), f'Cross-validation iteration {cv_iter} has to be in [1, 2, 3, 4, 5]'
        assert version in (
            'small',
            'large',
        ), f'Version {version} not found in the available versions'
        for band in bands:
            assert (
                band in self.all_bands
            ), f'Band {band} not found in the available Sentinel-2 bands'
        if self.extra_features:
            for feature in self.extra_features:
                assert (
                    feature in self.extra_features_all
                ), f'Feature {feature} not found in the available extra features'

        # set the local file paths
        label = f'dataset_{version}'
        self.fp_archive = self.root / f'{label}.tar.gz'
        self.dir_patches = self.root / label
        self.fp_splits_csv = self.root / 'splits.csv'

        # get the corresponding urls and checksums
        self.url_dataset = self.download_metadata[label]['url']
        self.md5_dataset = self.download_metadata[label]['checksum']
        self.url_csv_splits = self.download_metadata['splits_csv']['url']
        self.md5_csv_splits = self.download_metadata['splits_csv']['checksum']

        self._verify()
        self._prepare_files()

    def __len__(self) -> int:
        """The length of the dataset.

        It returns the number of patches in the dataset.
        """
        return len(self.fp_patches)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """A single sample from the dataset.

        It loads the netcdf file for the given index, and returns a dictionary containing:

        * the Sentinel-2 image (selected bands)
        * the glacier mask (binary mask with all the glaciers in the current patch)
        * the debris mask
        * the cloud and shadow mask
        * the additional features (DEM, derived features, etc.) if required
        """
        nc = xr.open_dataset(
            self.fp_patches[index], decode_coords='all', mask_and_scale=True
        )

        # extract the S2 image and masks from the netcdf file
        all_band_names = nc.band_data.long_name
        sample = {
            'image': nc.band_data.isel(
                band=[all_band_names.index(b) for b in self.bands]
            ).values.astype(np.float32),
            'mask_glacier': ~np.isnan(nc.mask_all_g_id.values),
            'mask_debris': (nc['mask_debris'].values == 1),
            'mask_clouds_and_shadows': ~(
                nc.band_data.isel(band=all_band_names.index('CLOUDLESS_MASK')).values
                == 1
            ),
        }

        # extract the additional features if needed
        if self.extra_features:
            for feature in self.extra_features:
                assert feature in nc, f'Feature {feature} not found in the netcdf file'
                data_crt_feature = nc[feature].values.astype(np.float32)

                # impute the missing values with the mean or zero (for dh/dt and surface velocity)
                v_fill = (
                    0.0 if feature in ('dhdt', 'v') else np.nanmean(data_crt_feature)
                )
                data_crt_feature[np.isnan(data_crt_feature)] = v_fill

                sample[feature] = data_crt_feature

        # convert to torch tensors
        for k, v in sample.items():
            sample[k] = torch.from_numpy(v)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        if self.dir_patches.exists() and self.fp_splits_csv.exists():
            return

        # check if the archive exists
        if self.fp_archive.exists():
            extract_archive(self.fp_archive, self.dir_patches)
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the patches and the csv with the splits."""
        # download and extract the archive
        download_and_extract_archive(
            url=self.url_dataset,
            download_root=self.root,
            extract_root=self.dir_patches,
            filename=self.fp_archive.name,
            md5=self.md5_dataset if self.checksum else None,
        )

        # download the splits csv
        download_url(
            url=self.url_csv_splits,
            root=self.root,
            filename='splits.csv',
            md5=self.md5_csv_splits if self.checksum else None,
        )

    def _prepare_files(self) -> None:
        """Prepare the files for the dataset."""
        # prepare the paths to the patches
        self.fp_patches = sorted(list(self.dir_patches.rglob('*.nc')))

        # get the glacier IDs of the current split of the cross-validation
        self.df_splits = pd.read_csv(self.fp_splits_csv)
        fold_name = f"fold_{self.split if self.split != 'val' else 'valid'}"
        idx = self.df_splits[f'split_{self.cv_iter}'] == fold_name
        glacier_ids = list(self.df_splits.loc[idx, 'entry_id'])

        # filter the patches to keep only the ones corresponding to the current split
        self.fp_patches = [
            fp for fp in self.fp_patches if fp.parent.name in glacier_ids
        ]

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
        clip_extrema: bool = True,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`GlacierMappingAlps.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            clip_extrema: flag indicating whether to clip the lowest/highest 2.5% of the values for contrast enhancement

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # we expect the RGB bands to be present
        assert {'B4', 'B3', 'B2'}.issubset(
            set(self.bands)
        ), 'RGB bands not found in the available bands.'
        nir_and_swir_present = {'B8', 'B11'}.issubset(set(self.bands))

        # prepare the RGB image and the masks
        idx_rgb = [self.bands.index(b) for b in ['B4', 'B3', 'B2']]
        rgb_img = sample['image'][idx_rgb].permute(1, 2, 0)
        images = {
            'RGB Image': rgb_img,
            'Glacier Mask': sample['mask_glacier'],
            'Debris Mask': sample['mask_debris'],
            'Clouds and Shadows Mask': sample['mask_clouds_and_shadows'],
        }

        # add the SWIR-NIR-R image if the bands are present
        if nir_and_swir_present:
            idx_swir_nir_r = [self.bands.index(b) for b in ['B11', 'B8', 'B4']]
            swir_nir_r_img = sample['image'][idx_swir_nir_r].permute(1, 2, 0)
            images['SWIR-NIR-R Image'] = swir_nir_r_img

        # add the extra features if present
        for extra_v, title in (
            ('prediction', 'Prediction'),
            ('dem', 'DEM'),
            ('slope', 'Slope'),
            ('aspect', 'Aspect'),
            ('planform_curvature', 'Planform Curvature'),
            ('profile_curvature', 'Profile Curvature'),
            ('terrain_ruggedness_index', 'Terrain Ruggedness Index'),
            ('dhdt', 'dh/dt'),
            ('v', 'Surface Velocity'),
        ):
            if extra_v in sample:
                images[title] = sample[extra_v]

        cmaps = {
            'RGB Image': None,
            'SWIR-NIR-R Image': None,
            'Glacier Mask': 'gray',
            'Prediction': 'gray',
            'Debris Mask': 'gray',
            'Clouds and Shadows Mask': 'gray',
            'DEM': 'terrain',
            'Slope': 'magma',
            'Aspect': 'jet',
            'Planform Curvature': 'magma',
            'Profile Curvature': 'magma',
            'Terrain Ruggedness Index': 'magma',
            'dh/dt': 'seismic_r',
            'Surface Velocity': 'magma',
        }

        # build the figure
        n_imgs = len(images)
        ncols = 4 if n_imgs <= 8 else 5
        nrows = int(np.ceil(n_imgs / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        for ax, k in zip(axs.flat, images):
            img = images[k].numpy()
            cmap = cmaps[k]

            # clip the extrema 5% of the values if needed
            if clip_extrema and k not in [
                'Glacier Mask',
                'Prediction',
                'Debris Mask',
                'Clouds and Shadows Mask',
            ]:
                q_lim_clip = 0.025
                img = np.clip(
                    img, np.quantile(img, q_lim_clip), np.quantile(img, 1 - q_lim_clip)
                )

            vmin, vmax = np.min(img), np.max(img)
            # smale the images to [0,1]
            if k in ['RGB Image', 'SWIR-NIR-R Image']:
                img = (img - vmin) / (vmax - vmin)

            if k == 'dh/dt':  # diverging colormap for the dh/dt, make it symmetric
                max_abs = max(abs(vmin), abs(vmax))
                vmin, vmax = -max_abs, max_abs

            ax.imshow(img, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
            if show_titles:
                ax.set_title(k)

        # disable the axes for all plots, including the empty plots
        for ax in axs.flat:
            ax.axis('off')

        if suptitle:
            fig.suptitle(suptitle)

        return fig
