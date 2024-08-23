# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Marine Debris dataset."""

import glob
import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torchvision.utils import draw_bounding_boxes

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, which


class NASAMarineDebris(NonGeoDataset):
    """NASA Marine Debris dataset.

    The `NASA Marine Debris <https://beta.source.coop/repositories/nasa/marine-debris/>`__
    dataset is a dataset for detection of floating marine debris in satellite imagery.

    Dataset features:

    * 707 patches with 3 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 1 object class: marine_debris
    * images taken by Planet Labs PlanetScope satellites
    * imagery taken from 2016-2019 from coasts of Greece, Honduras, and Ghana

    Dataset format:

    * images are three-channel geotiffs in uint8 format
    * labels are numpy files (.npy) containing bounding box (xyxy) coordinates
    * additional: images in jpg format and labels in geojson format

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/rdnt.9r6ekg

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.2
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/nasa-marine-debris'

    def __init__(
        self,
        root: Path = 'data',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new NASA Marine Debris Dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.transforms = transforms
        self.download = download

        self._verify()

        self.source = sorted(glob.glob(os.path.join(self.root, 'source', '*.tif')))
        self.labels = sorted(glob.glob(os.path.join(self.root, 'labels', '*.npy')))

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index
        """
        with rasterio.open(self.source[index]) as source:
            image = torch.from_numpy(source.read()).float()

        labels = np.load(self.labels[index])

        # Boxes contain unnecessary value of 1 after xyxy coords
        boxes = torch.from_numpy(labels[:, :4])

        # Filter invalid boxes
        w_check = (boxes[:, 2] - boxes[:, 0]) > 0
        h_check = (boxes[:, 3] - boxes[:, 1]) > 0
        indices = w_check & h_check
        boxes = boxes[indices]

        sample: Sample = {'image': image, 'boxes': boxes}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.source)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the directories already exist
        dirs = ['source', 'labels']
        exists = [os.path.exists(os.path.join(self.root, d)) for d in dirs]
        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', self.url, self.root, '--recursive=true')

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 1

        sample['image'] = sample['image'].byte()
        image = sample['image']
        if 'boxes' in sample and len(sample['boxes']):
            image = draw_bounding_boxes(image=sample['image'], boxes=sample['boxes'])
        image_arr = image.permute((1, 2, 0)).numpy()

        if 'prediction_boxes' in sample and len(sample['prediction_boxes']):
            ncols += 1
            preds = draw_bounding_boxes(
                image=sample['image'], boxes=sample['prediction_boxes']
            )
            preds_arr = preds.permute((1, 2, 0)).numpy()

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        if ncols < 2:
            axs.imshow(image_arr)
            axs.axis('off')
            if show_titles:
                axs.set_title('Ground Truth')
        else:
            axs[0].imshow(image_arr)
            axs[0].axis('off')
            axs[1].imshow(preds_arr)
            axs[1].axis('off')

            if show_titles:
                axs[0].set_title('Ground Truth')
                axs[1].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
