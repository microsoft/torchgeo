# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import fiona
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, check_integrity, download_url, extract_archive


class PASTIS(NonGeoDataset):
    """PASTIS dataset.

    The `PASTIS <https://github.com/VSainteuf/pastis-benchmark>`__
    dataset is a dataset for time-series panoptic segmentation of agricultural parcels.

    Dataset features:

    * support for the original PASTIS and PASTIS-R versions of the dataset
    * 2,433 time-series with 10 m per pixel resolution (128x128 px)
    * 18 crop categories, 1 background category, 1 void category
    * semantic and instance annotations
    * 3 Sentinel-1 Ascending bands
    * 3 Sentinel-1 Descending bands
    * 10 Sentinel-2 L2A multispectral bands

    Dataset format:

    * time-series and annotations are in numpy format (.npy)

    Dataset classes:

    0. Background
    1. Meadow
    2. Soft Winter Wheat
    3. Corn
    4. Winter Barley
    5. Winter Rapeseed
    6. Spring Barley
    7. Sunflower
    8. Grapevine
    9. Beet
    10. Winter Triticale
    11. Winter Durum Wheat
    12. Fruits Vegetables Flowers
    13. Potatoes
    14. Leguminous Fodder
    15. Soybeans
    16. Orchard
    17. Mixed Cereal
    18. Sorghum
    19. Void Label

    If you use this dataset in your research, please cite the following papers:

    * https://doi.org/10.1109/ICCV48922.2021.00483
    * https://doi.org/10.1016/j.isprsjprs.2022.03.012

    .. versionadded:: 0.5
    """

    classes = (
        'background',  # all non-agricultural land
        'meadow',
        'soft_winter_wheat',
        'corn',
        'winter_barley',
        'winter_rapeseed',
        'spring_barley',
        'sunflower',
        'grapevine',
        'beet',
        'winter_triticale',
        'winter_durum_wheat',
        'fruits_vegetables_flowers',
        'potatoes',
        'leguminous_fodder',
        'soybeans',
        'orchard',
        'mixed_cereal',
        'sorghum',
        'void_label',  # for parcels mostly outside their patch
    )
    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (0, 0, 0, 255),
        1: (174, 199, 232, 255),
        2: (255, 127, 14, 255),
        3: (255, 187, 120, 255),
        4: (44, 160, 44, 255),
        5: (152, 223, 138, 255),
        6: (214, 39, 40, 255),
        7: (255, 152, 150, 255),
        8: (148, 103, 189, 255),
        9: (197, 176, 213, 255),
        10: (140, 86, 75, 255),
        11: (196, 156, 148, 255),
        12: (227, 119, 194, 255),
        13: (247, 182, 210, 255),
        14: (127, 127, 127, 255),
        15: (199, 199, 199, 255),
        16: (188, 189, 34, 255),
        17: (219, 219, 141, 255),
        18: (23, 190, 207, 255),
        19: (255, 255, 255, 255),
    }
    directory = 'PASTIS-R'
    filename = 'PASTIS-R.zip'
    url = 'https://zenodo.org/records/5735646/files/PASTIS-R.zip?download=1'
    md5 = '4887513d6c2d2b07fa935d325bd53e09'
    prefix: ClassVar[dict[str, str]] = {
        's2': os.path.join('DATA_S2', 'S2_'),
        's1a': os.path.join('DATA_S1A', 'S1A_'),
        's1d': os.path.join('DATA_S1D', 'S1D_'),
        'semantic': os.path.join('ANNOTATIONS', 'TARGET_'),
        'instance': os.path.join('INSTANCE_ANNOTATIONS', 'INSTANCES_'),
    }

    def __init__(
        self,
        root: Path = 'data',
        folds: Sequence[int] = (1, 2, 3, 4, 5),
        bands: str = 's2',
        mode: str = 'semantic',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new PASTIS dataset instance.

        Args:
            root: root directory where dataset can be found
            folds: a sequence of integers from 0 to 4 specifying which of the five
                dataset folds to include
            bands: load Sentinel-1 ascending path data (s1a), Sentinel-1 descending path
                data (s1d), or Sentinel-2 data (s2)
            mode: load semantic (semantic) or instance (instance) annotations
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        for fold in folds:
            assert 1 <= fold <= 5
        assert bands in ['s1a', 's1d', 's2']
        assert mode in ['semantic', 'instance']
        self.root = root
        self.folds = folds
        self.bands = bands
        self.mode = mode
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()
        self.files = self._load_files()

        colors = []
        for i in range(len(self.cmap)):
            colors.append(
                (
                    self.cmap[i][0] / 255.0,
                    self.cmap[i][1] / 255.0,
                    self.cmap[i][2] / 255.0,
                )
            )
        self._cmap = ListedColormap(colors)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        if self.mode == 'semantic':
            mask = self._load_semantic_targets(index)
            sample: Sample = {'image': image, 'mask': mask}
        elif self.mode == 'instance':
            mask, boxes, labels = self._load_instance_targets(index)
            sample: Sample = {
                'image': image,
                'mask': mask,
                'boxes': boxes,
                'label': labels,
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.idxs)

    def _load_image(self, index: int) -> Tensor:
        """Load a single time-series.

        Args:
            index: index to return

        Returns:
            the time-series
        """
        path = self.files[index][self.bands]
        array = np.load(path)

        tensor = torch.from_numpy(array)
        return tensor

    def _load_semantic_targets(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        # See https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py#L201
        # even though the mask file is 3 bands, we just select the first band
        array = np.load(self.files[index]['semantic'])[0].astype(np.uint8)
        tensor = torch.from_numpy(array).long()
        return tensor

    def _load_instance_targets(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Load the instance segmentation targets for a single sample.

        Args:
            index: index to return

        Returns:
            the instance segmentation mask, box, and label for each instance
        """
        mask_array = np.load(self.files[index]['semantic'])[0]
        instance_array = np.load(self.files[index]['instance'])

        mask_tensor = torch.from_numpy(mask_array)
        instance_tensor = torch.from_numpy(instance_array)

        # Convert instance mask of N instances to N binary instance masks
        instance_ids = torch.unique(instance_tensor)
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]
        instance_ids = instance_ids[:, None, None]
        masks: Tensor = instance_tensor == instance_ids

        # Parse labels for each instance
        labels_list = []
        for mask in masks:
            label = mask_tensor[mask]
            label = torch.unique(label)[0]
            labels_list.append(label)

        # Get bounding boxes for each instance
        boxes_list = []
        for mask in masks:
            pos = torch.where(mask)
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            boxes_list.append([xmin, ymin, xmax, ymax])

        masks = masks.to(torch.uint8)
        boxes = torch.tensor(boxes_list).to(torch.float)
        labels = torch.tensor(labels_list).to(torch.long)

        return masks, boxes, labels

    def _load_files(self) -> list[dict[str, str]]:
        """List the image and target files.

        Returns:
            list of dicts containing image and semantic/instance target file paths
        """
        self.idxs = []
        metadata_fn = os.path.join(self.root, self.directory, 'metadata.geojson')
        with fiona.open(metadata_fn) as f:
            for row in f:
                fold = int(row['properties']['Fold'])
                if fold in self.folds:
                    self.idxs.append(row['properties']['ID_PATCH'])

        files = []
        for i in self.idxs:
            path = os.path.join(self.root, self.directory, '{}') + str(i) + '.npy'
            files.append(
                {
                    's2': path.format(self.prefix['s2']),
                    's1a': path.format(self.prefix['s1a']),
                    's1d': path.format(self.prefix['s1d']),
                    'semantic': path.format(self.prefix['semantic']),
                    'instance': path.format(self.prefix['instance']),
                }
            )
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the directory already exists
        path = os.path.join(self.root, self.directory)
        if os.path.exists(path):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError('Dataset found, but corrupted.')
            extract_archive(filepath)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        extract_archive(os.path.join(self.root, self.filename), self.root)

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
        # Keep the RGB bands and convert to T x H x W x C format
        images = sample['image'][:, [2, 1, 0], :, :].numpy().transpose(0, 2, 3, 1)
        mask = sample['mask'].numpy()

        if self.mode == 'instance':
            label = sample['label']
            mask = label[mask.argmax(axis=0)].numpy()

        num_panels = 3
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy()
            num_panels += 1
            if self.mode == 'instance':
                predictions = predictions.argmax(axis=0)
                label = sample['prediction_labels']
                predictions = label[predictions].numpy()

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 4))
        axs[0].imshow(images[0] / 5000)
        axs[1].imshow(images[1] / 5000)
        axs[2].imshow(mask, vmin=0, vmax=19, cmap=self._cmap, interpolation='none')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        if showing_predictions:
            axs[3].imshow(
                predictions, vmin=0, vmax=19, cmap=self._cmap, interpolation='none'
            )
            axs[3].axis('off')

        if show_titles:
            axs[0].set_title('Image 0')
            axs[1].set_title('Image 1')
            axs[2].set_title('Mask')
            if showing_predictions:
                axs[3].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
