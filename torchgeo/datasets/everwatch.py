# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EverWatch dataset."""

import os
from collections.abc import Callable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_and_extract_archive, extract_archive


class EverWatch(NonGeoDataset):
    """EverWatch Bird Detection dataset.

    The `EverWatch Bird Detection <https://zenodo.org/records/11165946>`__
    dataset contains high-resolution aerial images of birds in the Everglades National Park. Seven
    bird species haven been annotated and classified.

    Dataset features:

    * 5128 training images with 50491 annotations
    * 197 test images with 4113 annotations
    * seven different bird species

    Dataset format:

    * images are three-channel pngs
    * annotations are csv file

    Dataset Classes:

    0. White Ibis (Eudocimus albus)
    1. Great Egret (Ardea alba)
    2. Great Blue Heron (Ardea herodias)
    3. Snowy Egret (Egretta thula)
    4. Wood Stork (Mycteria americana)
    5. Roseate Spoonbill (Platalea ajaja)
    6. Anhinga (Anhinga anhinga)
    7. Unknown White (only present in test split)

    If you use this dataset in your research, please cite the following source:

    * https://doi.org/10.5281/zenodo.11165946

    .. versionadded:: 0.7
    """

    dir = 'everwatch-benchmark'

    url = 'https://zenodo.org/records/11165946/files/everwatch-benchmark.zip?download=1'

    md5 = 'ab34b0873b659656e36a9b41648f98db'

    zipfilename = 'everwatch-benchmark.zip'

    valid_splits = ('train', 'test')

    classes = (
        'White Ibis',
        'Great Egret',
        'Great Blue Heron',
        'Snowy Egret',
        'Wood Stork',
        'Roseate Spoonbill',
        'Anhinga',
        'Unknown White',  # only present in test split
    )

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EverWatch dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of {'train', 'val', 'test'} to specify the dataset split
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
            AssertionError: If *split* argument is invalid.
        """
        assert split in self.valid_splits, (
            f"Split '{split}' not supported, please use one of {self.valid_splits}"
        )

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.annot_df = pd.read_csv(
            os.path.join(self.root, self.dir, f'{self.split}.csv')
        )

        # remove all entries where xmin == xmax or ymin == ymax
        self.annot_df = self.annot_df[
            (self.annot_df['xmin'] != self.annot_df['xmax'])
            & (self.annot_df['ymin'] != self.annot_df['ymax'])
        ].reset_index(drop=True)

        # group per image path to get all annotations for one sample
        self.annot_df['sample_index'] = pd.factorize(self.annot_df['image_path'])[0]
        self.annot_df = self.annot_df.set_index(['sample_index', self.annot_df.index])

        self.class2idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.annot_df.index.levels[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample_df = self.annot_df.loc[index]

        img_path = os.path.join(self.root, self.dir, sample_df['image_path'].iloc[0])

        image = self._load_image(img_path)

        boxes, labels = self._load_target(sample_df)

        sample = {'image': image, 'bbox_xyxy': boxes, 'label': labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: np.typing.NDArray[np.uint8] = np.array(img)
            tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _load_target(self, sample_df: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """Load target from a dataframe row.

        Args:
            sample_df: df subset with annotations for specific image

        Returns:
            bounding boxes and labels
        """
        boxes = torch.from_numpy(
            sample_df[['xmin', 'ymin', 'xmax', 'ymax']].values
        ).float()
        labels = torch.Tensor(
            [self.class2idx[label] for label in sample_df['label'].tolist()]
        ).long()
        return boxes, labels

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        exists = []
        df_path = os.path.join(self.root, self.dir, f'{self.split}.csv')
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            image_paths = df['image_path'].unique().tolist()
            for path in image_paths:
                if os.path.exists(os.path.join(self.root, self.dir, path)):
                    exists.append(True)
        else:
            exists.append(False)

        if all(exists):
            return

        filepath = os.path.join(self.root, self.zipfilename)
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError('Dataset found, but corrupted.')
            extract_archive(filepath)
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.zipfilename,
            md5=self.md5 if self.checksum else None,
        )

    def plot(
        self,
        sample: dict[str, Tensor],
        suptitle: str | None = None,
        box_alpha: float = 0.7,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle
            box_alpha: alpha value for boxes

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample['image'].permute((1, 2, 0)).numpy()
        boxes = sample['bbox_xyxy'].numpy()
        labels = sample['label'].numpy()

        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))

        axs.imshow(image)
        axs.axis('off')

        cm = plt.get_cmap('gist_rainbow')

        for box, label_idx in zip(boxes, labels):
            color = cm(label_idx / len(self.classes))
            label = self.classes[label_idx]

            # Horizontal box: [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=box_alpha,
                linestyle='solid',
                edgecolor=color,
                facecolor='none',
            )
            axs.add_patch(rect)
            # Add label above box
            axs.text(
                x1,
                y1 - 5,
                label,
                color='white',
                fontsize=8,
                bbox=dict(facecolor=color, alpha=box_alpha),
            )

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
