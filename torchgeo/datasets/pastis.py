# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PASTIS dataset."""

import abc
import glob
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets.utils import check_integrity, extract_archive
from .geo import VisionDataset

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class PASTIS(VisionDataset, abc.ABC):
    """PASTIS dataset.

    The `PASTIS <https://github.com/VSainteuf/pastis-benchmark>`_
    dataset is a dataset for time-series panoptic segmentation of agricultural parcels

    Dataset features:

    * support for the original PASTIS and PASTIS-R versions of the dataset
    * 2,433 time-series with 10 m per pixel resolution (128x128 px)
    * 18 crop categories
    * semantic and instance annotations
    * 3 Sentinel-1 Ascending bands
    * 3 Sentinel-1 Descending bands
    * 10 Sentinel-2 multispectral bands

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

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2112.07558

    .. versionadded:: 0.2
    """

    classes = [
        "background",  # all non-agricultural land
        "meadow",
        "soft_winter_wheat",
        "corn",
        "winter_barley",
        "winter_rapeseed",
        "spring_barley",
        "sunflower",
        "grapevine",
        "beet",
        "winter_triticale",
        "winter_durum_wheat",
        "fruits_vegetables_flowers",
        "potatoes",
        "leguminous_fodder",
        "soybeans",
        "orchard",
        "mixed_cereal",
        "sorghum",
        "void_label",  # for parcels mostly outside their patch
    ]
    colormap = [
        (0.0, 0.0, 0.0),
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
        (1.0, 0.4980392156862745, 0.054901960784313725),
        (1.0, 0.7333333333333333, 0.47058823529411764),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        (1.0, 0.596078431372549, 0.5882352941176471),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
    ]
    directory = "PASTIS-R"
    filename = "PASTIS-R.zip"
    url = "https://zenodo.org/record/5735646/files/PASTIS-R.zip?download=1"
    md5 = "4887513d6c2d2b07fa935d325bd53e09"
    prefix = {
        "s2": os.path.join("DATA_S2", "S2_"),
        "s1a": os.path.join("DATA_S1A", "S1A_"),
        "s1d": os.path.join("DATA_S1D", "S1D_"),
        "semantic": os.path.join("ANNOTATIONS", "TARGET_"),
        "instance": os.path.join("INSTANCE_ANNOTATIONS", "INSTANCES_"),
    }

    def __init__(
        self,
        root: str = "data",
        bands: str = "all",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new PASTIS dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: load Sentinel-1, Sentinel-2, or both. One of {s1a, s1d, s2, all}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert bands in ["s1a", "s1d", "s2", "all"]
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.checksum = checksum
        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_image(self, index: int) -> Tensor:
        """Load a single time-series.

        Args:
            index: index to return

        Returns:
            the time-series
        """
        if self.bands == "all":
            paths = [self.files[index][sensor] for sensor in ["s1a", "s1d", "s2"]]
            arrays = [np.load(path) for path in paths]
            array = np.stack(arrays, axis=0)
        else:
            path = self.files[index][self.bands]
            array = np.load(path)

        tensor = torch.from_numpy(array)
        return tensor

    def _load_target(self, index: int) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask, box, and label for each mask
        """
        raise NotImplementedError

    def _load_files(self) -> List[Dict[str, str]]:
        """List the image and target files.

        Returns:
            list of dicts containing image and semantic/instance target file paths
        """
        ids = glob.glob(self.prefix["semantic"] + "*.npy")
        ids = [os.path.splitext(os.path.basename(i))[0].split("_")[-1] for i in ids]
        files = []
        for i in ids:
            suffix = f"{i}.npy"
            files.append(
                dict(
                    s2=os.path.join(self.root, self.prefix["s2"]) + suffix,
                    s1a=os.path.join(self.root, self.prefix["s1a"]) + suffix,
                    s1d=os.path.join(self.root, self.prefix["s1d"]) + suffix,
                    semantic=os.path.join(self.root, self.prefix["semantic"]) + suffix,
                    instance=os.path.join(self.root, self.prefix["instance"]) + suffix,
                )
            )
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not found
        """
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.directory)):
            return

        # Check if .zip files already exists (if so extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            if self.checksum and not check_integrity(filepath, self.md5):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        raise RuntimeError(
            "Dataset not found in `root` directory, "
            "specify a different `root` directory."
        )


class PASTISSemanticSegmentation(PASTIS):
    """PASTIS dataset for the semantic segmentation task."""

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        mask = self._load_target(index)
        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        array = np.load(self.files[index]["semantic"])
        tensor = torch.from_numpy(array)
        return tensor


class PASTISInstanceSegmentation(PASTIS):
    """PASTIS dataset for the instance segmentation task."""

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        mask, boxes, labels = self._load_target(index)
        sample = {"image": image, "mask": mask, "boxes": boxes, "label": labels}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_target(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask, box, and label for each mask
        """
        mask_array = np.load(self.files[index]["semantic"])
        instance_array = np.load(self.files[index]["instance"])

        mask_tensor = torch.from_numpy(mask_array)
        instance_tensor = torch.from_numpy(instance_array)

        # Convert from HxWxC to CxHxW
        mask_tensor = mask_tensor.permute((2, 0, 1))
        instance_tensor = instance_tensor.permute((2, 0, 1))

        # Convert instance mask of N instances to N binary instance masks
        instance_ids = torch.unique(instance_tensor)
        # Exclude a mask for unknown/background
        instance_ids = instance_ids[instance_ids != 0]
        instance_ids = instance_ids[:, None, None]
        masks: Tensor = instance_tensor == instance_ids

        # Parse labels for each instance
        labels_list = []
        for mask in masks:
            label = mask_tensor[mask[None, :, :]]
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
