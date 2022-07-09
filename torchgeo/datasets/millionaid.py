# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Million-AID dataset."""
import glob
import os
from typing import Any, Callable, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from torchgeo.datasets import NonGeoDataset

from .utils import check_integrity, extract_archive


class MillionAID(NonGeoDataset):
    """Million-AID Dataset.

    The `MillionAID <https://captain-whu.github.io/DiRS/>`_ dataset consists
    of one million aerial images from Google Earth Engine that offers
    either `a multi-class learning task
    <https://competitions.codalab.org/competitions/35945#learn_the_details-dataset>`_
    with 51 classes or a `multi-label learning task
    <https://competitions.codalab.org/competitions/35974#learn_the_details-dataset>`_
    with 73 different possible labels. For more details please consult
    the accompanying `paper <https://ieeexplore.ieee.org/document/9393553>`_.

    Dataset features:

    * RGB aerial images with varying resolutions from 0.5 m to 153 m per pixel
    * images within classes can have different pixel dimension

    Dataset format:

    * images are three-channel jpg

    If you use this dataset in your research, please cite the following paper:

    * https://ieeexplore.ieee.org/document/9393553

    .. versionadded:: 0.3
    """

    multi_label_categories = [
        "agriculture_land",
        "airport_area",
        "apartment",
        "apron",
        "arable_land",
        "bare_land",
        "baseball_field",
        "basketball_court",
        "beach",
        "bridge",
        "cemetery",
        "church",
        "commercial_area",
        "commercial_land",
        "dam",
        "desert",
        "detached_house",
        "dry_field",
        "factory_area",
        "forest",
        "golf_course",
        "grassland",
        "greenhouse",
        "ground_track_field",
        "helipad",
        "highway_area",
        "ice_land",
        "industrial_land",
        "intersection",
        "island",
        "lake",
        "leisure_land",
        "meadow",
        "mine",
        "mining_area",
        "mobile_home_park",
        "oil_field",
        "orchard",
        "paddy_field",
        "parking_lot",
        "pier",
        "port_area",
        "power_station",
        "public_service_land",
        "quarry",
        "railway",
        "railway_area",
        "religious_land",
        "residential_land",
        "river",
        "road",
        "rock_land",
        "roundabout",
        "runway",
        "solar_power_plant",
        "sparse_shrub_land",
        "special_land",
        "sports_land",
        "stadium",
        "storage_tank",
        "substation",
        "swimming_pool",
        "tennis_court",
        "terraced_field",
        "train_station",
        "transportation_land",
        "unutilized_land",
        "viaduct",
        "wastewater_plant",
        "water_area",
        "wind_turbine",
        "woodland",
        "works",
    ]

    multi_class_categories = [
        "apartment",
        "apron",
        "bare_land",
        "baseball_field",
        "bapsketball_court",
        "beach",
        "bridge",
        "cemetery",
        "church",
        "commercial_area",
        "dam",
        "desert",
        "detached_house",
        "dry_field",
        "forest",
        "golf_course",
        "greenhouse",
        "ground_track_field",
        "helipad",
        "ice_land",
        "intersection",
        "island",
        "lake",
        "meadow",
        "mine",
        "mobile_home_park",
        "oil_field",
        "orchard",
        "paddy_field",
        "parking_lot",
        "pier",
        "quarry",
        "railway",
        "river",
        "road",
        "rock_land",
        "roundabout",
        "runway",
        "solar_power_plant",
        "sparse_shrub_land",
        "stadium",
        "storage_tank",
        "substation",
        "swimming_pool",
        "tennis_court",
        "terraced_field",
        "train_station",
        "viaduct",
        "wastewater_plant",
        "wind_turbine",
        "works",
    ]

    md5s = {
        "train": "1b40503cafa9b0601653ca36cd788852",
        "test": "51a63ee3eeb1351889eacff349a983d8",
    }

    filenames = {"train": "train.zip", "test": "test.zip"}

    tasks = ["multi-class", "multi-label"]
    splits = ["train", "test"]

    def __init__(
        self,
        root: str = "data",
        task: str = "multi-class",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new MillionAID dataset instance.

        Args:
            root: root directory where dataset can be found
            task: type of task, either "multi-class" or "multi-label"
            split: train or test split
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if dataset is not found
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        assert task in self.tasks
        assert split in self.splits
        self.task = task
        self.split = split

        self._verify()

        self.files = self._load_files(self.root)

        self.classes = sorted({cls for f in self.files for cls in f["label"]})
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image = self._load_image(files["image"])
        cls_label = [self.class_to_idx[label] for label in files["label"]]
        label = torch.tensor(cls_label, dtype=torch.long)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_files(self, root: str) -> List[Dict[str, Any]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root directory of dataset

        Returns:
            list of dicts containing paths for each pair of image, and list of labels
        """
        imgs_no_subcat = list(
            glob.glob(os.path.join(root, self.split, "*", "*", "*.jpg"))
        )

        imgs_subcat = list(
            glob.glob(os.path.join(root, self.split, "*", "*", "*", "*.jpg"))
        )

        scenes = [p.split(os.sep)[-3] for p in imgs_no_subcat] + [
            p.split(os.sep)[-4] for p in imgs_subcat
        ]

        subcategories = ["Missing" for p in imgs_no_subcat] + [
            p.split(os.sep)[-3] for p in imgs_subcat
        ]

        classes = [p.split(os.sep)[-2] for p in imgs_no_subcat] + [
            p.split(os.sep)[-2] for p in imgs_subcat
        ]

        if self.task == "multi-label":
            labels = [
                [sc, sub, c] if sub != "Missing" else [sc, c]
                for sc, sub, c in zip(scenes, subcategories, classes)
            ]
        else:
            labels = [[c] for c in classes]

        images = imgs_no_subcat + imgs_subcat

        files = [dict(image=img, label=l) for img, l in zip(images, labels)]

        return files

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with Image.open(path) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("RGB"))
            tensor: Tensor = torch.from_numpy(array)
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    def _verify(self) -> None:
        """Checks the integrity of the dataset structure.

        Returns:
            True if the dataset directories are found, else False
        """
        filepath = os.path.join(self.root, self.split)
        if os.path.isdir(filepath):
            return

        filepath = os.path.join(self.root, self.split + ".zip")
        if os.path.isfile(filepath):
            if self.checksum and not check_integrity(filepath, self.md5s[self.split]):
                raise RuntimeError("Dataset found, but corrupted.")
            extract_archive(filepath)
            return

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` directory, either "
            "specify a different `root` directory or manually download "
            "the dataset to this directory."
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        labels = [self.classes[cast(int, label)] for label in sample["label"]]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_labels = [
                self.classes[cast(int, label)] for label in sample["prediction"]
            ]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {labels}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_labels}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
