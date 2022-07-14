# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Urban 3D Challenge dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.utils import draw_segmentation_masks

from torchgeo.datasets.utils import percentile_normalization

from .geo import VisionDataset


class Urban3DChallenge(VisionDataset):
    """Urban 3D Challenge dataset.

    The `Urban 3D Challenge <https://spacenet.ai/the-ussocom-urban-3d-competition/>`_
    dataset is a dataset for semantic/instance segmentation of building footprints in
    2D RGB imagery and 3D digital surface and terrain models.

    Dataset features:

    * 278 images with 0.5 m per pixel resolution (2,048-2,048 px)
    * 157,0000 building instances
    * 6 bands (RGB, DTM, DSM, normalized DSM (nDSM))
    * RGB imagery taken by Maxar WorldView 2-3 satellites

    Dataset classes:

    0. Background
    1. Building

    Dataset format:

    * images are three-channel tiffs
    * digital surface and terrain models are one-channel tiffs
    * instance and binary masks are one-channel tiffs

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/AIPR.2017.8457973

    .. note::

       This dataset can be downloaded using the following bash script:

       .. code-block:: bash

          mkdir 01-Provisional_Train 02-Provisional_Test 03-Sequestered_Test
          aws s3 sync s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/01-Provisional_Train/ 01-Provisional_Train/
          aws s3 sync s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/02-Provisional_Test/ 02-Provisional_Test/
          aws s3 sync s3://spacenet-dataset/Hosted-Datasets/Urban_3D_Challenge/03-Sequestered_Test/ 03-Sequestered_Test/

    .. versionadded:: 0.3
    """  # noqa: E501

    directories = {
        "train": "01-Provisional_Train",
        "val": "02-Provisional_Test",
        "test": "03-Sequestered_Test",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new Urban3DChallenge dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            AssertionError: if ``split`` is invalid
        """
        assert split in self.directories
        self.root = root
        self.split = split
        self.transforms = transforms
        self._verify()
        self.files = self._load_files()

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
        rgb = self._load_image(files["rgb"])
        dtm = self._load_image(files["dtm"])
        dsm = self._load_image(files["dsm"])
        image = torch.cat([rgb, dtm, dsm], dim=0)

        mask = self._load_target(files["binary_mask"])
        mask = mask.to(torch.long)

        instances = self._load_image(files["instance_mask"]).squeeze(dim=0)
        instances = instances.to(torch.long)

        sample = {"image": image, "mask": mask, "instances": instances}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> Tensor:
        """Load a single image.

        Args:
            path: path to image

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array = f.read(out_dtype="float32")
            tensor: Tensor = torch.from_numpy(array)
            return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load a single target.

        Args:
            path: path to target

        Returns:
            the target
        """
        mask = self._load_image(path).squeeze(dim=0)
        mask[mask == 2] = 0  # 2 = background
        mask[mask == 6] = 1  # 6 = building
        mask[mask == 65] = 1  # 65 = building partially overlapping with nodata
        return mask

    def _load_files(self) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each sample
        """
        image_root = os.path.join(self.root, self.directories[self.split], "Inputs")
        target_root = os.path.join(self.root, self.directories[self.split], "GT")
        basenames = [
            os.path.basename(f) for f in glob.glob(os.path.join(image_root, "*.tif"))
        ]
        prefixes = {os.path.splitext(f)[0].rsplit("_", 1)[0] for f in basenames}

        files = []
        for prefix in sorted(prefixes):
            files.append(
                dict(
                    rgb=os.path.join(image_root, f"{prefix}_RGB.tif"),
                    dtm=os.path.join(image_root, f"{prefix}_DTM.tif"),
                    dsm=os.path.join(image_root, f"{prefix}_DSM.tif"),
                    binary_mask=os.path.join(target_root, f"{prefix}_GTL.tif"),
                    instance_mask=os.path.join(target_root, f"{prefix}_GTI.tif"),
                )
            )
        return files

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if the dataset is not found
        """
        # Check if the files already exist
        exists = []
        for directory in self.directories.values():
            exists.append(os.path.exists(os.path.join(self.root, directory)))
        if all(exists):
            return

        raise RuntimeError(
            f"Dataset not found in {self.root} directory, "
            f"specify a different {self.root} directory."
        )

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure
            alpha: opacity with which to render predictions on top of the imagery

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 4
        image = sample["image"][:3].permute(1, 2, 0).numpy()
        image = percentile_normalization(image, axis=(0, 1))
        image = (image * 255).astype(np.uint8)
        dtm = percentile_normalization(sample["image"][3].numpy(), lower=0, upper=99)
        dsm = percentile_normalization(sample["image"][4].numpy(), lower=0, upper=99)
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask = draw_segmentation_masks(
            image=tensor, masks=sample["mask"].to(torch.bool), alpha=alpha, colors="red"
        ).permute(1, 2, 0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            preds = draw_segmentation_masks(
                image=tensor,
                masks=sample["prediction"].to(torch.bool),
                alpha=alpha,
                colors="red",
            ).permute(1, 2, 0)
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))
        for ax, data in zip(axs, [image, dtm, dsm, mask]):
            ax.imshow(data)
            ax.axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Digital Terrain Model (DTM)")
            axs[2].set_title("Digital Surface Model (DSM)")
            axs[3].set_title("Ground Truth")

        if showing_predictions:
            axs[4].imshow(preds)
            axs[4].axis("off")

            if show_titles:
                axs[4].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        return fig
