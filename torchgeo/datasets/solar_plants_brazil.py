# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SolarPlantsBrazil dataset."""

import os
import glob
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import rasterio
from matplotlib.figure import Figure
from huggingface_hub import snapshot_download

from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import Path


class SolarPlantsBrazil(NonGeoDataset):
    """Solar Plants Brazil dataset (semantic segmentation for photovoltaic detection).

    The `Solar Plants Brazil <https://huggingface.co/datasets/FederCO23/solar-plants-brazil>`__
    dataset provides satellite imagery and pixel-level annotations for detecting photovoltaic
    solar power stations.

    Dataset features:
    * 272 RGB+NIR GeoTIFF images (256x256 pixels)
    * Binary masks indicating presence of solar panels (1 = panel, 0 = background)
    * Organized into `train`, `val`, and `test` splits
    * Float32 GeoTIFF files for both input and mask images
    * Spatial metadata included (CRS, bounding box), but not used directly for training

    Folder structure:
        root/
            train/
                input/img(123).tif
                labels/target(123).tif
            val/
            test/

    Access:
    * Dataset is hosted on Hugging Face: https://huggingface.co/datasets/FederCO23/solar-plants-brazil
    * Code and preprocessing steps available at: https://github.com/FederCO23/UCSD_MLBootcamp_Capstone

    If you use this dataset, please cite or reference the project repository.

    .. versionadded:: 0.6
    """

    url = "https://huggingface.co/datasets/FederCO23/solar-plants-brazil"
    bands = ("Red", "Green", "Blue", "NIR")

    
    citation = """\
@misc{solarplantsbrazil2024,
  author       = {Federico Bessi},
  title        = {Solar Plants Brazil: A Semantic Segmentation Dataset for Photovoltaic Panel Detection},
  year         = {2024},
  howpublished = {Hugging Face Datasets},
  url          = {https://huggingface.co/datasets/FederCO23/solar-plants-brazil},
  note         = {Preprocessing and training code available at https://github.com/FederCO23/UCSD_MLBootcamp_Capstone}
}
"""

    def __init__(
        self,
        root: Path = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
    ) -> None:
        assert split in ["train", "val", "test"]

        self.root = root
        self.transforms = transforms
        self.dataset_path = os.path.join(self.root, split)
        self.split = split
        self.download = download

        self._verify()

        self.image_paths = sorted(glob.glob(os.path.join(self.dataset_path, "input", "img(*).tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.dataset_path, "labels", "target(*).tif")))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No input images found in {self.dataset_path}/input/")
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between image and mask files"

    def _verify(self) -> None:
        """Verify the dataset exists or download it."""
        if os.path.exists(self.dataset_path) and os.listdir(self.dataset_path):
            return

        if not self.download:
            raise RuntimeError(
                f"Dataset not found at {self.dataset_path}. Use download=True to fetch it."
            )

        self._download()


    def _download(self) -> None:
        """Download the dataset from Hugging Face."""
        snapshot_download(
            repo_id="FederCO23/solar-plants-brazil",
            repo_type="dataset",
            local_dir=self.root,
            token=False,
        )


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image = self._load_image(self.image_paths[index])
        mask = self._load_mask(self.mask_paths[index])
        sample = {"image": image, "mask": mask}
        if self.transforms:
            sample = self.transforms(sample)
        return sample


    def __len__(self) -> int:
        return len(self.image_paths)


    def _load_image(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)  # Shape: (bands, height, width)
        return torch.from_numpy(arr)


    def _load_mask(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.uint8)
        bin_mask = (arr > 0).astype(np.uint8)
        return torch.from_numpy(bin_mask).unsqueeze(0).long()


    def plot(
            self, 
            sample: dict[str, torch.Tensor], 
            suptitle: str | None = None
        ) -> Figure:
        """Plot a sample from the SolarPlantsBrazil dataset.

        Args:
            sample: A dictionary with 'image' and 'mask' tensors.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered image and mask.
        """
        image = sample["image"]
        mask = sample["mask"]

        # Use RGB only
        if image.shape[0] == 4:
            image = image[:3]

        # Normalize for display
        image_np = image.numpy()
        image_np = image_np / np.max(image_np)
        image_np = np.transpose(image_np, (1, 2, 0))

        mask_np = mask.squeeze().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("RGB Image")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Mask")
        axs[1].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()
        return fig

