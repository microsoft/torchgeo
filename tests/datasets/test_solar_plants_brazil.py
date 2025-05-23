# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import pytest
import torch
from torchgeo.datasets import SolarPlantsBrazil

import matplotlib.pyplot as plt
from PIL import Image


def create_dummy_dataset_structure(root: str) -> None:
    # Build train/input and train/labels structure
    input_dir = os.path.join(root, "train", "input")
    label_dir = os.path.join(root, "train", "labels")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Create dummy 4-band image and 1-band binary mask
    image = torch.randint(0, 255, (256, 256, 4), dtype=torch.uint8).numpy()
    mask = (torch.rand(256, 256) > 0.5).to(torch.uint8).numpy()

    Image.fromarray(image).save(os.path.join(input_dir, "img(1).tif"))
    Image.fromarray(mask).save(os.path.join(label_dir, "target(1).tif"))


def test_solar_plants_brazil_getitem_and_len() -> None:
    with tempfile.TemporaryDirectory() as root:
        create_dummy_dataset_structure(root)
        dataset = SolarPlantsBrazil(root=root, split="train")
        assert len(dataset) == 1

        sample = dataset[0]
        assert isinstance(sample, dict)
        assert "image" in sample and "mask" in sample
        assert sample["image"].shape == (4, 256, 256)
        assert sample["mask"].shape == (1, 256, 256)


def test_solar_plants_brazil_plot() -> None:
    with tempfile.TemporaryDirectory() as root:
        create_dummy_dataset_structure(root)
        dataset = SolarPlantsBrazil(root=root, split="train")
        sample = dataset[0]

        fig = dataset.plot(sample, suptitle="Test Sample")
        assert fig is not None
        plt.close(fig)
