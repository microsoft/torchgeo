#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
from pathlib import Path

import numpy as np
from PIL import Image


def create_dummy_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    """Create small dummy image."""
    img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def create_metadata_file(
    path: Path, boxes: list[dict[str, list[int]]] | None = None
) -> None:
    """Create dummy metadata JSON file."""
    if boxes is None:
        boxes = [{'box': [1, 2, 3, 4]}]

    metadata = {'bounding_boxes': boxes}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)


def create_test_data(root: Path) -> None:
    """Create fMoW test dataset."""
    splits = ['train', 'val']

    for split in splits:
        sequence_dir = root / split / 'airport' / 'airport_0'
        sequence_dir.mkdir(parents=True, exist_ok=True)

        image_path = sequence_dir / 'airport_0_0_rgb.jpg'
        metadata_path = sequence_dir / 'airport_0_0_rgb.json'

        create_dummy_image(image_path)
        create_metadata_file(metadata_path)


if __name__ == '__main__':
    create_test_data(Path('.'))
