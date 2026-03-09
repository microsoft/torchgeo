# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Script to create test data for SentinelKilnDB dataset."""

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def create_dummy_image_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    """Create a dummy RGB image and return as PNG bytes."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img, mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return buffer.getvalue()


def create_yolo_aa_label(num_boxes: int, num_classes: int = 3) -> np.ndarray:
    """Create YOLO axis-aligned bounding box labels.

    Format: class_id x_center y_center width height (normalized 0-1)
    Returns numpy array of label strings.
    """
    lines = []
    for _ in range(num_boxes):
        class_id = np.random.randint(0, num_classes)
        x_center = np.random.uniform(0.2, 0.8)
        y_center = np.random.uniform(0.2, 0.8)
        width = np.random.uniform(0.05, 0.2)
        height = np.random.uniform(0.05, 0.2)
        lines.append(
            f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        )
    return np.array(lines, dtype=object)


def create_yolo_obb_label(num_boxes: int, num_classes: int = 3) -> np.ndarray:
    """Create YOLO oriented bounding box labels.

    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0-1)
    Returns numpy array of label strings.
    """
    lines = []
    for _ in range(num_boxes):
        class_id = np.random.randint(0, num_classes)
        # Create a simple rotated rectangle
        cx = np.random.uniform(0.3, 0.7)
        cy = np.random.uniform(0.3, 0.7)
        w = np.random.uniform(0.05, 0.15)
        h = np.random.uniform(0.05, 0.15)
        angle = np.random.uniform(0, np.pi / 4)

        # Compute corners
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        rotated = []
        for dx, dy in corners:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            rotated.extend([rx, ry])

        coords = ' '.join(f'{c:.6f}' for c in rotated)
        lines.append(f'{class_id} {coords}')
    return np.array(lines, dtype=object)


def create_test_data(root: Path) -> None:
    """Create SentinelKilnDB test dataset with parquet files."""
    splits = ['train', 'val', 'test']
    samples_per_split = {'train': 4, 'val': 2, 'test': 2}

    np.random.seed(42)

    for split in splits:
        rows = []
        num_samples = samples_per_split[split]

        for i in range(num_samples):
            # Create image bytes
            img_bytes = create_dummy_image_bytes()

            # Create labels with different cases for coverage
            if i == 0:
                # Empty labels (negative sample)
                yolo_aa = np.array([], dtype=object)
                yolo_obb = np.array([], dtype=object)
            elif i == 1 and split == 'train':
                # Malformed labels (too few parts) - to test skip logic
                yolo_aa = np.array(['0 0.5 0.5'], dtype=object)  # Only 3 parts, need 5
                yolo_obb = np.array(
                    ['0 0.1 0.2 0.3'], dtype=object
                )  # Only 4 parts, need 9
            else:
                num_boxes = np.random.randint(1, 4)
                yolo_aa = create_yolo_aa_label(num_boxes)
                yolo_obb = create_yolo_obb_label(num_boxes)

            row = {
                'image': img_bytes,
                'yolo_aa_label': yolo_aa,
                'yolo_obb_label': yolo_obb,
            }
            rows.append(row)

        # Create DataFrame and save as parquet
        df = pd.DataFrame(rows)
        parquet_path = root / f'{split}.parquet'
        df.to_parquet(parquet_path)
        print(f'Created {parquet_path}')


if __name__ == '__main__':
    root = Path(os.path.dirname(__file__))
    create_test_data(root)
