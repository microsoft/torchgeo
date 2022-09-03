# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch.utils.data import TensorDataset

from torchgeo.datamodules.utils import dataset_split


def test_dataset_split() -> None:
    num_samples = 24
    x = torch.ones(num_samples, 5)
    y = torch.randint(low=0, high=2, size=(num_samples,))
    ds = TensorDataset(x, y)

    # Test only train/val set split
    train_ds, val_ds = dataset_split(ds, val_pct=1 / 2)
    assert len(train_ds) == round(num_samples / 2)
    assert len(val_ds) == round(num_samples / 2)

    # Test train/val/test set split
    train_ds, val_ds, test_ds = dataset_split(ds, val_pct=1 / 3, test_pct=1 / 3)
    assert len(train_ds) == round(num_samples / 3)
    assert len(val_ds) == round(num_samples / 3)
    assert len(test_ds) == round(num_samples / 3)
