# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import re

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from torchgeo.datamodules.utils import dataset_split, group_shuffle_split


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


def test_group_shuffle_split() -> None:
    with open("tests/datamodules/train_test_indices.json") as f:
        data = json.load(f)
        train_indices = data["train_indices"]
        test_indices = data["test_indices"]
    np.random.seed(0)
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    groups = np.random.randint(0, 26, size=(1000))
    groups = alphabet[groups]

    with pytest.raises(ValueError, match="You must specify `train_size` *"):
        group_shuffle_split(groups, train_size=None, test_size=None)
    with pytest.raises(ValueError, match="`train_size` and `test_size` must sum to 1."):
        group_shuffle_split(groups, train_size=0.2, test_size=1.0)
    with pytest.raises(
        ValueError,
        match=re.escape("`train_size` and `test_size` must be in the range (0,1)."),
    ):
        group_shuffle_split(groups, train_size=-0.2, test_size=1.2)
    with pytest.raises(ValueError, match="26 groups were found, however the current *"):
        group_shuffle_split(groups, train_size=None, test_size=0.999)

    test_cases = [(None, 0.2, 42), (0.8, None, 42)]

    for train_size, test_size, random_state in test_cases:
        train_indices1, test_indices1 = group_shuffle_split(
            groups,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        # Check that the results are the same as the saved results
        assert np.array_equal(train_indices, train_indices1)
        assert np.array_equal(test_indices, test_indices1)

        assert len(set(train_indices1) & set(test_indices1)) == 0
        assert len(set(groups[train_indices1])) == 21
