# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re

import numpy as np
import pytest

from torchgeo.datamodules.utils import group_shuffle_split


def test_group_shuffle_split() -> None:
    train_indices = [0, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    test_indices = [1, 3, 4, 12]
    np.random.seed(0)
    alphabet = np.array(list("abc"))
    groups = np.random.randint(0, 3, size=(15))
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
    with pytest.raises(ValueError, match="3 groups were found, however the current *"):
        group_shuffle_split(groups, train_size=None, test_size=0.999)

    test_cases = [(None, 0.2, 42), (0.8, None, 42)]

    for train_size, test_size, random_state in test_cases:
        train_indices1, test_indices1 = group_shuffle_split(
            groups,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        # Check that the results are the same as expected
        assert np.array_equal(train_indices, train_indices1)
        assert np.array_equal(test_indices, test_indices1)

        assert len(set(train_indices1) & set(test_indices1)) == 0
        assert len(set(groups[train_indices1])) == 2
