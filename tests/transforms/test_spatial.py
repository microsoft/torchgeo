# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import pytest
import torch

from torchgeo.transforms import SatSlideMix


def test_sat_slidemix() -> None:
    b, c, h, w = 2, 3, 64, 64
    gamma = 2
    batch = {
        'image': torch.randn(size=(b, c, h, w)),
        'mask': torch.randint(low=0, high=2, size=(b, 1, h, w)),
    }
    aug = K.AugmentationSequential(
        SatSlideMix(gamma=gamma, beta=(0.0, 1.0), p=1.0), data_keys=None
    )
    out = aug(batch)
    assert out['image'].shape == (b * gamma, c, h, w)
    assert out['mask'].shape == (b * gamma, 1, h, w)

    # Catch that assertion is thrown if gamma is not a positive integer
    with pytest.raises(AssertionError, match='gamma must be a positive integer'):
        SatSlideMix(gamma=0)
