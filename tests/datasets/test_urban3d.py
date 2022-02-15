# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import Urban3DChallenge


class TestUrban3DChallenge:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(self, request: SubRequest) -> Urban3DChallenge:
        root = os.path.join("tests", "data", "urban3d")
        split = request.param
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return Urban3DChallenge(root, split, transforms)

    def test_getitem(self, dataset: Urban3DChallenge) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].dtype == torch.float  # type: ignore[attr-defined]
        assert x["mask"].dtype == torch.long  # type: ignore[attr-defined]
        assert x["image"].ndim == 3
        assert x["image"].shape[0] == 5
        assert x["mask"].ndim == 2

    def test_len(self, dataset: Urban3DChallenge) -> None:
        assert len(dataset) == 4

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = f"Dataset not found in {str(tmp_path)} directory, "
        f"specify a different {str(tmp_path)} directory."
        with pytest.raises(RuntimeError, match=err):
            Urban3DChallenge(str(tmp_path))

    def test_plot(self, dataset: Urban3DChallenge) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction"] = x["mask"].clone()
        dataset.plot(x)
        plt.close()
