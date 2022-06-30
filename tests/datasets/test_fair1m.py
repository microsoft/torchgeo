# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import FAIR1M


class TestFAIR1M:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch) -> FAIR1M:
        md5s = ["f278aba757de9079225db42107e09e30", "aca59017207141951b53e91795d8179e"]
        monkeypatch.setattr(FAIR1M, "md5s", md5s)
        root = os.path.join("tests", "data", "fair1m")
        transforms = nn.Identity()
        return FAIR1M(root, transforms)

    def test_getitem(self, dataset: FAIR1M) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["boxes"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3
        assert x["boxes"].shape[-2:] == (5, 2)
        assert x["label"].ndim == 1

    def test_len(self, dataset: FAIR1M) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: FAIR1M, tmp_path: Path) -> None:
        shutil.rmtree(str(tmp_path))
        shutil.copytree(dataset.root, str(tmp_path))
        FAIR1M(root=str(tmp_path))

    def test_already_downloaded_not_extracted(
        self, dataset: FAIR1M, tmp_path: Path
    ) -> None:
        for filename in dataset.filenames:
            filepath = os.path.join("tests", "data", "fair1m", filename)
            shutil.copy(filepath, str(tmp_path))
        FAIR1M(root=str(tmp_path), checksum=True)

    def test_corrupted(self, tmp_path: Path) -> None:
        filenames = ["images.zip", "labelXmls.zip"]
        for filename in filenames:
            with open(os.path.join(tmp_path, filename), "w") as f:
                f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            FAIR1M(root=str(tmp_path), checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        err = "Dataset not found in `root` directory, "
        "specify a different `root` directory."
        with pytest.raises(RuntimeError, match=err):
            FAIR1M(str(tmp_path))

    def test_plot(self, dataset: FAIR1M) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x["prediction_boxes"] = x["boxes"].clone()
        dataset.plot(x)
        plt.close()
