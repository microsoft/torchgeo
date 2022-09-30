# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import CloudCoverDetection


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_cloud_cover_detection_challenge_v1", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestCloudCoverDetection:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> CloudCoverDetection:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)

        test_image_meta = {
            "filename": "ref_cloud_cover_detection_challenge_v1_test_source.tar.gz",
            "md5": "542e64a6e39b53c84c6462ec1b989e43",
        }
        monkeypatch.setitem(CloudCoverDetection.image_meta, "test", test_image_meta)

        test_target_meta = {
            "filename": "ref_cloud_cover_detection_challenge_v1_test_labels.tar.gz",
            "md5": "e8d41de08744a9845e74fca1eee3d1d3",
        }
        monkeypatch.setitem(CloudCoverDetection.target_meta, "test", test_target_meta)

        root = str(tmp_path)
        split = "test"
        transforms = nn.Identity()

        return CloudCoverDetection(
            root=root,
            transforms=transforms,
            split=split,
            download=True,
            api_key="",
            checksum=True,
        )

    def test_invalid_band(self, dataset: CloudCoverDetection) -> None:
        invalid_bands = ["B09"]
        with pytest.raises(ValueError):
            CloudCoverDetection(
                root=dataset.root,
                split="test",
                download=False,
                api_key="",
                bands=invalid_bands,
            )

    def test_get_item(self, dataset: CloudCoverDetection) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_add(self, dataset: CloudCoverDetection) -> None:
        assert len(dataset) == 1

    def test_already_downloaded(self, dataset: CloudCoverDetection) -> None:
        CloudCoverDetection(root=dataset.root, split="test", download=True, api_key="")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CloudCoverDetection(str(tmp_path))

    def test_plot(self, dataset: CloudCoverDetection) -> None:
        dataset.plot(dataset[0], suptitle="Test")
        plt.close()

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        dataset.plot(sample, suptitle="Pred")
        plt.close()

    def test_plot_rgb(self, dataset: CloudCoverDetection) -> None:
        dataset = CloudCoverDetection(
            root=dataset.root,
            split="test",
            bands=list(["B08"]),
            download=True,
            api_key="",
        )
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle="Single Band")
