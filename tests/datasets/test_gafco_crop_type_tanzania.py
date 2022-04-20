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
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, GAFCOCropTypeTanzania


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_tanzania_01", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestGAFCOCropTypeTanzania:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> GAFCOCropTypeTanzania:

        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)

        root = str(tmp_path)

        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "ref_african_crops_tanzania_01",
                "ref_african_crops_tanzania_01_labels.tar.gz",
            ),
            root,
        )
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "ref_african_crops_tanzania_01",
                "ref_african_crops_tanzania_01_source.tar.gz",
            ),
            root,
        )

        image_meta = {
            "filename": "ref_african_crops_tanzania_01_source.tar.gz",
            "directory": "ref_african_crops_tanzania_01_source",
            "md5": "3d42ebbae207704b66494367e7330571",
        }
        target_meta = {
            "filename": "ref_african_crops_tanzania_01_labels.tar.gz",
            "directory": "ref_african_crops_tanzania_01_labels",
            "md5": "d654c81d59ced4afad4eabf55e19c86a",
        }
        monkeypatch.setattr(GAFCOCropTypeTanzania, "image_meta", image_meta)
        monkeypatch.setattr(GAFCOCropTypeTanzania, "target_meta", target_meta)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return GAFCOCropTypeTanzania(
            root=root, transforms=transforms, download=True, api_key=""
        )

    def test_getitem(self, dataset: GAFCOCropTypeTanzania) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_corrupted(self, dataset: GAFCOCropTypeTanzania, tmp_path: Path) -> None:
        with open(
            os.path.join(tmp_path, "ref_african_crops_tanzania_01_labels.tar.gz"), "w"
        ) as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            GAFCOCropTypeTanzania(dataset.root, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            GAFCOCropTypeTanzania(str(tmp_path))

    def test_already_downloaded(self, dataset: GAFCOCropTypeTanzania) -> None:
        GAFCOCropTypeTanzania(root=dataset.root, download=True, api_key="")

    def test_invalid_query(self, dataset: GAFCOCropTypeTanzania) -> None:
        query = BoundingBox(100, 100, 100, 100, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            GAFCOCropTypeTanzania(bands=["B01", "B02"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="is an invalid band name."):
            GAFCOCropTypeTanzania(bands=("foo", "bar"))

    def test_plot(self, dataset: GAFCOCropTypeTanzania) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="test")
        plt.close()

    def test_plot_prediction(self, dataset: GAFCOCropTypeTanzania) -> None:
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_plot_rgb(self, dataset: GAFCOCropTypeTanzania) -> None:
        dataset = GAFCOCropTypeTanzania(root=dataset.root, bands=tuple(["B01"]))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[dataset.bounds], suptitle="Single Band")
