# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, GAFCOCropTypeTanzania


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_tanzania_01", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path, recursive=True):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestGAFCOCropTypeTanzania:
    @pytest.fixture(params=[True, False])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
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
            "md5": "0cc19f25993c1526dfeea9c8d681005d",
        }
        target_meta = {
            "filename": "ref_african_crops_tanzania_01_labels.tar.gz",
            "directory": "ref_african_crops_tanzania_01_labels",
            "md5": "076ecebb3006f969d479200e6179a78c",
        }
        monkeypatch.setattr(GAFCOCropTypeTanzania, "image_meta", image_meta)
        monkeypatch.setattr(GAFCOCropTypeTanzania, "target_meta", target_meta)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        cache = request.param
        return GAFCOCropTypeTanzania(
            root=root, transforms=transforms, api_key="", cache=cache
        )

    # def test_download(self, tmp_path: Path) -> None:
    #     GAFCOCropTypeTanzania(root=tmp_path, download=True)

    def test_getitem(self, dataset: GAFCOCropTypeTanzania) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            GAFCOCropTypeTanzania(root=str(tmp_path))

    def test_no_stac_json_found(self, dataset: GAFCOCropTypeTanzania) -> None:
        with tarfile.open(
            os.path.join(dataset.root, "ref_african_crops_tanzania_01_source.tar.gz")
        ) as f:
            f.extractall()
        stac_list = glob.glob(
            os.path.join(dataset.root, "**", "**", "stac.json"), recursive=True
        )
        for f in stac_list:
            if os.path.exists(f):
                os.remove(f)
        with pytest.raises(FileNotFoundError, match="No stac.json files found in"):
            GAFCOCropTypeTanzania(root=dataset.root, download=False, api_key="")

    def test_different_crs(self, dataset: GAFCOCropTypeTanzania) -> None:
        crs = CRS.from_epsg(32636)
        ds = GAFCOCropTypeTanzania(root=dataset.root, crs=crs)
        x = ds[ds.bounds]
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_corrupted(self, tmp_path: Path) -> None:
        path = os.path.join(tmp_path, "ref_african_crops_tanzania_01_labels.tar.gz")
        with open(path, "w") as f:
            f.write("bad")

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
        dataset.plot(x, time_step=0, suptitle="test")
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
