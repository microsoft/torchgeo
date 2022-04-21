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

from torchgeo.datasets import (
    BoundingBox,
    CropTypeKenyaPlantVillage,
    CropTypeTanzaniaGAFCO,
    CropTypeUgandaDalbergDataInsight,
)

IMAGERY_MD5 = "b789c59caed2c4ab96a47187d1856351"
KENYA_MD5 = "dd057c4ae203255061eaeeff5d527405"
TANZANIA_MD5 = "b1568959e416ed9b895af0a5a81a758c"
UGANDA_MD5 = "eb3ac3302cdf5ca9e54b2f8df1bd181e"


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_kenya_01", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path, recursive=True):
            shutil.copy(tarball, output_dir)


def fetch(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestCropTypeKenyaPlantVillage:
    @pytest.fixture(params=[True, False])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> CropTypeKenyaPlantVillage:

        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.3.1")
        monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch)

        root = str(tmp_path)
        # copy general imagery
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "radiant_ml_crop_type_datasets",
                "ref_african_crops_datasets.tar.gz",
            ),
            root,
        )

        # copy dataset specific labels
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "radiant_ml_crop_type_datasets",
                "ref_african_crops_kenya_01_labels.tar.gz",
            ),
            root,
        )

        image_meta = {
            "filename": "ref_african_crops_datasets.tar.gz",
            "directory": "ref_african_crops_datasets",
            "md5": IMAGERY_MD5,
        }
        target_meta = {
            "filename": "ref_african_crops_kenya_01_labels.tar.gz",
            "directory": "ref_african_crops_kenya_01_labels",
            "md5": KENYA_MD5,
        }
        monkeypatch.setattr(CropTypeKenyaPlantVillage, "image_meta", image_meta)
        monkeypatch.setattr(CropTypeKenyaPlantVillage, "target_meta", target_meta)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        cache = request.param
        return CropTypeKenyaPlantVillage(
            root=root, transforms=transforms, api_key="", cache=cache
        )

    def test_getitem(self, dataset: CropTypeKenyaPlantVillage) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_no_stac_json_found(self, dataset: CropTypeKenyaPlantVillage) -> None:
        with tarfile.open(
            os.path.join(dataset.root, "ref_african_crops_datasets.tar.gz")
        ) as f:
            f.extractall()
        stac_list = glob.glob(
            os.path.join(dataset.root, "**", "**", "stac.json"), recursive=True
        )
        for stac_file in stac_list:
            if os.path.exists(stac_file):
                os.remove(stac_file)
        with pytest.raises(FileNotFoundError, match="No stac.json files found in"):
            CropTypeKenyaPlantVillage(root=dataset.root, download=False, api_key="")

    # def test_download(self, tmp_path: Path) -> None:
    #     CropTypeKenyaPlantVillage(root=tmp_path, download=True, api_key="")

    def test_different_crs(self, dataset: CropTypeKenyaPlantVillage) -> None:
        crs = CRS.from_epsg(32736)
        ds = CropTypeKenyaPlantVillage(root=dataset.root, crs=crs)
        x = ds[ds.bounds]
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_corrupted(self, tmp_path: Path) -> None:
        path = os.path.join(tmp_path, "ref_african_crops_kenya_01_labels.tar.gz")
        with open(path, "w") as f:
            f.write("bad")

        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            CropTypeKenyaPlantVillage(root=str(tmp_path), checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            CropTypeKenyaPlantVillage(root=str(tmp_path))

    def test_already_downloaded(self, dataset: CropTypeKenyaPlantVillage) -> None:
        CropTypeKenyaPlantVillage(root=dataset.root, download=True, api_key="")

    def test_invalid_query(self, dataset: CropTypeKenyaPlantVillage) -> None:
        query = BoundingBox(100, 100, 100, 100, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            CropTypeKenyaPlantVillage(bands=["B01", "B02"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="is an invalid band name."):
            CropTypeKenyaPlantVillage(bands=("foo", "bar"))

    def test_plot(self, dataset: CropTypeKenyaPlantVillage) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, time_step=0, suptitle="test")
        plt.close()

    def test_plot_prediction(self, dataset: CropTypeKenyaPlantVillage) -> None:
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_plot_rgb(self, dataset: CropTypeKenyaPlantVillage) -> None:
        dataset = CropTypeKenyaPlantVillage(root=dataset.root, bands=tuple(["B01"]))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[dataset.bounds], suptitle="Single Band")


class TestCropTypeTanzaniaGAFCO:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> CropTypeTanzaniaGAFCO:

        root = str(tmp_path)

        # copy general imagery
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "radiant_ml_crop_type_datasets",
                "ref_african_crops_datasets.tar.gz",
            ),
            root,
        )

        # copy specific labels
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "radiant_ml_crop_type_datasets",
                "ref_african_crops_tanzania_01_labels.tar.gz",
            ),
            root,
        )

        image_meta = {
            "filename": "ref_african_crops_datasets.tar.gz",
            "directory": "ref_african_crops_datasets",
            "md5": IMAGERY_MD5,
        }
        target_meta = {
            "filename": "ref_african_crops_tanzania_01_labels.tar.gz",
            "directory": "ref_african_crops_tanzania_01_labels",
            "md5": TANZANIA_MD5,
        }
        monkeypatch.setattr(CropTypeTanzaniaGAFCO, "image_meta", image_meta)
        monkeypatch.setattr(CropTypeTanzaniaGAFCO, "target_meta", target_meta)
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return CropTypeTanzaniaGAFCO(root=root, transforms=transforms, api_key="")

    def test_getitem(self, dataset: CropTypeTanzaniaGAFCO) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)


class TestCropTypyUgandaDalbergDataInsight:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> CropTypeUgandaDalbergDataInsight:

        root = str(tmp_path)

        # copy general imagery
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "radiant_ml_crop_type_datasets",
                "ref_african_crops_datasets.tar.gz",
            ),
            root,
        )

        # copy specific labels
        shutil.copy(
            os.path.join(
                "tests",
                "data",
                "radiant_ml_crop_type_datasets",
                "ref_african_crops_uganda_01_labels.tar.gz",
            ),
            root,
        )

        image_meta = {
            "filename": "ref_african_crops_datasets.tar.gz",
            "directory": "ref_african_crops_datasets",
            "md5": IMAGERY_MD5,
        }
        target_meta = {
            "filename": "ref_african_crops_uganda_01_labels.tar.gz",
            "directory": "ref_african_crops_uganda_01_labels",
            "md5": UGANDA_MD5,
        }
        monkeypatch.setattr(CropTypeUgandaDalbergDataInsight, "image_meta", image_meta)
        monkeypatch.setattr(
            CropTypeUgandaDalbergDataInsight, "target_meta", target_meta
        )
        transforms = nn.Identity()  # type: ignore[no-untyped-call]
        return CropTypeUgandaDalbergDataInsight(
            root=root, transforms=transforms, api_key=""
        )

    def test_getitem(self, dataset: CropTypeUgandaDalbergDataInsight) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)
