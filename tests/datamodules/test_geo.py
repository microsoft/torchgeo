# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from lightning.pytorch import Trainer
from rasterio.crs import CRS
from torch import Tensor

from torchgeo.datamodules import (
    GeoDataModule,
    MisconfigurationException,
    NonGeoDataModule,
)
from torchgeo.datasets import BoundingBox, GeoDataset, NonGeoDataset
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler


class CustomGeoDataset(GeoDataset):
    def __init__(self, split: str = "train", download: bool = False) -> None:
        super().__init__()
        self.index.insert(0, (0, 1, 2, 3, 4, 5))
        self.res = 1

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        image = torch.arange(3 * 2 * 2).view(3, 2, 2)
        return {"image": image, "crs": CRS.from_epsg(4326), "bbox": query}

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        return plt.figure()


class CustomGeoDataModule(GeoDataModule):
    def __init__(self) -> None:
        super().__init__(CustomGeoDataset, 1, 1, 1, 0, download=True)


class SamplerGeoDataModule(CustomGeoDataModule):
    def setup(self, stage: str) -> None:
        self.dataset = CustomGeoDataset()
        self.train_sampler = RandomGeoSampler(self.dataset, 1, 1)
        self.val_sampler = RandomGeoSampler(self.dataset, 1, 1)
        self.test_sampler = RandomGeoSampler(self.dataset, 1, 1)
        self.predict_sampler = RandomGeoSampler(self.dataset, 1, 1)


class BatchSamplerGeoDataModule(CustomGeoDataModule):
    def setup(self, stage: str) -> None:
        self.dataset = CustomGeoDataset()
        self.train_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)
        self.val_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)
        self.test_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)
        self.predict_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)


class CustomNonGeoDataset(NonGeoDataset):
    def __init__(self, split: str = "train", download: bool = False) -> None:
        pass

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"image": torch.arange(3 * 2 * 2).view(3, 2, 2)}

    def __len__(self) -> int:
        return 1

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        return plt.figure()


class CustomNonGeoDataModule(NonGeoDataModule):
    def __init__(self) -> None:
        super().__init__(CustomNonGeoDataset, 1, 0, download=True)

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if stage in ["predict"]:
            self.predict_dataset = CustomNonGeoDataset()


class TestGeoDataModule:
    @pytest.fixture(params=[SamplerGeoDataModule, BatchSamplerGeoDataModule])
    def datamodule(self, request: SubRequest) -> CustomGeoDataModule:
        dm: CustomGeoDataModule = request.param()
        dm.trainer = Trainer(accelerator="cpu", max_epochs=1)
        return dm

    @pytest.mark.parametrize("stage", ["fit", "validate", "test"])
    def test_setup(self, stage: str) -> None:
        dm = CustomGeoDataModule()
        dm.prepare_data()
        dm.setup(stage)

    def test_train(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup("fit")
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_val(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup("validate")
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_test(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup("test")
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_predict(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup("predict")
        if datamodule.trainer:
            datamodule.trainer.predicting = True
        batch = next(iter(datamodule.predict_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device("cpu"), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_plot(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup("validate")
        datamodule.plot()
        plt.close()

    def test_no_datasets(self) -> None:
        dm = CustomGeoDataModule()
        msg = "CustomGeoDataModule.setup does not define a '{}_dataset'"
        with pytest.raises(MisconfigurationException, match=msg.format("train")):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format("val")):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format("test")):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format("predict")):
            dm.predict_dataloader()


class TestNonGeoDataModule:
    @pytest.fixture
    def datamodule(self) -> CustomNonGeoDataModule:
        dm = CustomNonGeoDataModule()
        dm.trainer = Trainer(accelerator="cpu", max_epochs=1)
        return dm

    @pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
    def test_setup(self, stage: str) -> None:
        dm = CustomNonGeoDataModule()
        dm.prepare_data()
        dm.setup(stage)

    def test_train(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup("fit")
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_val(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup("validate")
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_test(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup("test")
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_predict(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup("predict")
        if datamodule.trainer:
            datamodule.trainer.predicting = True
        batch = next(iter(datamodule.predict_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_plot(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup("validate")
        datamodule.plot()
        plt.close()

    def test_no_datasets(self) -> None:
        dm = CustomNonGeoDataModule()
        msg = "CustomNonGeoDataModule.setup does not define a '{}_dataset'"
        with pytest.raises(MisconfigurationException, match=msg.format("train")):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format("val")):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format("test")):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format("predict")):
            dm.predict_dataloader()
