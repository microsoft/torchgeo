# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict

import pytest
import torch
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

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        return {"image": torch.arange(3 * 2 * 2).view(3, 2, 2)}


class CustomGeoDataModule(GeoDataModule):
    def __init__(self) -> None:
        super().__init__(CustomGeoDataset, 1, 1, 1, 0, download=True)


class SamplerGeoDatModule(CustomGeoDataModule):
    def setup(self, stage: str) -> None:
        self.dataset = CustomGeoDataset()
        self.train_sampler = RandomGeoSampler(self.dataset, 1, 1)
        self.val_sampler = RandomGeoSampler(self.dataset, 1, 1)
        self.test_sampler = RandomGeoSampler(self.dataset, 1, 1)
        self.predict_sampler = RandomGeoSampler(self.dataset, 1, 1)


class BatchSamplerGeoDatModule(CustomGeoDataModule):
    def setup(self, stage: str) -> None:
        self.dataset = CustomGeoDataset()
        self.train_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)
        self.val_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)
        self.test_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)
        self.predict_batch_sampler = RandomBatchGeoSampler(self.dataset, 1, 1, 1)


class CustomNonGeoDataset(NonGeoDataset):
    def __init__(self, split: str = "train", download: bool = False) -> None:
        pass

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {"image": torch.arange(3 * 2 * 2).view(3, 2, 2)}

    def __len__(self) -> int:
        return 1


class CustomNonGeoDataModule(NonGeoDataModule):
    def __init__(self) -> None:
        super().__init__(CustomNonGeoDataset, 1, 0, download=True)


class TestGeoDataModule:
    @pytest.mark.parametrize("stage", ["fit", "validate", "test"])
    def test_setup(self, stage: str) -> None:
        dm = CustomGeoDataModule()
        dm.prepare_data()
        dm.setup(stage)

    def test_sampler(self) -> None:
        dm = SamplerGeoDatModule()
        dm.setup("fit")
        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()
        dm.predict_dataloader()

    def test_batch_sampler(self) -> None:
        dm = BatchSamplerGeoDatModule()
        dm.setup("fit")
        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()
        dm.predict_dataloader()

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
    @pytest.mark.parametrize("stage", ["fit", "validate", "test"])
    def test_setup(self, stage: str) -> None:
        dm = CustomNonGeoDataModule()
        dm.prepare_data()
        dm.setup(stage)

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
