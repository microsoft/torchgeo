# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import shapely
import torch
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame
from lightning.pytorch import Trainer
from matplotlib.figure import Figure
from pyproj import CRS
from torch import Tensor

from torchgeo.datamodules import (
    GeoDataModule,
    MisconfigurationException,
    NonGeoDataModule,
)
from torchgeo.datasets import BoundingBox, GeoDataset, NonGeoDataset
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler

MINT = pd.Timestamp(2025, 4, 24)
MAXT = pd.Timestamp(2025, 4, 25)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self, split: str = 'train', length: int = 1, download: bool = False
    ) -> None:
        geometry = [shapely.box(0, 0, 1, 1)] * length
        index = pd.IntervalIndex([pd.Interval(MINT, MAXT)] * length, name='datetime')
        crs = CRS.from_epsg(4326)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)
        self.res = (1, 1)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        image = torch.arange(3 * 2 * 2, dtype=torch.float).view(3, 2, 2)
        return {'image': image, 'crs': self.index.crs, 'bounds': query}

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
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
    def __init__(
        self, split: str = 'train', length: int = 1, download: bool = False
    ) -> None:
        self.length = length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {'image': torch.arange(3 * 2 * 2, dtype=torch.float).view(3, 2, 2)}

    def __len__(self) -> int:
        return self.length

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        return plt.figure()


class CustomNonGeoDataModule(NonGeoDataModule):
    def __init__(self) -> None:
        super().__init__(CustomNonGeoDataset, 1, 0, download=True)

    def setup(self, stage: str) -> None:
        super().setup(stage)

        if stage in ['predict']:
            self.predict_dataset = CustomNonGeoDataset()


class TestGeoDataModule:
    @pytest.fixture(params=[SamplerGeoDataModule, BatchSamplerGeoDataModule])
    def datamodule(self, request: SubRequest) -> CustomGeoDataModule:
        dm: CustomGeoDataModule = request.param()
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    @pytest.mark.parametrize('stage', ['fit', 'validate', 'test'])
    def test_setup(self, stage: str) -> None:
        dm = CustomGeoDataModule()
        dm.prepare_data()
        dm.setup(stage)

    def test_train(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup('fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device('cpu'), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_val(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup('validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device('cpu'), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_test(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup('test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device('cpu'), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_predict(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup('predict')
        if datamodule.trainer:
            datamodule.trainer.predicting = True
        batch = next(iter(datamodule.predict_dataloader()))
        batch = datamodule.transfer_batch_to_device(batch, torch.device('cpu'), 1)
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_plot(self, datamodule: CustomGeoDataModule) -> None:
        datamodule.setup('validate')
        datamodule.plot()
        plt.close()

    def test_no_datasets(self) -> None:
        dm = CustomGeoDataModule()
        msg = r'CustomGeoDataModule\.setup must define one of '
        msg += r"\('{0}_dataset', 'dataset'\)\."
        with pytest.raises(MisconfigurationException, match=msg.format('train')):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('val')):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('test')):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('predict')):
            dm.predict_dataloader()

    def test_no_samplers(self) -> None:
        dm = CustomGeoDataModule()
        dm.dataset = CustomGeoDataset()
        msg = r'CustomGeoDataModule\.setup must define one of '
        msg += r"\('{0}_batch_sampler', '{0}_sampler', 'batch_sampler', 'sampler'\)\."
        with pytest.raises(MisconfigurationException, match=msg.format('train')):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('val')):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('test')):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('predict')):
            dm.predict_dataloader()

    def test_zero_length_dataset(self) -> None:
        dm = CustomGeoDataModule()
        dm.dataset = CustomGeoDataset(length=0)
        msg = r'CustomGeoDataModule\.dataset has length 0.'
        with pytest.raises(MisconfigurationException, match=msg):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.predict_dataloader()

    def test_zero_length_sampler(self) -> None:
        dm = CustomGeoDataModule()
        dm.dataset = CustomGeoDataset()
        dm.sampler = RandomGeoSampler(dm.dataset, 1, 1)
        dm.sampler.length = 0
        msg = r'CustomGeoDataModule\.sampler has length 0.'
        with pytest.raises(MisconfigurationException, match=msg):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.predict_dataloader()


class TestNonGeoDataModule:
    @pytest.fixture
    def datamodule(self) -> CustomNonGeoDataModule:
        dm = CustomNonGeoDataModule()
        dm.trainer = Trainer(accelerator='cpu', max_epochs=1)
        return dm

    @pytest.mark.parametrize('stage', ['fit', 'validate', 'test', 'predict'])
    def test_setup(self, stage: str) -> None:
        dm = CustomNonGeoDataModule()
        dm.prepare_data()
        dm.setup(stage)

    def test_train(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup('fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_val(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup('validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_test(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup('test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_predict(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup('predict')
        if datamodule.trainer:
            datamodule.trainer.predicting = True
        batch = next(iter(datamodule.predict_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)

    def test_plot(self, datamodule: CustomNonGeoDataModule) -> None:
        datamodule.setup('validate')
        datamodule.plot()
        plt.close()

    def test_no_datasets(self) -> None:
        dm = CustomNonGeoDataModule()
        msg = r'CustomNonGeoDataModule\.setup must define one of '
        msg += r"\('{0}_dataset', 'dataset'\)\."
        with pytest.raises(MisconfigurationException, match=msg.format('train')):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('val')):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('test')):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg.format('predict')):
            dm.predict_dataloader()

    def test_zero_length_dataset(self) -> None:
        dm = CustomNonGeoDataModule()
        dm.dataset = CustomNonGeoDataset(length=0)
        msg = r'CustomNonGeoDataModule\.dataset has length 0.'
        with pytest.raises(MisconfigurationException, match=msg):
            dm.train_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.val_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.test_dataloader()
        with pytest.raises(MisconfigurationException, match=msg):
            dm.predict_dataloader()
