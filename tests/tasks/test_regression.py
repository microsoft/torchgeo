# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch import nn
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, TropicalCycloneDataModule
from torchgeo.datamodules.copernicus.biomass_s3 import (
    TARGET_SIZE,
    CopernicusBenchBiomassS3DataModule,
)
from torchgeo.datasets import (
    CopernicusBenchBiomassS3,
    RGBBandsMissingError,
    TropicalCyclone,
)
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.tasks import PixelwiseRegression, Regression

from .test_classification import ClassificationTestModel


class PixelwiseRegressionTestModel(Module):
    def __init__(self, in_channels: int = 3, classes: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


class RegressionTestModel(ClassificationTestModel):
    def __init__(self, in_chans: int = 3, num_classes: int = 1, **kwargs: Any) -> None:
        super().__init__(in_chans=in_chans, num_classes=num_classes)


class PredictRegressionDataModule(TropicalCycloneDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = TropicalCyclone(split='test', **self.kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    return None


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class TestRegression:
    @classmethod
    def create_model(*args: Any, **kwargs: Any) -> Module:
        return RegressionTestModel(**kwargs)

    @pytest.mark.parametrize(
        'name',
        [
            'cowc_counting',
            'cyclone',
            'digital_typhoon_id',
            'digital_typhoon_time',
            'sustainbench_crop_yield',
            'skippd',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        if name in ['skippd', 'digital_typhoon_id', 'digital_typhoon_time']:
            pytest.importorskip('h5py', minversion='3.10')

        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(timm, 'create_model', self.create_model)

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_weight_file(self, checkpoint: str) -> None:
        with pytest.warns(UserWarning):
            Regression(model='resnet18', weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            Regression(
                model=mocked_weights.meta['model'],
                weights=mocked_weights,
                in_channels=mocked_weights.meta['in_chans'],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            Regression(
                model=mocked_weights.meta['model'],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta['in_chans'],
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        Regression(
            model=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        Regression(
            model=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(TropicalCycloneDataModule, 'plot', plot)
        datamodule = TropicalCycloneDataModule(
            root='tests/data/cyclone', batch_size=1, num_workers=0
        )
        model = Regression(model='resnet18')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(TropicalCycloneDataModule, 'plot', plot_missing_bands)
        datamodule = TropicalCycloneDataModule(
            root='tests/data/cyclone', batch_size=1, num_workers=0
        )
        model = Regression(model='resnet18')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictRegressionDataModule(
            root='tests/data/cyclone', batch_size=1, num_workers=0
        )
        model = Regression(model='resnet18')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize(
        'model_name', ['resnet18', 'efficientnetv2_s', 'vit_base_patch16_224']
    )
    def test_freeze_backbone(self, model_name: str) -> None:
        model = Regression(model=model_name, freeze_backbone=True)
        assert not all(param.requires_grad for param in model.model.parameters())
        assert all(
            param.requires_grad for param in model.model.get_classifier().parameters()
        )


class TestPixelwiseRegression:
    @classmethod
    def create_model(*args: Any, **kwargs: Any) -> Module:
        return PixelwiseRegressionTestModel(**kwargs)

    @pytest.mark.parametrize(
        'name',
        [
            'inria_unet',
            'inria_deeplab',
            'inria_fcn',
            'inria_segformer',
            'inria_upernet',
            'inria_dpt',
            'copernicus_biomass_s3',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(smp, 'Unet', self.create_model)
        monkeypatch.setattr(smp, 'DeepLabV3Plus', self.create_model)
        monkeypatch.setattr(smp, 'UPerNet', self.create_model)
        monkeypatch.setattr(smp, 'Segformer', self.create_model)
        monkeypatch.setattr(smp, 'DPT', self.create_model)

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    def test_biomass_s3_existing_transform_is_composed(self) -> None:
        root = 'tests/data/copernicus/l3_biomass_s3'
        bands = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')
        count = 0
        shape: tuple[int, int] | None = None

        def existing_transform(
            sample: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            nonlocal count, shape
            count += 1
            shape = (int(sample['image'].shape[-2]), int(sample['image'].shape[-1]))
            sample['transformed'] = torch.tensor(True)
            return sample

        datamodule = CopernicusBenchBiomassS3DataModule(
            root=root,
            batch_size=1,
            num_workers=0,
            bands=bands,
            transforms=existing_transform,
        )

        composed = datamodule.kwargs['transforms']
        assert composed is not existing_transform

        dataset = CopernicusBenchBiomassS3(
            root=root, split='train', bands=bands, transforms=composed
        )
        sample = dataset[0]

        assert count == 1
        assert shape == TARGET_SIZE
        assert sample['transformed']
        assert sample['image'].shape[-2:] == TARGET_SIZE

    def test_biomass_s3_nonfinite_imagery_is_sanitized(self) -> None:
        datamodule = CopernicusBenchBiomassS3DataModule(
            root='tests/data/copernicus/l3_biomass_s3', batch_size=1, num_workers=0
        )
        batch = {
            'image': torch.tensor([[[[float('nan'), float('inf')]]]]),
            'mask': torch.ones(1, 1, 2),
        }

        batch = datamodule.on_after_batch_transfer(batch, dataloader_idx=0)

        assert torch.isfinite(batch['image']).all()
        assert batch['image'].eq(0).all()

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_weight_file(self, checkpoint: str) -> None:
        PixelwiseRegression(model='unet', backbone='resnet18', weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        PixelwiseRegression(
            model='unet',
            backbone=mocked_weights.meta['model'],
            weights=mocked_weights,
            in_channels=mocked_weights.meta['in_chans'],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        PixelwiseRegression(
            model='unet',
            backbone=mocked_weights.meta['model'],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        PixelwiseRegression(
            model='unet',
            backbone=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        PixelwiseRegression(
            model='unet',
            backbone=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.parametrize(
        'model_name', ['unet', 'deeplabv3+', 'segformer', 'upernet']
    )
    @pytest.mark.parametrize(
        'backbone', ['resnet18', 'mobilenet_v2', 'efficientnet-b0']
    )
    def test_freeze_backbone(self, model_name: str, backbone: str) -> None:
        model = PixelwiseRegression(
            model=model_name, backbone=backbone, freeze_backbone=True
        )
        assert all(
            param.requires_grad is False for param in model.model.encoder.parameters()
        )
        assert all(param.requires_grad for param in model.model.decoder.parameters())
        assert all(
            param.requires_grad for param in model.model.segmentation_head.parameters()
        )

    @pytest.mark.parametrize(
        'model_name', ['unet', 'deeplabv3+', 'segformer', 'upernet']
    )
    def test_freeze_decoder(self, model_name: str) -> None:
        model = PixelwiseRegression(
            model=model_name, backbone='resnet18', freeze_decoder=True
        )
        assert all(
            param.requires_grad is False for param in model.model.decoder.parameters()
        )
        assert all(param.requires_grad for param in model.model.encoder.parameters())
        assert all(
            param.requires_grad for param in model.model.segmentation_head.parameters()
        )

    def test_vit_backbone(self) -> None:
        PixelwiseRegression(model='dpt', backbone='tu-vit_base_patch16_224')
