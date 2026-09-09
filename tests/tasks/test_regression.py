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
from torchgeo.datasets import RGBBandsMissingError, TropicalCyclone
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
    @pytest.mark.parametrize('pinball', [False, True])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool, pinball: bool
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

        if pinball:
            args.extend(['--model.init_args.loss', 'pinball'])

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
    @pytest.mark.parametrize('pinball', [False, True])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool, pinball: bool
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

        if pinball:
            args.extend(['--model.init_args.loss', 'pinball'])

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


class TestQuantileRegression:
    @pytest.fixture(params=[Regression, PixelwiseRegression])
    def task_class(
        self, request: pytest.FixtureRequest, monkeypatch: MonkeyPatch
    ) -> type[Regression]:
        monkeypatch.setattr(timm, 'create_model', TestRegression.create_model)
        monkeypatch.setattr(smp, 'Unet', TestPixelwiseRegression.create_model)
        return request.param

    def test_steps(self, task_class: type[Regression]) -> None:
        model = 'resnet18' if task_class is Regression else 'unet'
        task = task_class(model=model, loss='pinball', quantiles=[0.5, 0.9, 0.1])
        task.trainer = Trainer(accelerator='cpu', barebones=True)
        with torch.no_grad():
            for parameter in task.model.parameters():
                parameter.zero_()
            head = task.model.fc if task_class is Regression else task.model.conv1
            head.bias.copy_(torch.tensor([1.0, 5.0, -1.0]))
        shape = (2,) if task_class is Regression else (2, 2, 4)
        batch = {
            'image': torch.zeros(2, 3, 2, 4),
            task.target_key: torch.full(shape, 2.0),
        }

        loss = task.training_step(batch, 0)
        torch.testing.assert_close(loss, torch.tensor(1.1 / 3))
        loss.backward()
        torch.testing.assert_close(head.bias.grad, torch.tensor([-0.5, 0.1, -0.1]) / 3)
        task.validation_step(batch, 0)
        task.test_step(batch, 0)
        for metrics in [task.train_metrics, task.val_metrics, task.test_metrics]:
            for value in metrics.compute().values():
                torch.testing.assert_close(value, torch.tensor(1.0))

        predictions = task.predict_step(batch, 0)
        expected = torch.tensor([1.0, 5.0, -1.0]).view(1, 3, *([1] * (len(shape) - 1)))
        torch.testing.assert_close(predictions, expected.expand(2, 3, *shape[1:]))

    @pytest.mark.parametrize('num_outputs,quantiles', [(2, [0.5]), (1, [0.1, 0.9])])
    def test_invalid_config(
        self, task_class: type[Regression], num_outputs: int, quantiles: list[float]
    ) -> None:
        with pytest.raises(ValueError, match='requires num_outputs=1 and quantile 0.5'):
            task_class(
                model='unet',
                loss='pinball',
                num_outputs=num_outputs,
                quantiles=quantiles,
            )
