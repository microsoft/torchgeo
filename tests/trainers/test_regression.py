# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, TropicalCycloneDataModule
from torchgeo.datasets import RGBBandsMissingError, TropicalCyclone
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import (
    PixelwiseRegressionTask,
    RegressionTask,
    SpatioTemporalPixelwiseRegressionTask,
)

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


class TestRegressionTask:
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
            RegressionTask(model='resnet18', weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            RegressionTask(
                model=mocked_weights.meta['model'],
                weights=mocked_weights,
                in_channels=mocked_weights.meta['in_chans'],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            RegressionTask(
                model=mocked_weights.meta['model'],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta['in_chans'],
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        RegressionTask(
            model=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        RegressionTask(
            model=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(TropicalCycloneDataModule, 'plot', plot)
        datamodule = TropicalCycloneDataModule(
            root='tests/data/cyclone', batch_size=1, num_workers=0
        )
        model = RegressionTask(model='resnet18')
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
        model = RegressionTask(model='resnet18')
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
        model = RegressionTask(model='resnet18')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_invalid_loss(self) -> None:
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            RegressionTask(model='resnet18', loss='invalid_loss')

    @pytest.mark.parametrize(
        'model_name', ['resnet18', 'efficientnetv2_s', 'vit_base_patch16_224']
    )
    def test_freeze_backbone(self, model_name: str) -> None:
        model = RegressionTask(model=model_name, freeze_backbone=True)
        assert not all([param.requires_grad for param in model.model.parameters()])
        assert all(
            [param.requires_grad for param in model.model.get_classifier().parameters()]
        )


class TestPixelwiseRegressionTask:
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
        PixelwiseRegressionTask(model='unet', backbone='resnet18', weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model='unet',
            backbone=mocked_weights.meta['model'],
            weights=mocked_weights,
            in_channels=mocked_weights.meta['in_chans'],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model='unet',
            backbone=mocked_weights.meta['model'],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model='unet',
            backbone=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
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
        model = PixelwiseRegressionTask(
            model=model_name, backbone=backbone, freeze_backbone=True
        )
        assert all(
            [param.requires_grad is False for param in model.model.encoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.decoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )

    @pytest.mark.parametrize(
        'model_name', ['unet', 'deeplabv3+', 'segformer', 'upernet']
    )
    def test_freeze_decoder(self, model_name: str) -> None:
        model = PixelwiseRegressionTask(
            model=model_name, backbone='resnet18', freeze_decoder=True
        )
        assert all(
            [param.requires_grad is False for param in model.model.decoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.encoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )

    def test_vit_backbone(self) -> None:
        PixelwiseRegressionTask(model='dpt', backbone='tu-vit_base_patch16_224')


class TestSpatioTemporalPixelwiseRegressionTask:
    @staticmethod
    def _create_video_model(**kwargs: Any) -> SpatioTemporalPixelwiseRegressionTask:
        model = SpatioTemporalPixelwiseRegressionTask(
            convlstm_hidden_dim=8, convlstm_num_layers=1, **kwargs
        )
        model.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
        model.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]
        return model

    def test_video_forward_defaults_to_convlstm(self) -> None:
        model = SpatioTemporalPixelwiseRegressionTask(in_channels=3)
        y_hat = model(torch.randn(2, 7, 3, 16, 16))
        assert y_hat.shape == (2, 1, 16, 16)

    def test_video_forward_multiple_outputs(self) -> None:
        model = SpatioTemporalPixelwiseRegressionTask(in_channels=3, num_outputs=2)
        y_hat = model(torch.randn(2, 7, 3, 16, 16))
        assert y_hat.shape == (2, 2, 16, 16)

    def test_unsupported_video_model(self) -> None:
        with pytest.raises(
            ValueError,
            match="SpatioTemporalPixelwiseRegressionTask only supports 'convlstm'",
        ):
            SpatioTemporalPixelwiseRegressionTask(model='unet', in_channels=3)

    @pytest.mark.parametrize(
        ('loss', 'expected_type'), [('mse', nn.MSELoss), ('mae', nn.L1Loss)]
    )
    def test_loss_selection(self, loss: str, expected_type: type[nn.Module]) -> None:
        model = self._create_video_model(in_channels=3, loss=loss)
        assert isinstance(model.criterion, expected_type)

    def test_invalid_loss(self) -> None:
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            SpatioTemporalPixelwiseRegressionTask(in_channels=3, loss='invalid_loss')

    def test_convlstm_timeseries_forward_and_steps(self) -> None:
        model = self._create_video_model(
            model='convlstm', in_channels=10, num_outputs=1
        )
        batch = {
            'image': torch.randn(2, 7, 10, 16, 16),
            'mask': torch.randn(2, 16, 16),
            'length': torch.tensor([7, 5]),
        }
        y_hat = model(batch['image'], lengths=batch['length'])
        assert y_hat.shape == (2, 1, 16, 16)

        y_hat_no_lengths = model(batch['image'])
        y_hat_last_step = model(batch['image'], lengths=torch.tensor([7, 7]))
        torch.testing.assert_close(y_hat_no_lengths, y_hat_last_step)

        y_hat_clamped = model(batch['image'], lengths=torch.tensor([9.0, 12.0]))
        torch.testing.assert_close(y_hat_no_lengths, y_hat_clamped)

        train_loss = model.training_step(batch, 0)
        assert train_loss.ndim == 0
        assert model.validation_step(batch, 0) is None
        assert model.test_step(batch, 0) is None

        predictions = model.predict_step(batch, 0)
        assert predictions.shape == (2, 1, 16, 16)

    def test_multichannel_targets(self) -> None:
        model = self._create_video_model(in_channels=3, num_outputs=2)
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'mask': torch.randn(2, 2, 16, 16),
            'length': torch.tensor([4, 3]),
        }

        train_loss = model.training_step(batch, 0)
        assert train_loss.ndim == 0
