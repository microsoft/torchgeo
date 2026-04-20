# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any

import pytest
import torch
from lightning.pytorch import Trainer
from pytest import MonkeyPatch

from torchgeo.datamodules import MisconfigurationException, NASAMarineDebrisDataModule
from torchgeo.datasets import VHR10, NASAMarineDebris, RGBBandsMissingError
from torchgeo.main import main
from torchgeo.tasks import ObjectDetection

# MAP metric requires pycocotools to be installed
pytest.importorskip('pycocotools')


class PredictObjectDetectionDataModule(NASAMarineDebrisDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = NASAMarineDebris(**self.kwargs)


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


def plot(*args: Any, **kwargs: Any) -> None:
    return None


class TestObjectDetection:
    @pytest.mark.parametrize(
        'name', ['nasa_marine_debris', 'reforestree', 'vhr10_obj_det']
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        if name.startswith('vhr10'):
            monkeypatch.setattr(VHR10, '__len__', lambda self: 5)

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

    def test_invalid_model(self) -> None:
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetection(model='invalid_model')

    def test_invalid_backbone(self) -> None:
        match = "Backbone type 'invalid_backbone' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetection(backbone='invalid_backbone')

    def test_rf_detr_preserves_num_classes_api(self) -> None:
        model = ObjectDetectionTask(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )
        assert model.rf_detr_model_config.num_classes == 1

    def test_rf_detr_accepts_kwargs(self) -> None:
        model = ObjectDetectionTask(
            model='rf-detr-nano',
            num_classes=2,
            pretrain_weights=None,
            resolution=512,
            lr_encoder=1e-5,
        )
        assert model.rf_detr_model_config.resolution == 512
        assert model.rf_detr_train_config.lr_encoder == 1e-5

    def test_rf_detr_defers_runtime_import_errors(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        class FakeModelConfig:
            model_fields = {
                'num_classes': None,
                'pretrain_weights': None,
                'resolution': None,
                'freeze_encoder': None,
            }

            def __init__(self, **kwargs: Any) -> None:
                self.num_classes = kwargs.get('num_classes', 90)
                self.pretrain_weights = kwargs.get('pretrain_weights')
                self.resolution = kwargs.get('resolution', 384)
                self.freeze_encoder = kwargs.get('freeze_encoder', False)

        class FakeTrainConfig:
            model_fields = {
                'dataset_dir': None,
                'output_dir': None,
                'lr': None,
                'lr_encoder': None,
            }

            def __init__(self, **kwargs: Any) -> None:
                self.dataset_dir = kwargs.get('dataset_dir', '.')
                self.output_dir = kwargs.get('output_dir', '.')
                self.lr = kwargs.get('lr', 1e-3)
                self.lr_encoder = kwargs.get('lr_encoder', 1e-5)

        monkeypatch.setattr(
            ObjectDetectionTask,
            '_load_rf_detr_config_dependencies',
            staticmethod(
                lambda: (
                    FakeModelConfig,
                    FakeModelConfig,
                    FakeModelConfig,
                    FakeModelConfig,
                    FakeModelConfig,
                    FakeTrainConfig,
                )
            ),
        )

        def broken_runtime_import() -> Any:
            raise ImportError(
                "cannot import name 'BackboneConfigMixin' from 'transformers'"
            )

        monkeypatch.setattr(
            ObjectDetectionTask,
            '_load_rf_detr_runtime_dependencies',
            staticmethod(broken_runtime_import),
        )

        model = ObjectDetectionTask(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        assert model.rf_detr_model_config.num_classes == 1

        match = 'RF-DETR runtime could not be imported'
        with pytest.raises(ImportError, match=match):
            model._ensure_rf_detr_runtime()

    def test_rf_detr_requires_rgb(self) -> None:
        match = 'RF-DETR currently requires in_channels=3.'
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(
                model='rf-detr-nano',
                num_classes=2,
                in_channels=4,
                pretrain_weights=None,
            )

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(NASAMarineDebrisDataModule, 'plot', plot)
        datamodule = NASAMarineDebrisDataModule(
            root='tests/data/nasa_marine_debris', batch_size=1, num_workers=0
        )
        model = ObjectDetection(backbone='resnet18', num_classes=2)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(NASAMarineDebrisDataModule, 'plot', plot_missing_bands)
        datamodule = NASAMarineDebrisDataModule(
            root='tests/data/nasa_marine_debris', batch_size=1, num_workers=0
        )
        model = ObjectDetection(backbone='resnet18', num_classes=2)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictObjectDetectionDataModule(
            root='tests/data/nasa_marine_debris', batch_size=1, num_workers=0
        )
        model = ObjectDetection(backbone='resnet18', num_classes=2)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize('model_name', ['faster-rcnn', 'fcos', 'retinanet'])
    def test_freeze_backbone(self, model_name: str) -> None:
        model = ObjectDetection(
            model=model_name, backbone='resnet18', freeze_backbone=True
        )
        assert not all(param.requires_grad for param in model.model.parameters())

    @pytest.mark.parametrize('model_name', ['faster-rcnn', 'fcos', 'retinanet'])
    @pytest.mark.parametrize('in_channels', [1, 4])
    def test_multispectral_support(self, model_name: str, in_channels: int) -> None:
        model = ObjectDetection(
            model=model_name,
            backbone='resnet18',
            num_classes=2,
            in_channels=in_channels,
        )
        model.eval()
        sample = [torch.randn(in_channels, 224, 224)]
        with torch.inference_mode():
            model(sample)
