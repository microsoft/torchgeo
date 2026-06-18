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
from torchgeo.trainers import ObjectDetectionTask, PointDetectionTask

# MAP metric requires pycocotools to be installed
pytest.importorskip('pycocotools')


class PredictObjectDetectionDataModule(NASAMarineDebrisDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = NASAMarineDebris(**self.kwargs)


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


def plot(*args: Any, **kwargs: Any) -> None:
    return None


class TestObjectDetectionTask:
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
            ObjectDetectionTask(model='invalid_model')

    def test_invalid_backbone(self) -> None:
        match = "Backbone type 'invalid_backbone' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(backbone='invalid_backbone')

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(NASAMarineDebrisDataModule, 'plot', plot)
        datamodule = NASAMarineDebrisDataModule(
            root='tests/data/nasa_marine_debris', batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(backbone='resnet18', num_classes=2)
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
        model = ObjectDetectionTask(backbone='resnet18', num_classes=2)
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
        model = ObjectDetectionTask(backbone='resnet18', num_classes=2)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize('model_name', ['faster-rcnn', 'fcos', 'retinanet'])
    def test_freeze_backbone(self, model_name: str) -> None:
        model = ObjectDetectionTask(
            model=model_name, backbone='resnet18', freeze_backbone=True
        )
        assert not all([param.requires_grad for param in model.model.parameters()])

    @pytest.mark.parametrize('model_name', ['faster-rcnn', 'fcos', 'retinanet'])
    @pytest.mark.parametrize('in_channels', [1, 4])
    def test_multispectral_support(self, model_name: str, in_channels: int) -> None:
        model = ObjectDetectionTask(
            model=model_name,
            backbone='resnet18',
            num_classes=2,
            in_channels=in_channels,
        )
        model.eval()
        sample = [torch.randn(in_channels, 224, 224)]
        with torch.inference_mode():
            model(sample)


class TestPointDetectionTask:
    def test_invalid_distance_threshold(self) -> None:
        with pytest.raises(ValueError, match='distance_threshold must be positive'):
            PointDetectionTask(distance_threshold=0)

    def test_invalid_score_threshold(self) -> None:
        with pytest.raises(ValueError, match='score_threshold must be in the range'):
            PointDetectionTask(score_threshold=1.1)

    def test_add_prediction_points(self) -> None:
        model = PointDetectionTask(backbone='resnet18', num_classes=2)
        predictions = [
            {
                'boxes': torch.tensor([[0, 2, 10, 22]], dtype=torch.float32),
                'labels': torch.tensor([1]),
                'scores': torch.tensor([0.9]),
            }
        ]

        outputs = model._add_prediction_points(predictions)

        assert torch.equal(outputs[0]['points'], torch.tensor([[5, 12.0]]))
        assert torch.equal(outputs[0]['boxes'], predictions[0]['boxes'])

    def test_point_metrics(self) -> None:
        model = PointDetectionTask(
            backbone='resnet18',
            num_classes=3,
            distance_threshold=5,
            score_threshold=0.5,
        )
        batch = {
            'image': torch.zeros(1, 3, 32, 32),
            'points': [torch.tensor([[5, 5], [20, 20]], dtype=torch.float32)],
            'label': [torch.tensor([1, 2])],
        }
        predictions = [
            {
                'boxes': torch.tensor(
                    [[3, 3, 7, 7], [18, 18, 22, 22], [19, 19, 23, 23]],
                    dtype=torch.float32,
                ),
                'labels': torch.tensor([1, 1, 2]),
                'scores': torch.tensor([0.9, 0.8, 0.7]),
            }
        ]

        metrics = model._point_metrics(batch, predictions, prefix='val_')

        assert torch.equal(metrics['val_point_tp'], torch.tensor(2.0))
        assert torch.equal(metrics['val_point_fp'], torch.tensor(1.0))
        assert torch.equal(metrics['val_point_fn'], torch.tensor(0.0))
        assert torch.isclose(metrics['val_point_precision'], torch.tensor(2 / 3))
        assert torch.equal(metrics['val_point_recall'], torch.tensor(1.0))
        assert torch.isclose(metrics['val_point_f1'], torch.tensor(0.8))
