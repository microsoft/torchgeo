# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from types import ModuleType
from typing import Any

import pytest
import torch
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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


class FakeRFDETRModelConfig:
    model_fields = frozenset(
        ('num_classes', 'pretrain_weights', 'resolution', 'freeze_encoder')
    )

    def __init__(self, **kwargs: Any) -> None:
        self.num_classes = kwargs.get('num_classes', 90)
        self.pretrain_weights = kwargs.get('pretrain_weights')
        self.resolution = kwargs.get('resolution', 384)
        self.freeze_encoder = kwargs.get('freeze_encoder', False)


class FakeRFDETRTrainConfig:
    model_fields = frozenset(('dataset_dir', 'output_dir', 'lr', 'lr_encoder'))

    def __init__(self, **kwargs: Any) -> None:
        self.dataset_dir = kwargs.get('dataset_dir', '.')
        self.output_dir = kwargs.get('output_dir', '.')
        self.lr = kwargs.get('lr', 1e-3)
        self.lr_encoder = kwargs.get('lr_encoder', 1e-5)


def patch_fake_rfdetr_configs(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        ObjectDetection,
        '_load_rf_detr_config_dependencies',
        staticmethod(
            lambda: (
                FakeRFDETRModelConfig,
                FakeRFDETRModelConfig,
                FakeRFDETRModelConfig,
                FakeRFDETRModelConfig,
                FakeRFDETRModelConfig,
                FakeRFDETRTrainConfig,
            )
        ),
    )


def patch_fake_rfdetr_runtime(monkeypatch: MonkeyPatch) -> dict[str, list[Any]]:
    state: dict[str, list[Any]] = {
        'build_namespace_calls': [],
        'load_pretrain_calls': [],
    }

    def fake_build_namespace(model_config: Any, train_config: Any) -> dict[str, Any]:
        namespace = {
            'model_config': model_config,
            'train_config': train_config,
            'call_index': len(state['build_namespace_calls']),
        }
        state['build_namespace_calls'].append(namespace)
        return namespace

    class FakeRFDETRModel(torch.nn.Module):
        def __init__(self, namespace: dict[str, Any]) -> None:
            super().__init__()
            self.namespace = namespace

        def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {'namespace': self.namespace, 'args': args, 'kwargs': kwargs}

    def fake_build_model(namespace: dict[str, Any]) -> FakeRFDETRModel:
        return FakeRFDETRModel(namespace)

    def fake_load_pretrain_weights(model: Any, model_config: Any) -> None:
        state['load_pretrain_calls'].append((model, model_config))

    def fake_build_criterion_and_postprocessors(
        namespace: dict[str, Any],
    ) -> tuple[object, Any]:
        return object(), lambda outputs, _: {'outputs': outputs, 'namespace': namespace}

    monkeypatch.setattr(
        ObjectDetection,
        '_load_rf_detr_runtime_dependencies',
        staticmethod(
            lambda: (
                fake_build_namespace,
                fake_build_criterion_and_postprocessors,
                fake_build_model,
                fake_load_pretrain_weights,
            )
        ),
    )
    return state


def patch_fake_rfdetr_tensor_utils(monkeypatch: MonkeyPatch) -> dict[str, list[Any]]:
    state: dict[str, list[Any]] = {'calls': []}

    def nested_tensor_from_tensor_list(images: list[torch.Tensor]) -> dict[str, Any]:
        state['calls'].append(images)
        return {'images': images}

    rfdetr_module = ModuleType('rfdetr')
    utilities_module = ModuleType('rfdetr.utilities')
    tensors_module = ModuleType('rfdetr.utilities.tensors')
    tensors_module.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
    utilities_module.tensors = tensors_module
    rfdetr_module.utilities = utilities_module

    monkeypatch.setitem(sys.modules, 'rfdetr', rfdetr_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.utilities', utilities_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.utilities.tensors', tensors_module)
    return state


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

    def test_rf_detr_preserves_num_classes_api(self, monkeypatch: MonkeyPatch) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )
        assert model.rf_detr_model_config.num_classes == 1

    def test_rf_detr_accepts_kwargs(self, monkeypatch: MonkeyPatch) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        model = ObjectDetection(
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
        patch_fake_rfdetr_configs(monkeypatch)

        def broken_runtime_import() -> Any:
            raise ImportError(
                "cannot import name 'BackboneConfigMixin' from 'transformers'"
            )

        monkeypatch.setattr(
            ObjectDetection,
            '_load_rf_detr_runtime_dependencies',
            staticmethod(broken_runtime_import),
        )

        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        assert model.rf_detr_model_config.num_classes == 1

        match = 'RF-DETR runtime could not be imported'
        with pytest.raises(ImportError, match=match):
            model._ensure_rf_detr_runtime()

    def test_rf_detr_missing_runtime_dependency_errors(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)

        def missing_runtime_dependency() -> Any:
            raise ModuleNotFoundError("No module named 'rfdetr'")

        monkeypatch.setattr(
            ObjectDetection,
            '_load_rf_detr_runtime_dependencies',
            staticmethod(missing_runtime_dependency),
        )

        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        match = "RF-DETR support requires the optional 'rfdetr' dependency"
        with pytest.raises(ImportError, match=match):
            model._ensure_rf_detr_runtime()

    def test_rf_detr_initializes_runtime_and_loads_pretrain_weights(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        state = patch_fake_rfdetr_runtime(monkeypatch)

        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights='rf-detr-nano.pth'
        )

        assert model._rf_detr_runtime_ready
        assert model.rf_detr_criterion is not None
        assert model.rf_detr_postprocess is not None
        assert len(state['load_pretrain_calls']) == 1
        assert len(state['build_namespace_calls']) == 2

        result = model(torch.randn(1, 3, 16, 16))
        assert result['namespace']['model_config'] is model.rf_detr_model_config

    def test_rf_detr_build_batch_converts_boxes(self, monkeypatch: MonkeyPatch) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        tensor_state = patch_fake_rfdetr_tensor_utils(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=3, pretrain_weights=None
        )

        batch = {
            'image': torch.arange(3 * 10 * 20, dtype=torch.float32).reshape(
                1, 3, 10, 20
            ),
            'bbox_xyxy': [torch.tensor([[2.0, 1.0, 6.0, 5.0]])],
            'label': [torch.tensor([1])],
        }

        samples, targets, metric_targets = model._build_rf_detr_batch(batch)

        assert len(tensor_state['calls']) == 1
        assert samples['images'][0].shape == (3, 10, 20)
        assert torch.equal(targets[0]['orig_size'], torch.tensor([10, 20]))
        assert torch.allclose(targets[0]['boxes'], torch.tensor([[0.2, 0.3, 0.2, 0.4]]))
        assert torch.equal(targets[0]['labels'], torch.tensor([0]))
        assert torch.equal(targets[0]['area'], torch.tensor([16.0]))
        assert torch.equal(targets[0]['iscrowd'], torch.tensor([0]))
        assert torch.equal(metric_targets[0]['boxes'], batch['bbox_xyxy'][0])
        assert torch.equal(metric_targets[0]['labels'], batch['label'][0])

    def test_rf_detr_build_batch_supports_predict_inputs(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        patch_fake_rfdetr_tensor_utils(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        samples, targets, metric_targets = model._build_rf_detr_batch(
            {'image': torch.randn(2, 3, 8, 6)}
        )

        assert len(samples['images']) == 2
        assert len(targets) == 2
        assert metric_targets == []
        assert all('boxes' not in target for target in targets)

    def test_rf_detr_build_batch_rejects_background_labels(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        patch_fake_rfdetr_tensor_utils(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        batch = {
            'image': torch.randn(1, 3, 8, 8),
            'bbox_xyxy': [torch.tensor([[1.0, 1.0, 2.0, 2.0]])],
            'label': [torch.tensor([0])],
        }

        match = 'TorchGeo RF-DETR support expects foreground labels to start at 1'
        with pytest.raises(ValueError, match=match):
            model._build_rf_detr_batch(batch)

    def test_rf_detr_postprocess_filters_background_class(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=3, pretrain_weights=None
        )
        assert model.rf_detr_model_config is not None
        model.rf_detr_model_config.num_classes = 2

        called: dict[str, Tensor] = {}

        def fake_postprocess(
            outputs: dict[str, Tensor], orig_sizes: Tensor
        ) -> list[dict[str, Tensor]]:
            called['orig_sizes'] = orig_sizes
            return [
                {
                    'boxes': torch.tensor(
                        [
                            [1.0, 1.0, 2.0, 2.0],
                            [3.0, 3.0, 4.0, 4.0],
                            [5.0, 5.0, 6.0, 6.0],
                        ]
                    ),
                    'labels': torch.tensor([0, 1, 2]),
                    'scores': torch.tensor([0.9, 0.8, 0.1]),
                }
            ]

        model.rf_detr_postprocess = fake_postprocess
        predictions = model._postprocess_rf_detr(
            outputs={'pred_logits': torch.tensor([1.0])},
            targets=[{'orig_size': torch.tensor([10, 20])}],
        )

        assert torch.equal(called['orig_sizes'], torch.tensor([[10, 20]]))
        assert len(predictions) == 1
        assert torch.equal(
            predictions[0]['boxes'],
            torch.tensor([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]),
        )
        assert torch.equal(predictions[0]['labels'], torch.tensor([1, 2]))
        assert torch.equal(predictions[0]['scores'], torch.tensor([0.9, 0.8]))

    def test_rf_detr_requires_rgb(self) -> None:
        match = 'RF-DETR currently requires in_channels=3.'
        with pytest.raises(ValueError, match=match):
            ObjectDetection(
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

    def test_metrics_use_300_max_detection_threshold(self) -> None:
        model = ObjectDetection(backbone='resnet18', num_classes=2)
        metric = next(iter(model.val_metrics.values()))
        assert isinstance(metric, MeanAveragePrecision)
        assert tuple(metric.max_detection_thresholds) == (1, 10, 300)
