# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

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
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 4)
        self.warmup_epochs = kwargs.get('warmup_epochs', 1)
        self.lr_scheduler = kwargs.get('lr_scheduler', 'cosine')
        self.lr_min_factor = kwargs.get('lr_min_factor', 0.2)
        self.lr_drop = kwargs.get('lr_drop', 3)


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


def patch_fake_rfdetr_import_modules(monkeypatch: MonkeyPatch) -> dict[str, Any]:
    class FakeImportedModelConfig:
        model_fields = frozenset(('num_classes',))

    class FakeImportedLargeConfig(FakeImportedModelConfig):
        pass

    class FakeImportedMediumConfig(FakeImportedModelConfig):
        pass

    class FakeImportedNanoConfig(FakeImportedModelConfig):
        pass

    class FakeImportedSmallConfig(FakeImportedModelConfig):
        pass

    class FakeImportedTrainConfig:
        model_fields = frozenset(('lr',))

    def fake_build_namespace(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return args, kwargs

    def fake_build_criterion_and_postprocessors(
        *args: Any, **kwargs: Any
    ) -> tuple[Any, Any]:
        return args, kwargs

    def fake_build_model(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return args, kwargs

    def fake_load_pretrain_weights(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return args, kwargs

    rfdetr_module = ModuleType('rfdetr')
    config_module = ModuleType('rfdetr.config')
    namespace_module = ModuleType('rfdetr._namespace')
    models_module = ModuleType('rfdetr.models')
    lwdetr_module = ModuleType('rfdetr.models.lwdetr')
    weights_module = ModuleType('rfdetr.models.weights')

    config_module.ModelConfig = FakeImportedModelConfig
    config_module.RFDETRLargeConfig = FakeImportedLargeConfig
    config_module.RFDETRMediumConfig = FakeImportedMediumConfig
    config_module.RFDETRNanoConfig = FakeImportedNanoConfig
    config_module.RFDETRSmallConfig = FakeImportedSmallConfig
    config_module.TrainConfig = FakeImportedTrainConfig

    namespace_module.build_namespace = fake_build_namespace
    lwdetr_module.build_criterion_and_postprocessors = (
        fake_build_criterion_and_postprocessors
    )
    lwdetr_module.build_model = fake_build_model
    weights_module.load_pretrain_weights = fake_load_pretrain_weights

    rfdetr_module.config = config_module
    rfdetr_module._namespace = namespace_module
    rfdetr_module.models = models_module
    models_module.lwdetr = lwdetr_module
    models_module.weights = weights_module

    monkeypatch.setitem(sys.modules, 'rfdetr', rfdetr_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.config', config_module)
    monkeypatch.setitem(sys.modules, 'rfdetr._namespace', namespace_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.models', models_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.models.lwdetr', lwdetr_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.models.weights', weights_module)

    return {
        'ModelConfig': FakeImportedModelConfig,
        'RFDETRLargeConfig': FakeImportedLargeConfig,
        'RFDETRMediumConfig': FakeImportedMediumConfig,
        'RFDETRNanoConfig': FakeImportedNanoConfig,
        'RFDETRSmallConfig': FakeImportedSmallConfig,
        'TrainConfig': FakeImportedTrainConfig,
        'build_namespace': fake_build_namespace,
        'build_criterion_and_postprocessors': fake_build_criterion_and_postprocessors,
        'build_model': fake_build_model,
        'load_pretrain_weights': fake_load_pretrain_weights,
    }


def patch_fake_rfdetr_optimizer_modules(
    monkeypatch: MonkeyPatch, param_dicts: list[dict[str, Any]]
) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def fake_build_namespace(model_config: Any, train_config: Any) -> dict[str, Any]:
        namespace = {'model_config': model_config, 'train_config': train_config}
        calls['namespace'] = namespace
        return namespace

    def fake_get_param_dict(
        namespace: dict[str, Any], model: Any
    ) -> list[dict[str, Any]]:
        calls['get_param_dict'] = (namespace, model)
        return param_dicts

    rfdetr_module = sys.modules.get('rfdetr', ModuleType('rfdetr'))
    namespace_module = ModuleType('rfdetr._namespace')
    training_module = ModuleType('rfdetr.training')
    param_groups_module = ModuleType('rfdetr.training.param_groups')

    namespace_module.build_namespace = fake_build_namespace
    param_groups_module.get_param_dict = fake_get_param_dict
    training_module.param_groups = param_groups_module
    rfdetr_module._namespace = namespace_module
    rfdetr_module.training = training_module

    monkeypatch.setitem(sys.modules, 'rfdetr', rfdetr_module)
    monkeypatch.setitem(sys.modules, 'rfdetr._namespace', namespace_module)
    monkeypatch.setitem(sys.modules, 'rfdetr.training', training_module)
    monkeypatch.setitem(
        sys.modules, 'rfdetr.training.param_groups', param_groups_module
    )
    return calls


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

    def test_load_rf_detr_config_dependencies(self, monkeypatch: MonkeyPatch) -> None:
        imported = patch_fake_rfdetr_import_modules(monkeypatch)
        assert ObjectDetection._load_rf_detr_config_dependencies() == (
            imported['ModelConfig'],
            imported['RFDETRLargeConfig'],
            imported['RFDETRMediumConfig'],
            imported['RFDETRNanoConfig'],
            imported['RFDETRSmallConfig'],
            imported['TrainConfig'],
        )

    def test_load_rf_detr_runtime_dependencies(self, monkeypatch: MonkeyPatch) -> None:
        imported = patch_fake_rfdetr_import_modules(monkeypatch)
        assert ObjectDetection._load_rf_detr_runtime_dependencies() == (
            imported['build_namespace'],
            imported['build_criterion_and_postprocessors'],
            imported['build_model'],
            imported['load_pretrain_weights'],
        )

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

    def test_rf_detr_freeze_backbone_maps_to_freeze_encoder(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        model = ObjectDetectionTask(
            model='rf-detr-nano',
            num_classes=2,
            freeze_backbone=True,
            pretrain_weights=None,
        )
        assert model.rf_detr_model_config.freeze_encoder

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

    def test_rf_detr_ensure_runtime_noops_for_torchvision_backend(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        model = ObjectDetection(backbone='resnet18', num_classes=2)
        monkeypatch.setattr(
            model,
            '_initialize_rf_detr_runtime',
            lambda: pytest.fail('_initialize_rf_detr_runtime should not be called'),
        )
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

    def test_rf_detr_missing_config_dependency_errors(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ObjectDetection,
            '_load_rf_detr_config_dependencies',
            staticmethod(lambda: (_ for _ in ()).throw(ModuleNotFoundError('rfdetr'))),
        )

        match = "RF-DETR support requires the optional 'rfdetr' dependency"
        with pytest.raises(ImportError, match=match):
            ObjectDetection(
                model='rf-detr-nano', num_classes=2, pretrain_weights=None
            )

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

    def test_rf_detr_training_step_uses_rf_detr_loss_path(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        calls: dict[str, Any] = {}
        samples = {'samples': 'value'}
        targets = [{'target': 1}]

        monkeypatch.setattr(
            model, '_build_rf_detr_batch', lambda batch: (samples, targets, [])
        )

        class FakeCriterion:
            def __init__(self) -> None:
                self.weight_dict = {'loss_ce': 0.5, 'loss_bbox': 2.0}

            def __call__(
                self, outputs: dict[str, Tensor], targets_arg: list[dict[str, int]]
            ) -> dict[str, Tensor]:
                calls['criterion'] = (outputs, targets_arg)
                return {
                    'loss_ce': torch.tensor(2.0),
                    'loss_bbox': torch.tensor(3.0),
                    'unused': torch.tensor(100.0),
                }

        class FakeModel(torch.nn.Module):
            def forward(self, samples_arg: Any, targets_arg: Any) -> dict[str, Tensor]:
                calls['model'] = (samples_arg, targets_arg)
                return {'pred_logits': torch.tensor([1.0])}

        logged: dict[str, Any] = {}
        model.model = FakeModel()
        model.rf_detr_criterion = FakeCriterion()
        monkeypatch.setattr(
            model,
            'log_dict',
            lambda metrics, batch_size: logged.update(
                {'metrics': metrics, 'batch_size': batch_size}
            ),
        )

        batch = {
            'image': torch.randn(2, 3, 8, 8),
            'bbox_xyxy': [torch.tensor([[1.0, 1.0, 2.0, 2.0]])] * 2,
            'label': [torch.tensor([1])] * 2,
        }
        train_loss = model.training_step(batch, batch_idx=0)

        assert calls['model'] == (samples, targets)
        assert calls['criterion'][1] == targets
        assert torch.isclose(train_loss, torch.tensor(7.0))
        assert logged['batch_size'] == 2
        assert set(logged['metrics']) == {'loss_ce', 'loss_bbox', 'unused'}

    def test_rf_detr_validation_step_uses_rf_detr_prediction_path(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        calls: dict[str, Any] = {}
        samples = {'samples': 'value'}
        targets = [{'target': 1}]
        metric_targets = [{'boxes': torch.tensor([[1.0, 1.0, 2.0, 2.0]])}]
        predictions = [
            {
                'boxes': torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                'labels': torch.tensor([1]),
                'scores': torch.tensor([0.9]),
            }
        ]

        monkeypatch.setattr(
            model,
            '_build_rf_detr_batch',
            lambda batch: (samples, targets, metric_targets),
        )

        class FakeModel(torch.nn.Module):
            def forward(self, samples_arg: Any) -> dict[str, Tensor]:
                calls['model'] = samples_arg
                return {'pred_logits': torch.tensor([1.0])}

        def fake_postprocess(
            outputs: dict[str, Tensor], targets_arg: list[dict[str, int]]
        ) -> list[dict[str, Tensor]]:
            calls['postprocess'] = (outputs, targets_arg)
            return predictions

        class FakeMetrics(torch.nn.Module):
            def forward(
                self, y_hat: list[dict[str, Tensor]], y: list[dict[str, Tensor]]
            ) -> dict[str, Tensor]:
                calls['metrics'] = (y_hat, y)
                return {'val_map': torch.tensor(0.5), 'val_classes': torch.tensor([1])}

        logged: dict[str, Any] = {}
        model.model = FakeModel()
        monkeypatch.setattr(model, '_postprocess_rf_detr', cast(Any, fake_postprocess))
        monkeypatch.setattr(model, 'val_metrics', cast(Any, FakeMetrics()))
        monkeypatch.setattr(
            model, '_trainer', cast(Any, SimpleNamespace(datamodule=None))
        )
        monkeypatch.setattr(model, '_logger', cast(Any, None), raising=False)
        monkeypatch.setattr(
            model,
            'log_dict',
            lambda metrics, batch_size: logged.update(
                {'metrics': metrics, 'batch_size': batch_size}
            ),
        )

        batch = {
            'image': torch.randn(1, 3, 8, 8),
            'bbox_xyxy': [torch.tensor([[1.0, 1.0, 2.0, 2.0]])],
            'label': [torch.tensor([1])],
        }
        model.validation_step(batch, batch_idx=20)

        assert calls['model'] == samples
        assert calls['postprocess'][1] == targets
        assert calls['metrics'][0] is predictions
        assert calls['metrics'][1] is metric_targets
        assert logged['batch_size'] == 1
        assert set(logged['metrics']) == {'val_map'}
        assert torch.equal(logged['metrics']['val_map'], torch.tensor(0.5))

    def test_rf_detr_test_step_uses_rf_detr_prediction_path(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        calls: dict[str, Any] = {}
        samples = {'samples': 'value'}
        targets = [{'target': 1}]
        metric_targets = [{'boxes': torch.tensor([[1.0, 1.0, 2.0, 2.0]])}]
        predictions = [
            {
                'boxes': torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                'labels': torch.tensor([1]),
                'scores': torch.tensor([0.9]),
            }
        ]

        monkeypatch.setattr(
            model,
            '_build_rf_detr_batch',
            lambda batch: (samples, targets, metric_targets),
        )

        class FakeModel(torch.nn.Module):
            def forward(self, samples_arg: Any) -> dict[str, Tensor]:
                calls['model'] = samples_arg
                return {'pred_logits': torch.tensor([1.0])}

        def fake_postprocess(
            outputs: dict[str, Tensor], targets_arg: list[dict[str, int]]
        ) -> list[dict[str, Tensor]]:
            calls['postprocess'] = (outputs, targets_arg)
            return predictions

        class FakeMetrics(torch.nn.Module):
            def forward(
                self, y_hat: list[dict[str, Tensor]], y: list[dict[str, Tensor]]
            ) -> dict[str, Tensor]:
                calls['metrics'] = (y_hat, y)
                return {
                    'test_map': torch.tensor(0.5),
                    'test_classes': torch.tensor([1]),
                }

        logged: dict[str, Any] = {}
        model.model = FakeModel()
        monkeypatch.setattr(model, '_postprocess_rf_detr', cast(Any, fake_postprocess))
        monkeypatch.setattr(model, 'test_metrics', cast(Any, FakeMetrics()))
        monkeypatch.setattr(
            model,
            'log_dict',
            lambda metrics, batch_size: logged.update(
                {'metrics': metrics, 'batch_size': batch_size}
            ),
        )

        batch = {
            'image': torch.randn(1, 3, 8, 8),
            'bbox_xyxy': [torch.tensor([[1.0, 1.0, 2.0, 2.0]])],
            'label': [torch.tensor([1])],
        }
        model.test_step(batch, batch_idx=0)

        assert calls['model'] == samples
        assert calls['postprocess'][1] == targets
        assert calls['metrics'][0] is predictions
        assert calls['metrics'][1] is metric_targets
        assert logged['batch_size'] == 1
        assert set(logged['metrics']) == {'test_map'}
        assert torch.equal(logged['metrics']['test_map'], torch.tensor(0.5))

    def test_rf_detr_predict_step_uses_rf_detr_prediction_path(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        calls: dict[str, Any] = {}
        samples = {'samples': 'value'}
        targets = [{'target': 1}]
        predictions = [
            {
                'boxes': torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
                'labels': torch.tensor([1]),
                'scores': torch.tensor([0.9]),
            }
        ]

        monkeypatch.setattr(
            model, '_build_rf_detr_batch', lambda batch: (samples, targets, [])
        )

        class FakeModel(torch.nn.Module):
            def forward(self, samples_arg: Any) -> dict[str, Tensor]:
                calls['model'] = samples_arg
                return {'pred_logits': torch.tensor([1.0])}

        def fake_postprocess(
            outputs: dict[str, Tensor], targets_arg: list[dict[str, int]]
        ) -> list[dict[str, Tensor]]:
            calls['postprocess'] = (outputs, targets_arg)
            return predictions

        model.model = FakeModel()
        monkeypatch.setattr(model, '_postprocess_rf_detr', cast(Any, fake_postprocess))

        batch = {'image': torch.randn(1, 3, 8, 8)}
        result = model.predict_step(batch, batch_idx=0)

        assert calls['model'] == samples
        assert calls['postprocess'][1] == targets
        assert result is predictions

    def test_rf_detr_configure_optimizers_uses_rf_detr_backend(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        patch_fake_rfdetr_runtime(monkeypatch)
        model = ObjectDetection(
            model='rf-detr-nano', num_classes=2, pretrain_weights=None
        )

        class FakeParams(list[Any]):
            @property
            def requires_grad(self) -> bool:
                return any(param.requires_grad for param in self)

        trainable_param = torch.nn.Parameter(torch.tensor(1.0))
        frozen_param = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        param_dicts = [
            {'params': FakeParams([trainable_param]), 'lr': 1e-3},
            {'params': FakeParams([frozen_param]), 'lr': 1e-4},
        ]
        calls = patch_fake_rfdetr_optimizer_modules(monkeypatch, param_dicts)

        inner_model = torch.nn.Linear(1, 1)

        class FakeCompiledModel(torch.nn.Module):
            def __init__(self, orig_mod: torch.nn.Module) -> None:
                super().__init__()
                self._orig_mod = orig_mod

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self._orig_mod(*args, **kwargs)

        model.model = FakeCompiledModel(inner_model)
        assert model.rf_detr_train_config is not None
        model.rf_detr_train_config.lr = 1e-3
        model.rf_detr_train_config.weight_decay = 1e-4
        model.rf_detr_train_config.epochs = 4
        model.rf_detr_train_config.warmup_epochs = 1
        model.rf_detr_train_config.lr_scheduler = 'cosine'
        model.rf_detr_train_config.lr_min_factor = 0.2
        model.rf_detr_train_config.lr_drop = 3
        monkeypatch.setattr(
            model, '_trainer', cast(Any, SimpleNamespace(estimated_stepping_batches=20))
        )

        optimizers = model.configure_optimizers()

        optimizer = optimizers['optimizer']
        scheduler = optimizers['lr_scheduler']['scheduler']
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizers['lr_scheduler']['interval'] == 'step'
        assert calls['get_param_dict'][0] == calls['namespace']
        assert calls['get_param_dict'][1] is inner_model
        assert len(optimizer.param_groups) == 1

        lr_lambda = scheduler.lr_lambdas[0]
        assert lr_lambda(0) == 0.0
        assert lr_lambda(2) == 0.4
        cosine_value = lr_lambda(7)
        assert 0.2 <= cosine_value <= 1.0

        model.rf_detr_train_config.lr_scheduler = 'step'
        assert lr_lambda(12) == 1.0
        assert lr_lambda(16) == 0.1

    def test_rf_detr_rejects_unsupported_backbone(self) -> None:
        match = 'Backbone selection is not supported for RF-DETR.'
        with pytest.raises(ValueError, match=match):
            ObjectDetection(
                model='rf-detr-nano',
                backbone='resnet18',
                num_classes=2,
                pretrain_weights=None,
            )

    def test_rf_detr_rejects_weights_argument(self) -> None:
        match = "The 'weights' argument is not supported for RF-DETR."
        with pytest.raises(ValueError, match=match):
            ObjectDetection(
                model='rf-detr-nano',
                num_classes=2,
                weights=cast(Any, 'sentinel'),
                pretrain_weights=None,
            )

    def test_rf_detr_requires_num_classes_api_with_background(self) -> None:
        match = (
            "RF-DETR requires num_classes >= 2 when using TorchGeo's num_classes API"
        )
        with pytest.raises(ValueError, match=match):
            ObjectDetection(
                model='rf-detr-nano', num_classes=1, pretrain_weights=None
            )

    def test_rf_detr_requires_rgb(self) -> None:
        match = 'RF-DETR currently requires in_channels=3.'
        with pytest.raises(ValueError, match=match):
            ObjectDetection(
                model='rf-detr-nano',
                num_classes=2,
                in_channels=4,
                pretrain_weights=None,
            )

    def test_rf_detr_rejects_unknown_kwargs(self, monkeypatch: MonkeyPatch) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        match = "Unknown RF-DETR parameter 'bogus'."
        with pytest.raises(ValueError, match=match):
            ObjectDetection(
                model='rf-detr-nano', num_classes=2, pretrain_weights=None, bogus=123
            )

    def test_rf_detr_rejects_num_classes_in_kwargs(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        patch_fake_rfdetr_configs(monkeypatch)
        task = ObjectDetection(backbone='resnet18', num_classes=2)
        match = 'Do not pass num_classes through RF-DETR kwargs.'
        with pytest.raises(ValueError, match=match):
            task._configure_rf_detr_model(
                model='rf-detr-nano',
                backbone='resnet50',
                in_channels=3,
                num_classes=2,
                freeze_backbone=False,
                rf_detr_kwargs={'num_classes': 1},
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
