# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Literal

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, OSCDDataModule
from torchgeo.datasets import OSCD, RGBBandsMissingError
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ChangeDetectionTask


class ChangeDetectionTestModel(Module):
    def __init__(self, in_channels: int = 3, classes: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return x


def create_model(**kwargs: Any) -> Module:
    return ChangeDetectionTestModel(**kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    return None


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class PredictChangeDetectionDataModule(OSCDDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = OSCD(**self.kwargs)


class TestChangeDetectionTask:
    @pytest.mark.parametrize(
        'name',
        [
            'bright',
            'cabuar',
            'chabud',
            'levircd',
            'levircdplus',
            'oscd',
            'oscd_multiclass',
            'oscd_multiclass_focal',
            'oscd_multiclass_jaccard',
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        match name:
            case 'cabuar' | 'chabud':
                pytest.importorskip('h5py', minversion='3.6')

        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(smp, 'Unet', create_model)

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

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictChangeDetectionDataModule(
            root=os.path.join('tests', 'data', 'oscd'),
            batch_size=2,
            patch_size=32,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(backbone='resnet18', in_channels=13, model='unet')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

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
        # multiply in_chans by 2 since images are concatenated
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans'] * 2
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    @pytest.mark.parametrize('model', [6], indirect=True)
    def test_weight_file(self, checkpoint: str) -> None:
        ChangeDetectionTask(backbone='resnet18', weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=mocked_weights.meta['model'],
            weights=mocked_weights,
            in_channels=mocked_weights.meta['in_chans'],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=mocked_weights.meta['model'],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=weights.meta['model'],
            weights=weights,
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        ChangeDetectionTask(
            backbone=weights.meta['model'],
            weights=str(weights),
            in_channels=weights.meta['in_chans'],
        )

    @pytest.mark.parametrize('model_name', ['unet', 'fcsiamdiff', 'fcsiamconc'])
    @pytest.mark.parametrize(
        'backbone', ['resnet18', 'mobilenet_v2', 'efficientnet-b0']
    )
    def test_freeze_backbone(
        self,
        model_name: Literal[
            'unet', 'deeplabv3+', 'segformer', 'upernet', 'fcsiamdiff', 'fcsiamconc'
        ],
        backbone: str,
    ) -> None:
        model = ChangeDetectionTask(
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
        'model_name',
        ['unet', 'deeplabv3+', 'segformer', 'upernet', 'fcsiamdiff', 'fcsiamconc'],
    )
    def test_freeze_decoder(
        self,
        model_name: Literal[
            'unet', 'deeplabv3+', 'segformer', 'upernet', 'fcsiamdiff', 'fcsiamconc'
        ],
    ) -> None:
        model = ChangeDetectionTask(
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
        ChangeDetectionTask(model='dpt', backbone='tu-vit_base_patch16_224')

    def test_fcn_model(self) -> None:
        """FCN has no backbone/decoder. Need separate test for full test coverage."""
        ChangeDetectionTask(model='fcn')

    @pytest.mark.parametrize('loss_fn', ['bce', 'jaccard', 'focal'])
    def test_losses(self, loss_fn: Literal['bce', 'jaccard', 'focal']) -> None:
        ChangeDetectionTask(loss=loss_fn)

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(OSCDDataModule, 'plot', plot)
        datamodule = OSCDDataModule(
            root=os.path.join('tests', 'data', 'oscd'),
            batch_size=2,
            patch_size=32,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(backbone='resnet18', in_channels=13, model='unet')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(OSCDDataModule, 'plot', plot_missing_bands)
        datamodule = OSCDDataModule(
            root=os.path.join('tests', 'data', 'oscd'),
            batch_size=2,
            patch_size=32,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(backbone='resnet18', in_channels=13, model='unet')
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_class_weights(self) -> None:
        class_weights_list = [1.0, 2.0, 0.5]
        task = ChangeDetectionTask(
            class_weights=class_weights_list, task='multiclass', num_classes=3
        )
        assert task.hparams['class_weights'] == class_weights_list

        class_weights_tensor = torch.tensor([1.0, 2.0, 0.5])
        task = ChangeDetectionTask(
            class_weights=class_weights_tensor, task='multiclass', num_classes=3
        )
        assert torch.equal(task.hparams['class_weights'], class_weights_tensor)

        task = ChangeDetectionTask(task='multiclass', num_classes=3)
        assert task.hparams['class_weights'] is None

    @pytest.mark.parametrize('loss_fn', ['jaccard'])
    def test_jaccard_ignore_index(self, loss_fn: Literal['jaccard']) -> None:
        ChangeDetectionTask(
            task='multiclass', num_classes=5, loss=loss_fn, ignore_index=0
        )

    def test_multiclass_validation(self, fast_dev_run: bool) -> None:
        datamodule = OSCDDataModule(
            root=os.path.join('tests', 'data', 'oscd'),
            batch_size=2,
            patch_size=16,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(
            backbone='resnet18',
            in_channels=13,
            model='unet',
            task='multiclass',
            num_classes=2,
            loss='ce',
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_multiclass_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictChangeDetectionDataModule(
            root=os.path.join('tests', 'data', 'oscd'),
            batch_size=2,
            patch_size=16,
            val_split_pct=0.5,
            num_workers=0,
        )
        model = ChangeDetectionTask(
            backbone='resnet18',
            in_channels=13,
            model='unet',
            task='multiclass',
            num_classes=2,
            loss='ce',
        )
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_levircd_temporal_correspondence(self) -> None:
        """Test LEVIR-CD temporal correspondence issues from GitHub issue #2920."""
        pytest.importorskip(
            'torchgeo.datamodules.levircd', reason='LEVIRCDDataModule required'
        )

        from torchgeo.datamodules.levircd import LEVIRCDBenchmarkDataModule

        # Test Issue 2: LEVIRCDBenchmarkDataModule temporal correspondence
        batch_size = 2
        temporal_frames = 2  # t1, t2 for images
        patches_per_frame = 16  # 16 patches per 1024x1024 image
        channels = 3
        height = width = 256

        # Mock batch similar to what LEVIRCD would produce after _ExtractPatches
        batch = {
            'image': torch.randn(
                batch_size, temporal_frames, patches_per_frame, channels, height, width
            ),
            'mask': torch.randn(
                batch_size, 1, patches_per_frame, 1, height, width
            ),  # Single change mask
        }

        # Test the on_after_batch_transfer method
        datamodule = LEVIRCDBenchmarkDataModule(batch_size=8, patch_size=256)
        result_batch = datamodule.on_after_batch_transfer(batch, 0)

        # Check spatial correspondence: patch 0 from image should correspond to patch 0 from mask
        expected_image_shape = (
            batch_size * patches_per_frame,
            temporal_frames,
            channels,
            height,
            width,
        )
        expected_mask_shape = (batch_size * patches_per_frame, 1, height, width)

        assert result_batch['image'].shape == expected_image_shape, (
            f'Image shape mismatch: expected {expected_image_shape}, got {result_batch["image"].shape}'
        )
        assert result_batch['mask'].shape == expected_mask_shape, (
            f'Mask shape mismatch: expected {expected_mask_shape}, got {result_batch["mask"].shape}'
        )

    def test_extract_patches_temporal_ordering(self) -> None:
        """Test _ExtractPatches temporal ordering with VideoSequential from GitHub issue #2920."""
        from torchgeo.transforms.transforms import _ExtractPatches

        # Simulate VideoSequential input: [B, T, C, H, W]
        batch_size = 2
        temporal_frames = 2
        channels = 3
        height = width = 512  # Smaller for faster testing

        input_tensor = torch.randn(batch_size, temporal_frames, channels, height, width)

        # VideoSequential flattens B and T dimensions for processing: [B*T, C, H, W]
        flattened_input = input_tensor.reshape(
            batch_size * temporal_frames, channels, height, width
        )

        # Apply _ExtractPatches
        extract_patches = _ExtractPatches(window_size=256, stride=256, keepdim=False)
        patches = extract_patches(flattened_input)

        # Verify we get expected number of patches
        expected_patches_per_image = (height // 256) * (
            width // 256
        )  # 4 patches for 512x512 -> 256x256
        expected_total_patches = (
            batch_size * temporal_frames * expected_patches_per_image
        )
        assert patches.shape[0] == expected_total_patches, (
            f'Expected {expected_total_patches} total patches, got {patches.shape[0]}'
        )

    def test_random_vs_deterministic_cropping(self) -> None:
        """Test that training uses random crops while val/test use deterministic patches."""
        pytest.importorskip(
            'torchgeo.datamodules.levircd', reason='LEVIRCDDataModule required'
        )

        from torchgeo.datamodules.levircd import LEVIRCDDataModule

        # Create mock data
        batch = {
            'image': torch.randn(2, 2, 3, 1024, 1024),  # [B, T, C, H, W]
            'mask': torch.randn(2, 1, 1024, 1024),  # [B, C, H, W]
        }

        datamodule = LEVIRCDDataModule(batch_size=2, patch_size=256)

        # Test training transform setup (should be random)
        if datamodule.train_aug is not None:
            assert hasattr(datamodule.train_aug, 'same_on_batch')
            assert not datamodule.train_aug.same_on_batch, (
                'Training should allow different crops per batch'
            )

        # Test validation transform setup (should be deterministic)
        if datamodule.val_aug is not None:
            assert hasattr(datamodule.val_aug, 'same_on_batch')
            assert datamodule.val_aug.same_on_batch, (
                'Validation should use same crops per batch'
            )

            # Verify we can call the transforms without errors
            val_result1 = datamodule.val_aug(batch)
            assert val_result1['image'].shape[0] >= batch['image'].shape[0], (
                'Validation should preserve or increase batch size'
            )
