# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from timm.models import VisionTransformer
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import SSL4EOS12
from torchgeo.main import main
from torchgeo.models.vit import ViTSmall16_Weights
from torchgeo.trainers import ChangeDetectionTask, IJEPATask


def create_model(*args: Any, **kwargs: Any) -> Module:
    """Create a tiny ViT for fast testing."""
    kwargs.pop('pretrained', None)
    return VisionTransformer(depth=1, **kwargs)


class TestIJEPATask:
    @pytest.mark.parametrize('name', ['ssl4eo_s12_ijepa_1'])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(SSL4EOS12, '__len__', lambda self: 2)
        monkeypatch.setattr(timm, 'create_model', create_model)

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

    def test_full_scheduler(self, monkeypatch: MonkeyPatch) -> None:
        config = os.path.join('tests', 'conf', 'ssl4eo_s12_ijepa_1.yaml')
        monkeypatch.setattr(SSL4EOS12, '__len__', lambda self: 2)
        monkeypatch.setattr(timm, 'create_model', create_model)

        args = [
            '--config',
            config,
            '--model.init_args.warmup_epochs',
            '0',
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            'True',
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])

    def test_wrong_model_type(self) -> None:
        with pytest.raises(ValueError, match='not compatible with IJEPA'):
            IJEPATask(model='resnet18', weights=None)

    def test_wrong_model_type_2(self) -> None:
        with pytest.raises(
            ValueError, match="not compatible with IJEPA:', 'MultiScaleVit'"
        ):
            IJEPATask(model='mvitv2_tiny', weights=None)

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ViTSmall16_Weights.SENTINEL2_ALL_MAE  # No IJEPA weights available yet

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

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        match = 'num classes .* != num classes in pretrained model'
        with pytest.warns(UserWarning, match=match):
            IJEPATask(
                model=mocked_weights.meta['model'],
                weights=mocked_weights,
                in_channels=mocked_weights.meta['in_chans'],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        match = 'num classes .* != num classes in pretrained model'
        with pytest.warns(UserWarning, match=match):
            IJEPATask(
                model=mocked_weights.meta['model'],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta['in_chans'],
            )

    def test_ijepa_to_changevit(self, tmp_path: Path) -> None:
        """Test that an IJEPATask trained with Lightning can save a checkpoint
        and load its encoder weights into a ChangeDetectionTask."""
        ijepa = IJEPATask(model='vit_tiny_patch16_224', in_channels=3, size=256)

        class RandomImageDataset(Dataset):
            def __init__(self) -> None:
                self.num_samples = 4

            def __len__(self) -> int:
                return self.num_samples

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                return {'image': torch.randn(3, 256, 256)}

        dataloader = DataLoader(RandomImageDataset(), batch_size=2)
        trainer = Trainer(
            accelerator='cpu',
            max_epochs=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(ijepa, train_dataloaders=dataloader)

        ckpt_path = tmp_path / 'autosave.ckpt'
        trainer.save_checkpoint(str(ckpt_path))

        # This tests the on_load_checkpoint() overload
        trainer.fit(ijepa, train_dataloaders=dataloader, ckpt_path=ckpt_path)

        change = ChangeDetectionTask(
            model='changevit',
            backbone='vit_tiny_patch16_224',
            weights=str(ckpt_path),
            in_channels=3,
        )
        x = torch.randn(1, 2, 3, 256, 256)
        y = change(x)
        assert y.shape == (1, 1, 256, 256)

        ckpt_path.unlink()
