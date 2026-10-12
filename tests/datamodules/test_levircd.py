# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License
# .
import einops
import pytest
import torch
from lightning import Trainer

from torchgeo.datamodules import (
    LEVIRCDDataModule,
    LEVIRCDPlusDataModule,
    NonGeoDataModule,
)


class TestLEVIR:
    """Test that the LEVIR-CD augmentations work correctly for images and masks.

    For future reference: VideoSequential silently caused wrong mask crops due to expecting masks to be
    (b, t, c, h, w) while they were (b, c, h, w), where c is 1.
    """

    @pytest.fixture
    def make_mock_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Make 16x16 mask with a random 4x4 square checkerboard."""
        torch.manual_seed(0)
        mask = (torch.rand(4, 1, 4, 4) > 0.5).long()
        mask = torch.repeat_interleave(mask, 4, dim=2)
        mask = torch.repeat_interleave(mask, 4, dim=3)
        # img for cd needs to be (b, t, c, h, w), so just repeat mask in T dim twice and C dim 3 times
        # (and scale by 255 so it gets back to 0-1 due to norm in aug)
        image = einops.repeat(
            mask.float().clone() * 255, 'b c h w -> b t (r c) h w', t=2, r=3
        )
        return image, mask

    @pytest.fixture
    def make_mock_trainer(self) -> Trainer:
        trainer = Trainer(accelerator='cpu', max_epochs=1)
        trainer.training = True
        return trainer

    @pytest.mark.parametrize(
        'datamodule',
        [LEVIRCDDataModule(patch_size=5), LEVIRCDPlusDataModule(patch_size=5)],
    )
    def test_levir_datamodule_augs_work(
        self,
        datamodule: NonGeoDataModule,
        make_mock_data: tuple[torch.Tensor, torch.Tensor],
        make_mock_trainer: Trainer,
    ) -> None:
        """Test that the same aug is applied to both images and belonging mask."""
        image, mask = make_mock_data
        # trigger train aug
        datamodule.trainer = make_mock_trainer

        out = datamodule.on_after_batch_transfer(
            {'image': image.clone(), 'mask': mask.clone()}, 0
        )
        assert out['image'].shape == (4, 2, 3, 5, 5)
        assert out['mask'].shape == (4, 1, 5, 5)
        # image pair (which we set to be mask repeated) should be the same as mask after crop augments
        assert (out['image'][:, 0, 0] == out['mask'][:, 0]).all()
        assert (out['image'][:, 1, 0] == out['mask'][:, 0]).all()
