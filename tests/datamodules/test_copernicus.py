"""Tests for Copernicus datamodules."""

import shutil
from pathlib import Path
from typing import cast

import kornia.augmentation as K
import torch

from torchgeo.datamodules.copernicus.biomass_s3 import (
    TARGET_MEAN,
    TARGET_SIZE,
    TARGET_STD,
    CopernicusBenchBiomassS3DataModule,
)
from torchgeo.datasets import CopernicusBenchBiomassS3

BANDS = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')


def test_existing_transform_is_composed() -> None:
    count = 0
    shape: tuple[int, int] | None = None

    def existing_transform(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nonlocal count, shape
        count += 1
        shape = (int(sample['image'].shape[-2]), int(sample['image'].shape[-1]))
        sample['transformed'] = torch.tensor(True)
        return sample

    datamodule = CopernicusBenchBiomassS3DataModule(
        root='tests/data/copernicus/l3_biomass_s3',
        batch_size=1,
        num_workers=0,
        bands=BANDS,
        transforms=existing_transform,
    )

    composed = datamodule.kwargs['transforms']
    assert composed is not existing_transform

    dataset = CopernicusBenchBiomassS3(
        root='tests/data/copernicus/l3_biomass_s3',
        split='train',
        bands=BANDS,
        transforms=composed,
    )
    sample = dataset[0]

    assert count == 1
    assert shape == TARGET_SIZE
    assert sample['transformed']
    assert sample['image'].shape[-2:] == TARGET_SIZE


def test_time_series_uses_video_sequential() -> None:
    datamodule = CopernicusBenchBiomassS3DataModule(
        root='tests/data/copernicus/l3_biomass_s3',
        batch_size=1,
        num_workers=0,
        mode='time-series',
    )

    aug = cast(K.AugmentationSequential, datamodule.aug)
    children = list(aug.children())
    assert len(children) == 1

    video_seq = cast(K.VideoSequential, children[0])
    assert isinstance(video_seq, K.VideoSequential)

    normalize_layers = list(video_seq.children())
    assert normalize_layers
    assert isinstance(normalize_layers[0], K.Normalize)

    assert aug.same_on_batch is True


def test_time_series_collate_pads_variable_sequence_lengths(tmp_path: Path) -> None:
    src_root = Path('tests/data/copernicus/l3_biomass_s3')
    dst_root = tmp_path / 'l3_biomass_s3'
    shutil.copytree(src_root, dst_root)

    path = (
        dst_root
        / 'biomass_s3'
        / 's3_olci'
        / 'S32E141_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv4.0_01_05'
        / 'S3A_20210119T033546_20210119T033846.tif'
    )
    path.unlink()

    datamodule = CopernicusBenchBiomassS3DataModule(
        root=str(dst_root), batch_size=2, num_workers=0, mode='time-series', bands=BANDS
    )
    datamodule.setup('validate')

    batch = next(iter(datamodule.val_dataloader()))

    assert batch['image'].shape == (2, 2, len(BANDS), *TARGET_SIZE)
    assert batch['mask'].shape == (2, *TARGET_SIZE)
    assert batch['time'].shape == (2, 2)
    assert batch['lat'].shape == (2, 2)
    assert batch['lon'].shape == (2, 2)
    assert sorted((batch['time'] != 0).sum(dim=1).tolist()) == [1, 2]


def test_biomass_targets_are_normalized() -> None:
    datamodule = CopernicusBenchBiomassS3DataModule(
        root='tests/data/copernicus/l3_biomass_s3', batch_size=1, num_workers=0
    )
    batch = {
        'image': torch.zeros(1, len(BANDS), *TARGET_SIZE),
        'mask': torch.full((1, *TARGET_SIZE), (TARGET_MEAN + TARGET_STD).item()),
    }

    batch = datamodule.on_after_batch_transfer(batch, 0)

    assert torch.allclose(batch['mask'], torch.ones_like(batch['mask']))
