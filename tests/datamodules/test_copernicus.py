"""Tests for Copernicus datamodules."""

import kornia.augmentation as K
import torch

from torchgeo.datamodules.copernicus.biomass_s3 import (
    TARGET_SIZE,
    CopernicusBenchBiomassS3DataModule,
)
from torchgeo.datasets import CopernicusBenchBiomassS3

BANDS = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')


def test_existing_transform_is_composed() -> None:
    called = {'count': 0, 'shape': None}

    def existing_transform(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        called['count'] += 1
        called['shape'] = sample['image'].shape[-2:]
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

    assert called['count'] == 1
    assert called['shape'] == TARGET_SIZE
    assert sample['transformed']
    assert sample['image'].shape[-2:] == TARGET_SIZE


def test_time_series_uses_video_sequential() -> None:
    datamodule = CopernicusBenchBiomassS3DataModule(
        root='tests/data/copernicus/l3_biomass_s3',
        batch_size=1,
        num_workers=0,
        mode='time-series',
    )

    children = list(datamodule.aug.children())
    assert len(children) == 1

    video_seq = children[0]
    assert isinstance(video_seq, K.VideoSequential)

    normalize_layers = list(video_seq.children())
    assert normalize_layers
    assert isinstance(normalize_layers[0], K.Normalize)

    assert datamodule.aug.same_on_batch is True
