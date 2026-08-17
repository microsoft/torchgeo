# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor, nn

from torchgeo.datasets import DatasetNotFoundError, ForestChange

DATA_DIR = os.path.join('tests', 'data', 'forestchange')

tokenizers = pytest.importorskip('tokenizers', minversion='0.14')


class TestForestChange:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> ForestChange:
        url = os.path.join(DATA_DIR, 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        return ForestChange(
            root=tmp_path, split='train', transforms=nn.Identity(), download=True
        )

    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset_split(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ForestChange:
        url = os.path.join(DATA_DIR, 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        return ForestChange(
            root=tmp_path, split=request.param, transforms=nn.Identity(), download=True
        )

    def test_getitem(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        for key in ('image', 'mask', 'caption'):
            assert key in sample, f'missing key: {key}'
        assert sample['image'].shape[0] == 2
        assert sample['image'].shape[1] == 3
        assert sample['mask'].shape[0] == 1
        assert isinstance(sample['image'], Tensor)
        assert isinstance(sample['mask'], Tensor)
        assert isinstance(sample['caption'], Tensor)

    def test_len(self, dataset: ForestChange) -> None:
        assert len(dataset) == 2

    @pytest.mark.parametrize(
        ('key', 'dtype'),
        [('mask', torch.int64), ('image', torch.float32), ('caption', torch.int64)],
    )
    def test_dtypes(self, dataset: ForestChange, key: str, dtype: torch.dtype) -> None:
        assert dataset[0][key].dtype == dtype

    def test_tokenizer(self, dataset: ForestChange) -> None:
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        ds = ForestChange(root=dataset.root, split=dataset.split, tokenizer=tokenizer)
        sample = ds[0]
        assert isinstance(sample, dict)
        assert isinstance(sample['image'], Tensor)
        assert isinstance(sample['mask'], Tensor)
        assert isinstance(sample['caption'], Tensor)

    def test_caption_selection(self, dataset_split: ForestChange) -> None:
        tokens_seen = {tuple(dataset_split[0]['caption'].numpy()) for _ in range(20)}
        if dataset_split.split == 'train':
            assert len(tokens_seen) > 1
        else:
            assert len(tokens_seen) == 1

    def test_caption_iterator(self, dataset: ForestChange) -> None:
        captions = list(dataset._caption_iterator('train'))
        assert len(captions) > 0
        assert all(isinstance(c, str) for c in captions)

    def test_transforms_applied(self, dataset: ForestChange) -> None:
        class AddOne:
            def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
                sample['image'] = sample['image'] + 1
                return sample

        dataset.transforms = AddOne()
        assert torch.all(dataset[0]['image'] >= 1)

    def test_plot(self, dataset: ForestChange) -> None:
        fig = dataset.plot(dataset[0], suptitle='Test')
        assert len(fig.texts) > 0
        plt.close(fig)

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        sample['caption_prediction'] = sample['caption'].clone()

        fig = dataset.plot(sample)
        caption_fig_text = fig.texts[-1].get_text()
        assert 'Predicted:' in caption_fig_text
        plt.close(fig)

    def test_already_downloaded(self, dataset: ForestChange) -> None:
        ForestChange(root=dataset.root, split=dataset.split, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            ForestChange(root=tmp_path, split='train')

    def test_integrity_missing_image_dir(self, dataset_split: ForestChange) -> None:
        base = Path(dataset_split.root) / ForestChange.directory
        shutil.rmtree(base / 'images' / dataset_split.split / 'A')
        with pytest.raises(DatasetNotFoundError):
            ForestChange(root=dataset_split.root, split=dataset_split.split)

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            ForestChange(root=tmp_path, split='invalid')  # type: ignore

    def test_load_tokens_invalid_caption_index(self, dataset: ForestChange) -> None:
        ds = dataset
        ds.files[0]['token_id'] = 999
        with pytest.raises(ValueError, match='out of range'):
            ds[0]

    def test_load_tokens_empty_caption_list(self, dataset: ForestChange) -> None:
        with pytest.raises(ValueError, match='No captions available'):
            dataset._load_tokens([], None)
