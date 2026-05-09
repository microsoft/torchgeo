# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for ForestChange."""

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, ForestChange

DATA_DIR = os.path.join('tests', 'data', 'forestchange')


class TestForestChange:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ForestChange:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        return ForestChange(
            root=tmp_path,
            split=request.param,
            transforms=nn.Identity(),
            max_length=42,
            download=True,
        )

    def test_getitem(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        for key in ('image', 'mask', 'token', 'token_all', 'token_len'):
            assert key in sample, f'missing key: {key}'
        assert sample['image'].shape[0] == 2
        assert sample['image'].shape[1] == 3
        assert sample['mask'].shape[0] == 1
        assert sample['token'].shape[0] == dataset.max_length

    def test_len(self, dataset: ForestChange) -> None:
        assert len(dataset) == 2

    def test_mask_dtype(self, dataset: ForestChange) -> None:
        assert dataset[0]['mask'].dtype == torch.int64

    def test_image_dtype(self, dataset: ForestChange) -> None:
        assert dataset[0]['image'].dtype == torch.float32

    def test_token_dtype(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert sample['token'].dtype == torch.int64
        assert sample['token_all'].dtype == torch.int64
        assert sample['token_len'].dtype == torch.int64

    def test_random_caption_selection(self, dataset: ForestChange) -> None:
        random.seed(0)
        tokens_seen = set()
        for _ in range(20):
            tokens_seen.add(tuple(dataset[0]['token'].numpy()))
        assert len(tokens_seen) > 1

    def test_indexed_caption_selection(self, dataset: ForestChange) -> None:
        dataset.files[0]['token_id'] = 1
        sample = dataset[0]
        assert torch.equal(sample['token'], sample['token_all'][1])

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

    def test_plot_with_prediction(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        fig = dataset.plot(sample)
        plt.close(fig)

    def test_already_downloaded(self, dataset: ForestChange) -> None:
        ForestChange(root=dataset.root, split=dataset.split, download=True)

    def test_preprocess_skips_empty_raw(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ForestChange(root=tmp_path, split='train', download=True)
        base = os.path.join(str(tmp_path), ForestChange.directory)
        captions_path = os.path.join(base, ForestChange.captions_filename)
        with open(captions_path) as f:
            data = json.load(f)
        data['images'][0]['sentences'].insert(0, {'raw': '', 'tokens': []})
        with open(captions_path, 'w') as f:
            json.dump(data, f)
        shutil.rmtree(os.path.join(base, ForestChange.token_directory))
        os.remove(os.path.join(base, ForestChange.vocab_filename + '.json'))
        ForestChange(root=tmp_path, split='train')

    def test_preprocess_rewrites_split_files(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ForestChange(root=tmp_path, split='train', download=True)
        base = os.path.join(str(tmp_path), ForestChange.directory)
        train_list = os.path.join(base, 'train.txt')
        mtime = os.path.getmtime(train_list)
        shutil.rmtree(os.path.join(base, ForestChange.token_directory))
        os.remove(os.path.join(base, ForestChange.vocab_filename + '.json'))
        time.sleep(0.05)
        ForestChange(root=tmp_path, split='train')
        assert os.path.getmtime(train_list) != mtime

    def test_integrity_missing_image_dir(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ForestChange(root=tmp_path, split='train', download=True)
        shutil.rmtree(
            os.path.join(str(tmp_path), ForestChange.directory, 'images', 'train', 'A')
        )
        with pytest.raises(DatasetNotFoundError):
            ForestChange(root=tmp_path, split='train')

    def test_preprocessed_missing_token_dir(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ForestChange(root=tmp_path, split='train', download=True)
        shutil.rmtree(
            os.path.join(
                str(tmp_path), ForestChange.directory, ForestChange.token_directory
            )
        )
        ForestChange(root=tmp_path, split='train')

    def test_preprocessed_missing_split_file(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ForestChange(root=tmp_path, split='train', download=True)
        os.remove(os.path.join(str(tmp_path), ForestChange.directory, 'train.txt'))
        ForestChange(root=tmp_path, split='train')

    def test_tokenize_preserves_numbers(self) -> None:
        tokens = ForestChange._tokenize(
            '42 trees removed', add_start_token=False, add_end_token=False
        )
        assert '42' in tokens

    def test_encode_allow_unknown(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ds = ForestChange(
            root=tmp_path, split='train', allow_unknown=True, download=True
        )
        assert ds._encode(['unknown_xyz'], ds.word_vocab) == [
            ForestChange.special_tokens['<UNK>']
        ]

    def test_encode_raises_for_unknown(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ds = ForestChange(root=tmp_path, split='train', download=True)
        with pytest.raises(KeyError):
            ds._encode(['unknown_xyz'], ds.word_vocab)

    def test_load_files_caption_index(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'forestchange', 'Forest-Change-dataset.zip')
        monkeypatch.setattr(ForestChange, 'url', url)
        ForestChange(root=tmp_path, split='train', download=True)
        base = os.path.join(str(tmp_path), ForestChange.directory)
        list_path = os.path.join(base, 'train.txt')
        with open(list_path) as f:
            first = f.readline().strip()
        with open(list_path, 'w') as f:
            f.write(f'{first}-2\n')
        ds = ForestChange(root=tmp_path, split='train')
        assert ds.files[0]['token_id'] == 2

    def test_dataset_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            ForestChange(root=tmp_path)

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            ForestChange(root=tmp_path, split='invalid')  # type: ignore

    def test_decode_tokens(self, dataset: ForestChange) -> None:
        sample = dataset[0]

        decoded = dataset._decode_tokens(sample['token'])

        assert isinstance(decoded, str)
        assert '<START>' not in decoded
        assert '<END>' not in decoded
