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

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, ForestChange
from torchgeo.datasets.utils import extract_archive

matplotlib.use('Agg')

DATA_DIR = os.path.join('tests', 'data', 'forestchange')


class TestForestChange:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ForestChange:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        split = request.param
        return ForestChange(
            root=tmp_path,
            split=split,
            transforms=nn.Identity(),
            max_length=42,
            download=True,
        )

    def test_getitem(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        for key in (
            'image',
            'mask',
            'token',
            'token_all',
            'token_all_len',
            'token_len',
        ):
            assert key in sample, f'missing key: {key}'
        assert sample['image'].shape[0] == 2
        assert sample['image'].shape[1] == 3
        assert sample['mask'].shape[0] == 1
        assert sample['token'].shape[0] == dataset.max_length

    def test_len(self, dataset: ForestChange) -> None:
        assert len(dataset) == 2

    def test_mask_binary(self, dataset: ForestChange) -> None:
        assert set(dataset[0]['mask'].unique().tolist()).issubset({0, 1})

    def test_mask_dtype(self, dataset: ForestChange) -> None:
        assert dataset[0]['mask'].dtype == torch.int64

    def test_image_dtype(self, dataset: ForestChange) -> None:
        assert dataset[0]['image'].dtype == torch.float32

    def test_token_dtype(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert sample['token'].dtype == torch.int64
        assert sample['token_all'].dtype == torch.int64
        assert sample['token_len'].dtype == torch.int64

    def test_token_all_shape(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert sample['token_all'].ndim == 2
        assert sample['token_all'].shape[1] == dataset.max_length

    def test_token_len_scalar(self, dataset: ForestChange) -> None:
        assert dataset[0]['token_len'].ndim == 0

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

    def test_apply_max_iters_inflate(self, dataset: ForestChange) -> None:
        dataset._apply_max_iters(5)
        assert len(dataset) == 5
        assert any(
            '_aug' in f['name'] or '_rep' in f['name'] for f in dataset.files[2:]
        )

    def test_apply_max_iters_truncate(self, dataset: ForestChange) -> None:
        dataset._apply_max_iters(1)
        assert len(dataset) == 1

    def test_max_iters_via_init(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ds = ForestChange(root=tmp_path, split='train', max_iters=5, download=True)
        assert len(ds) == 5

    def test_max_percent_samples(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ds = ForestChange(
            root=tmp_path, split='train', max_percent_samples=50.0, download=True
        )
        assert len(ds) == 1

    def test_transforms_applied(self, dataset: ForestChange) -> None:
        class AddOne:
            def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
                sample['image'] = sample['image'] + 1
                return sample

        dataset.transforms = AddOne()
        assert torch.all(dataset[0]['image'] >= 1)

    def test_fallback_normalisation(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ds = ForestChange(root=tmp_path, split='train', download=True)
        assert ds[0]['image'].max().item() < 250.0

    def test_plot(self, dataset: ForestChange) -> None:
        fig = dataset.plot(dataset[0], suptitle='Test')
        plt.close(fig)

    def test_plot_with_prediction(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        fig = dataset.plot(sample)
        plt.close(fig)

    def test_plot_no_titles(self, dataset: ForestChange) -> None:
        fig = dataset.plot(dataset[0], show_titles=False)
        plt.close(fig)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ForestChange(root=tmp_path, split='train', download=True)
        ForestChange(root=tmp_path, split='train', download=True)

    def test_preprocessing_skipped_on_second_load(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        _ = ForestChange(root=tmp_path, split='train', download=True)

        vocab_path = os.path.join(str(tmp_path), ForestChange.directory, 'vocab.json')
        mtime = os.path.getmtime(vocab_path)
        ForestChange(root=tmp_path, split='train')
        assert os.path.getmtime(vocab_path) == mtime

    def test_preprocessing_not_repeated(self, tmp_path: Path) -> None:
        src = os.path.join(DATA_DIR, 'Forest-Change-dataset')
        dst = os.path.join(tmp_path, 'Forest-Change-dataset')

        shutil.copytree(src, dst)

        _ = ForestChange(root=tmp_path, split='train')

        vocab_path = os.path.join(dst, 'vocab.json')
        mtime = os.path.getmtime(vocab_path)

        _ = ForestChange(root=tmp_path, split='train')
        assert os.path.getmtime(vocab_path) == mtime

    def test_preprocess_removes_and_rewrites_split_files(self, tmp_path: Path) -> None:
        src = os.path.join(DATA_DIR, 'Forest-Change-dataset')
        dst = os.path.join(str(tmp_path), ForestChange.directory)
        shutil.copytree(src, dst)

        ForestChange(root=tmp_path, split='train')

        base = os.path.join(str(tmp_path), ForestChange.directory)
        train_list = os.path.join(base, 'train.txt')
        mtime_after_first = os.path.getmtime(train_list)

        # Force _preprocess to run again by removing its outputs
        shutil.rmtree(os.path.join(base, ForestChange.token_directory))
        os.remove(os.path.join(base, ForestChange.vocab_filename + '.json'))
        time.sleep(0.05)

        ForestChange(root=tmp_path, split='train')
        assert os.path.getmtime(train_list) != mtime_after_first

    def test_preprocess_skips_empty_raw_captions(self, tmp_path: Path) -> None:
        src = os.path.join(DATA_DIR, 'Forest-Change-dataset')
        dst = os.path.join(str(tmp_path), ForestChange.directory)
        shutil.copytree(src, dst)

        captions_path = os.path.join(dst, ForestChange.captions_filename)
        with open(captions_path) as f:
            data: dict[str, Any] = json.load(f)

        # Inject an empty-raw sentence; _preprocess must skip it without error
        data['images'][0]['sentences'].insert(0, {'raw': '', 'tokens': []})
        with open(captions_path, 'w') as f:
            json.dump(data, f)

        # Constructing the dataset triggers _preprocess; no error means skip worked
        ForestChange(root=tmp_path, split='train')

    def test_check_integrity_fails_missing_captions(self, tmp_path: Path) -> None:
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_integrity() is False

    def test_check_integrity_fails_missing_image_directory(
        self, tmp_path: Path
    ) -> None:
        base = os.path.join(str(tmp_path), ForestChange.directory)
        os.makedirs(base, exist_ok=True)
        # Captions file present so the first check passes, but image dirs absent
        with open(os.path.join(base, ForestChange.captions_filename), 'w') as f:
            json.dump({}, f)
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_integrity() is False

    def test_check_preprocessed_fails_missing_vocab(self, tmp_path: Path) -> None:
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_preprocessed() is False

    def test_check_preprocessed_fails_missing_token_dir(self, tmp_path: Path) -> None:
        base = os.path.join(str(tmp_path), ForestChange.directory)
        os.makedirs(base, exist_ok=True)
        # Vocab present but token directory absent
        with open(os.path.join(base, ForestChange.vocab_filename + '.json'), 'w') as f:
            json.dump({}, f)
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_preprocessed() is False

    def test_check_preprocessed_fails_missing_split_file(self, tmp_path: Path) -> None:
        base = os.path.join(str(tmp_path), ForestChange.directory)
        os.makedirs(os.path.join(base, ForestChange.token_directory), exist_ok=True)
        # Vocab and token dir present but no split .txt files
        with open(os.path.join(base, ForestChange.vocab_filename + '.json'), 'w') as f:
            json.dump({}, f)
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_preprocessed() is False

    def test_download_prints_when_already_present(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        extract_archive(
            os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(tmp_path)
        )
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        ds.checksum = False
        ds._download()
        assert 'already downloaded' in capsys.readouterr().out

    def test_download_calls_download_and_extract_archive(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_download_and_extract(
            url: str, root: Any, filename: str, md5: Any
        ) -> None:
            calls.append({'url': url, 'md5': md5})
            extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(root)
            )

        monkeypatch.setattr(
            'torchgeo.datasets.forestchange.download_and_extract_archive',
            fake_download_and_extract,
        )
        ForestChange(root=tmp_path, split='train', download=True, checksum=False)
        assert len(calls) == 1
        assert calls[0]['url'] == ForestChange.url
        assert calls[0]['md5'] is None

    def test_tokenize_preserves_numbers(self) -> None:
        tokens = ForestChange._tokenize(
            '42 trees removed', add_start_token=False, add_end_token=False
        )
        assert '42' in tokens

    def test_tokenize_preserves_decimal_numbers(self) -> None:
        tokens = ForestChange._tokenize(
            '3.5 hectares lost', add_start_token=False, add_end_token=False
        )
        assert '3.5' in tokens

    def test_encode_maps_unknown_to_unk_when_allowed(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ds = ForestChange(root=tmp_path, split='train', allow_unk=True, download=True)
        unk_idx = ForestChange.special_tokens['<UNK>']
        assert ds._encode(['totally_unknown_xyz'], ds.word_vocab) == [unk_idx]

    def test_encode_raises_for_unknown_token(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ds = ForestChange(root=tmp_path, split='train', allow_unk=False, download=True)
        with pytest.raises(KeyError, match='not in vocab'):
            ds._encode(['totally_unknown_xyz'], ds.word_vocab)

    def test_load_files_parses_caption_index_for_train(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            '_download',
            lambda self: extract_archive(
                os.path.join(DATA_DIR, 'Forest-Change-dataset.zip'), str(self.root)
            ),
        )
        ForestChange(root=tmp_path, split='train', download=True)

        # Overwrite the train split file with a caption-index suffixed entry
        base = os.path.join(str(tmp_path), ForestChange.directory)
        list_path = os.path.join(base, 'train.txt')
        with open(list_path) as f:
            first_filename = f.readline().strip()
        with open(list_path, 'w') as f:
            f.write(f'{first_filename}-3\n')

        ds = ForestChange(root=tmp_path, split='train')
        assert ds.files[0]['token_id'] == 3

    def test_dataset_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            ForestChange(root=tmp_path)

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            ForestChange(root=tmp_path, split='invalid')

    def test_classes(self) -> None:
        assert ForestChange.classes == ['no_change', 'deforestation']

    def test_index_out_of_range(self, dataset: ForestChange) -> None:
        with pytest.raises(IndexError):
            dataset[999]

    def test_negative_index_raises(self, dataset: ForestChange) -> None:
        with pytest.raises(IndexError):
            dataset[-1]
