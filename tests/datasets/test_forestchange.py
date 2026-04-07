# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for ForestChange."""

import os
import random
import shutil
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

matplotlib.use("Agg")

DATA_DIR = os.path.join("tests", "data", "forestchange")


class TestForestChange:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ForestChange:
        monkeypatch.setattr(
            ForestChange,
            "_download",
            lambda self: extract_archive(
                os.path.join(DATA_DIR, "Forest-Change-dataset.zip"),
                str(self.root),
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
            "image",
            "mask",
            "token",
            "token_all",
            "token_all_len",
            "token_len",
        ):
            assert key in sample, f"missing key: {key}"
        assert sample["image"].shape[0] == 2
        assert sample["image"].shape[1] == 3
        assert sample["mask"].shape[0] == 1
        assert sample["token"].shape[0] == dataset.max_length

    def test_len(self, dataset: ForestChange) -> None:
        assert len(dataset) == 2

    def test_mask_binary(self, dataset: ForestChange) -> None:
        assert set(dataset[0]["mask"].unique().tolist()).issubset({0, 1})

    def test_mask_dtype(self, dataset: ForestChange) -> None:
        assert dataset[0]["mask"].dtype == torch.int64

    def test_image_dtype(self, dataset: ForestChange) -> None:
        assert dataset[0]["image"].dtype == torch.float32

    def test_token_dtype(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert sample["token"].dtype == torch.int64
        assert sample["token_all"].dtype == torch.int64
        assert sample["token_len"].dtype == torch.int64

    def test_token_all_shape(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert sample["token_all"].ndim == 2
        assert sample["token_all"].shape[1] == dataset.max_length

    def test_token_len_scalar(self, dataset: ForestChange) -> None:
        assert dataset[0]["token_len"].ndim == 0

    def test_random_caption_selection(self, dataset: ForestChange) -> None:
        random.seed(0)
        tokens_seen = set()
        for _ in range(20):
            tokens_seen.add(tuple(dataset[0]["token"].numpy()))
        assert len(tokens_seen) > 1

    def test_indexed_caption_selection(self, dataset: ForestChange) -> None:
        dataset.files[0]["token_id"] = 1
        sample = dataset[0]
        assert torch.equal(sample["token"], sample["token_all"][1])

    def test_apply_max_iters_inflate(self, dataset: ForestChange) -> None:
        dataset._apply_max_iters(5)
        assert len(dataset) == 5
        assert any(
            "_aug" in f["name"] or "_rep" in f["name"] for f in dataset.files[2:]
        )

    def test_apply_max_iters_truncate(self, dataset: ForestChange) -> None:
        dataset._apply_max_iters(1)
        assert len(dataset) == 1

    def test_max_percent_samples(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            "_download",
            lambda self: extract_archive(
                os.path.join(DATA_DIR, "Forest-Change-dataset.zip"),
                str(self.root),
            ),
        )
        ds = ForestChange(
            root=tmp_path,
            split="train",
            max_percent_samples=50.0,
            download=True,
        )
        assert len(ds) == 1

    def test_transforms_applied(self, dataset: ForestChange) -> None:
        class AddOne:
            def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
                sample["image"] = sample["image"] + 1
                return sample

        dataset.transforms = AddOne()
        assert torch.all(dataset[0]["image"] >= 1)

    def test_fallback_normalisation(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            "_download",
            lambda self: extract_archive(
                os.path.join(DATA_DIR, "Forest-Change-dataset.zip"),
                str(self.root),
            ),
        )
        ds = ForestChange(root=tmp_path, split="train", download=True)
        assert ds[0]["image"].max().item() < 250.0

    def test_plot(self, dataset: ForestChange) -> None:
        fig = dataset.plot(dataset[0], suptitle="Test")
        plt.close(fig)

    def test_plot_with_prediction(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        fig = dataset.plot(sample)
        plt.close(fig)

    def test_plot_no_titles(self, dataset: ForestChange) -> None:
        fig = dataset.plot(dataset[0], show_titles=False)
        plt.close(fig)

    def test_already_downloaded(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            ForestChange,
            "_download",
            lambda self: extract_archive(
                os.path.join(DATA_DIR, "Forest-Change-dataset.zip"),
                str(self.root),
            ),
        )
        ForestChange(root=tmp_path, split="train", download=True)
        ForestChange(root=tmp_path, split="train", download=True)

    def test_preprocessing_skipped_on_second_load(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            ForestChange,
            "_download",
            lambda self: extract_archive(
                os.path.join(DATA_DIR, "Forest-Change-dataset.zip"),
                str(self.root),
            ),
        )
        _ = ForestChange(root=tmp_path, split="train", download=True)

        vocab_path = os.path.join(str(tmp_path), ForestChange.directory, "vocab.json")
        mtime = os.path.getmtime(vocab_path)
        ForestChange(root=tmp_path, split="train")
        assert os.path.getmtime(vocab_path) == mtime

    def test_preprocessing_not_repeated(self, tmp_path: Path) -> None:
        src = os.path.join(DATA_DIR, "Forest-Change-dataset")
        dst = os.path.join(tmp_path, "Forest-Change-dataset")

        shutil.copytree(src, dst)

        _ = ForestChange(root=tmp_path, split="train")

        vocab_path = os.path.join(dst, "vocab.json")
        mtime = os.path.getmtime(vocab_path)

        _ = ForestChange(root=tmp_path, split="train")
        assert os.path.getmtime(vocab_path) == mtime

    def test_check_integrity_fails_missing_captions(self, tmp_path: Path) -> None:
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_integrity() is False

    def test_check_preprocessed_fails_missing_vocab(self, tmp_path: Path) -> None:
        ds = ForestChange.__new__(ForestChange)
        ds.root = str(tmp_path)
        assert ds._check_preprocessed() is False

    def test_dataset_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            ForestChange(root=tmp_path)

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            ForestChange(root=tmp_path, split="invalid")

    def test_classes(self) -> None:
        assert ForestChange.classes == ["no_change", "deforestation"]

    def test_index_out_of_range(self, dataset: ForestChange) -> None:
        with pytest.raises(IndexError):
            dataset[999]

    def test_negative_index_raises(self, dataset: ForestChange) -> None:
        with pytest.raises(IndexError):
            dataset[-1]
