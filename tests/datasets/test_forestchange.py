# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, ForestChange

DATA_DIR = os.path.join("tests", "data", "forestchange")

tokenizers = pytest.importorskip("tokenizers", minversion="0.14")


class TestForestChange:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ForestChange:
        # Point the dataset to the on-disk test zip fixture
        url = os.path.join(DATA_DIR, "Forest-Change-dataset.zip")
        monkeypatch.setattr(ForestChange, "url", url)

        ds = ForestChange(
            root=tmp_path,
            split=request.param,
            transforms=nn.Identity(),
            max_length=42,
            download=True,
        )
        return ds

    def test_getitem(self, dataset: ForestChange) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        for key in ("image", "mask", "token", "token_all", "token_len"):
            assert key in sample, f"missing key: {key}"
        assert sample["image"].shape[0] == 2
        assert sample["image"].shape[1] == 3
        assert sample["mask"].shape[0] == 1
        assert sample["token"].shape[0] == dataset.max_length

    def test_len(self, dataset: ForestChange) -> None:
        assert len(dataset) == 2

    @pytest.mark.parametrize(
        ("key", "dtype"),
        [
            ("mask", torch.int64),
            ("image", torch.float32),
            ("token", torch.int64),
            ("token_all", torch.int64),
            ("token_len", torch.int64),
        ],
    )
    def test_dtypes(self, dataset: ForestChange, key: str, dtype: torch.dtype) -> None:
        assert dataset[0][key].dtype == dtype

    def test_caption_selection(self, dataset: ForestChange) -> None:
        random.seed(0)
        tokens_seen = {tuple(dataset[0]["token"].numpy()) for _ in range(20)}
        assert len(tokens_seen) > 1

        dataset.files[0]["token_id"] = 1
        sample = dataset[0]
        assert torch.equal(sample["token"], sample["token_all"][1])

    def test_transforms_applied(self, dataset: ForestChange) -> None:
        class AddOne:
            def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
                sample["image"] = sample["image"] + 1
                return sample

        dataset.transforms = AddOne()
        assert torch.all(dataset[0]["image"] >= 1)

    def test_plot(self, dataset: ForestChange) -> None:
        fig = dataset.plot(dataset[0], suptitle="Test")
        assert len(fig.texts) > 0
        plt.close(fig)

        sample = dataset[0]
        sample["prediction"] = sample["mask"].clone()
        fig = dataset.plot(sample)
        plt.close(fig)

    def test_already_downloaded(self, dataset: ForestChange) -> None:
        ForestChange(
            root=dataset.root,
            split=dataset.split,
            download=True,
        )

    def test_preprocess_skips_empty_raw(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        captions_path = base / ForestChange.captions_filename

        with open(captions_path) as f:
            data = json.load(f)

        data["images"][0]["sentences"].insert(0, {"raw": ""})

        with open(captions_path, "w") as f:
            json.dump(data, f)

        shutil.rmtree(base / ForestChange.token_directory, ignore_errors=True)

        ForestChange(root=dataset.root, split=dataset.split)

    def test_preprocess_rewrites_split_files(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        split_list = base / f"{dataset.split}.txt"

        with open(split_list) as f:
            original = f.read()

        shutil.rmtree(base / ForestChange.token_directory, ignore_errors=True)

        with open(split_list, "w") as f:
            f.write("corrupted")

        ForestChange(
            root=dataset.root,
            split=dataset.split,
        )

        with open(split_list) as f:
            rewritten = f.read()

        assert rewritten != "corrupted"
        assert rewritten == original

    def test_integrity_missing_image_dir(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        shutil.rmtree(base / "images" / dataset.split / "A")
        with pytest.raises(DatasetNotFoundError):
            ForestChange(
                root=dataset.root,
                split=dataset.split,
            )

    def test_preprocessed_missing_token_dir(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        shutil.rmtree(base / ForestChange.token_directory, ignore_errors=True)
        ForestChange(
            root=dataset.root,
            split=dataset.split,
        )

    def test_preprocessed_missing_split_file(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        try:
            os.remove(base / f"{dataset.split}.txt")
        except Exception:
            pass
        ForestChange(
            root=dataset.root,
            split=dataset.split,
        )

    def test_load_files_caption_index(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        with open(base / f"{dataset.split}.txt") as f:
            first = f.readline().strip()

        with open(base / f"{dataset.split}.txt", "w") as f:
            f.write(f"{first}-2\n")
        ds = ForestChange(
            root=dataset.root,
            split=dataset.split,
        )
        assert ds.files[0]["token_id"] == 2

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            ForestChange(root=tmp_path, split="invalid")  # type: ignore

    def test_load_tokens_truncates_to_max_length(self, dataset: ForestChange) -> None:
        ds = ForestChange(
            root=dataset.root,
            split=dataset.split,
            download=False,
            max_length=4,
        )
        sample = ds[0]
        assert sample["token"].shape[0] == 4
        assert sample["token_len"] <= 4
        assert torch.equal(
            sample["token"][sample["token_len"] :],
            torch.zeros(4 - sample["token_len"], dtype=torch.int64),
        )

    def test_load_tokens_invalid_caption_index(self, dataset: ForestChange) -> None:
        ds = dataset
        ds.files[0]["token_id"] = 999
        with pytest.raises(ValueError, match="out of range"):
            ds[0]

    def test_load_tokens_empty_caption_list(self, dataset: ForestChange) -> None:
        base = Path(dataset.root) / ForestChange.directory
        token_path = base / "empty.json"
        with open(token_path, "w") as f:
            json.dump([], f)
        with pytest.raises(ValueError, match="No captions available"):
            dataset._load_tokens(token_path, None)
