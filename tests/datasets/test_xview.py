# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, XView2, XView2DistShift


class TestXView2:
    @pytest.fixture(params=['train', 'test'])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> XView2:
        monkeypatch.setattr(
            XView2,
            'metadata',
            {
                'train': {
                    'filename': 'train_images_labels_targets.tar.gz',
                    'md5': '373e61d55c1b294aa76b94dbbd81332b',
                    'directory': 'train',
                },
                'test': {
                    'filename': 'test_images_labels_targets.tar.gz',
                    'md5': 'bc6de81c956a3bada38b5b4e246266a1',
                    'directory': 'test',
                },
            },
        )
        root = os.path.join('tests', 'data', 'xview2')
        split = request.param
        transforms = nn.Identity()
        return XView2(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: XView2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: XView2) -> None:
        assert len(dataset) == 2

    def test_extract(self, tmp_path: Path) -> None:
        shutil.copyfile(
            os.path.join(
                'tests', 'data', 'xview2', 'train_images_labels_targets.tar.gz'
            ),
            os.path.join(tmp_path, 'train_images_labels_targets.tar.gz'),
        )
        shutil.copyfile(
            os.path.join(
                'tests', 'data', 'xview2', 'test_images_labels_targets.tar.gz'
            ),
            os.path.join(tmp_path, 'test_images_labels_targets.tar.gz'),
        )
        XView2(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(
            os.path.join(tmp_path, 'train_images_labels_targets.tar.gz'), 'w'
        ) as f:
            f.write('bad')
        with open(
            os.path.join(tmp_path, 'test_images_labels_targets.tar.gz'), 'w'
        ) as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            XView2(root=tmp_path, checksum=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            XView2(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            XView2(tmp_path)

    def test_plot(self, dataset: XView2) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'][0].clone()
        dataset.plot(x)
        plt.close()


class TestXView2DistShift:
    @pytest.fixture(params=['train', 'test'])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> XView2DistShift:
        monkeypatch.setattr(
            XView2DistShift,
            'metadata',
            {
                'train': {
                    'filename': 'train_images_labels_targets.tar.gz',
                    'md5': '373e61d55c1b294aa76b94dbbd81332b',
                    'directory': 'train',
                },
                'test': {
                    'filename': 'test_images_labels_targets.tar.gz',
                    'md5': 'bc6de81c956a3bada38b5b4e246266a1',
                    'directory': 'test',
                },
            },
        )
        root = os.path.join('tests', 'data', 'xview2')
        split = request.param
        transforms = nn.Identity()

        return XView2DistShift(
            root=root,
            split=split,
            id_ood_disaster=[
                {'disaster_name': 'hurricane-harvey', 'pre-post': 'post'},
                {'disaster_name': 'hurricane-harvey', 'pre-post': 'post'},
            ],
            transforms=transforms,
            checksum=True,
        )

    def test_getitem(self, dataset: XView2DistShift) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert set(torch.unique(x['mask']).tolist()).issubset({0, 1})  # binary mask

    def test_len(self, dataset: XView2DistShift) -> None:
        assert len(dataset) > 0

    def test_invalid_disaster(self) -> None:
        with pytest.raises(ValueError, match='Invalid disaster name'):
            XView2DistShift(
                root='tests/data/xview2',
                id_ood_disaster=[
                    {'disaster_name': 'not-a-real-one', 'pre-post': 'post'},
                    {'disaster_name': 'hurricane-harvey', 'pre-post': 'post'},
                ],
            )

    def test_missing_disaster_name_key(self) -> None:
        with pytest.raises(
            ValueError, match="Each disaster entry must contain a 'disaster_name' key."
        ):
            XView2DistShift(
                root='tests/data/xview2',
                id_ood_disaster=[
                    {'pre-post': 'post'},  # missing 'disaster_name'
                    {'disaster_name': 'hurricane-harvey', 'pre-post': 'post'},
                ],
            )

    def test_missing_pre_post_key(self) -> None:
        with pytest.raises(
            ValueError,
            match="Each disaster entry must contain 'disaster_name' and 'pre-post' keys.",
        ):
            XView2DistShift(
                root='tests/data/xview2',
                id_ood_disaster=[
                    {'disaster_name': 'hurricane-harvey'},  # missing 'pre-post'
                    {'disaster_name': 'hurricane-harvey', 'pre-post': 'post'},
                ],
            )

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            XView2DistShift(
                root='tests/data/xview2',
                split='bad',  # type: ignore[arg-type]
                id_ood_disaster=[
                    {'disaster_name': 'hurricane-matthew', 'pre-post': 'post'},
                    {'disaster_name': 'mexico-earthquake', 'pre-post': 'post'},
                ],
            )
