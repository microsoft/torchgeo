# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import pytest

from torchgeo.datamodules import HabitAlp2DataModule


class TestHabitAlp2DataModule:
    @pytest.fixture
    def datamodule_seg(self, tmp_path: Path) -> HabitAlp2DataModule:
        src = os.path.join('tests', 'data', 'habitalp')
        for folder in ['data_2003', 'data_2013', 'data_2020', 'labels']:
            src_folder = os.path.join(src, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)

        return HabitAlp2DataModule(
            root=tmp_path,
            batch_size=1,
            patch_size=32,
            length=1,
            num_workers=0,
            task='segmentation',
            year='2013',
        )

    @pytest.fixture
    def datamodule_cd(self, tmp_path: Path) -> HabitAlp2DataModule:
        src = os.path.join('tests', 'data', 'habitalp')
        for folder in ['data_2003', 'data_2013', 'data_2020', 'labels']:
            src_folder = os.path.join(src, folder)
            dst_folder = os.path.join(tmp_path, folder)
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)

        return HabitAlp2DataModule(
            root=tmp_path,
            batch_size=1,
            patch_size=32,
            length=1,
            num_workers=0,
            task='change_detection',
            pair='2013_2020',
        )

    def test_init_segmentation(self, datamodule_seg: HabitAlp2DataModule) -> None:
        assert datamodule_seg.task == 'segmentation'

    def test_init_change_detection(self, datamodule_cd: HabitAlp2DataModule) -> None:
        assert datamodule_cd.task == 'change_detection'

    def test_invalid_task(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='segmentation'):
            HabitAlp2DataModule(root=tmp_path, task='invalid')

    def test_setup_fit(self, datamodule_seg: HabitAlp2DataModule) -> None:
        datamodule_seg.setup('fit')
        assert datamodule_seg.train_dataset is not None
        assert datamodule_seg.val_dataset is not None

    def test_setup_validate(self, datamodule_seg: HabitAlp2DataModule) -> None:
        datamodule_seg.setup('validate')
        assert datamodule_seg.val_dataset is not None

    def test_setup_test(self, datamodule_seg: HabitAlp2DataModule) -> None:
        datamodule_seg.setup('test')
        assert datamodule_seg.test_dataset is not None
