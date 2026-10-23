# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import MARSS2L, DatasetNotFoundError


class TestMARSS2L:
    @pytest.fixture(
        params=[
            # split, load_ch4, load_cloudmask, load_wind, load_fluxrate
            ('train', False, False, False, False),
            ('train', True, True, True, True),
            ('val', True, False, True, False),
            ('test', False, True, False, True),
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> MARSS2L:
        monkeypatch.setattr(
            MARSS2L, 'url', os.path.join('tests', 'data', 'mars-s2l', '')
        )
        split, ch4, cloudmask, wind, flux = request.param
        return MARSS2L(
            tmp_path,
            split=split,
            transforms=nn.Identity(),
            load_ch4=ch4,
            load_cloudmask=cloudmask,
            load_wind=wind,
            load_fluxrate=flux,
            download=True,
        )

    def test_getitem(self, dataset: MARSS2L) -> None:
        # the last sample contains a plume, the others do not
        for index in range(len(dataset)):
            x = dataset[index]
            assert isinstance(x, dict)
            assert isinstance(x['image'], torch.Tensor)
            assert isinstance(x['mask'], torch.Tensor)
            assert isinstance(x['is_plume'], torch.Tensor)
            # 6 plume scene bands followed by 6 reference scene bands
            assert x['image'].shape == (12, 200, 200)
            assert x['mask'].shape == (200, 200)
            assert ('ch4' in x) == dataset.load_ch4
            assert ('cloudmask' in x) == dataset.load_cloudmask
            assert ('wind_u' in x) == dataset.load_wind
            assert ('wind_v' in x) == dataset.load_wind
            assert ('fluxrate' in x) == dataset.load_fluxrate
            assert ('flux_std' in x) == dataset.load_fluxrate

    def test_len(self, dataset: MARSS2L) -> None:
        assert len(dataset) == 3

    def test_already_downloaded(self, dataset: MARSS2L) -> None:
        MARSS2L(dataset.root, split=dataset.split, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MARSS2L(tmp_path)

    def test_plot(self, dataset: MARSS2L) -> None:
        x = dataset[0]
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
