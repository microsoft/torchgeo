# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import cast

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import galileo, GalileoWeights


class TestGalileo:
    @pytest.fixture(params=[*GalileoWeights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        """
        Create a fake local checkpoint matching the Galileo encoder
        and monkeypatch the weights URL to point to it.
        """
        weights = GalileoWeights.GALILEO_S2_NANO_V1
        path = tmp_path / f"{weights}.pth"

        model = galileo(variant=weights.meta["variant"])
        torch.save(model.state_dict(), path)

        monkeypatch.setattr(weights.value, "url", str(path))
        return weights

    def test_galileo(self) -> None:
        galileo()

    def test_galileo_weights(self, mocked_weights: WeightsEnum) -> None:
        galileo(weights=mocked_weights)

    def test_forward_shape(self, weights: WeightsEnum) -> None:
        variant = weights.meta["variant"]
        embed_dim = weights.meta["embed_dim"]

        model = galileo(variant=variant)
        x = torch.randn(2, 4, 224, 224)
        y = model(x)

        assert y.shape == (2, embed_dim)

    def test_invalid_input_shape(self) -> None:
        model = galileo()
        x = torch.randn(1, 4, 256, 256)

        with pytest.raises(ValueError):
            model(x)

    def test_variant_weight_mismatch(self) -> None:
        with pytest.raises(ValueError):
            galileo(
                variant="tiny",
                weights=GalileoWeights.GALILEO_S2_BASE_V1,
            )

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta["in_channels"]
        sample = {
            "image": torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: WeightsEnum) -> None:
        """Ensure transforms are torch.export compatible."""
        torch = pytest.importorskip("torch", minversion="2.6.0")
        torch.compiler.reset()

        c = weights.meta["in_channels"]
        inputs = (torch.randn(1, c, 224, 224),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_galileo_download(self, weights: WeightsEnum) -> None:
        galileo(weights=weights)
