# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models.tsvit import TSViT, TSViT_Weights, tsvit


class TestTSViT:
    @pytest.fixture
    def model(self) -> TSViT:
        return TSViT(
            img_res=24,
            patch_size=2,
            num_channels=11,
            num_classes=19,
            max_seq_len=60,
            dim=128,
            temporal_depth=4,
            spatial_depth=4,
            heads=4,
            dim_head=32,
        )

    def test_forward(self, model: TSViT) -> None:
        """Test a full TSViT forward pass."""
        model.eval()
        x = torch.rand(1, 60, 24, 24, 11)

        with torch.no_grad():
            y = model(x)

        assert y.shape == (1, 19, 24, 24)
        assert torch.isfinite(y).all()

    def test_invalid_patch_size(self) -> None:
        """Test validation of the spatial patch size."""
        with pytest.raises(ValueError, match='divisible by patch size'):
            TSViT(img_res=24, patch_size=5)

    def test_invalid_attention_dimensions(self) -> None:
        """Test validation of attention dimensions."""
        with pytest.raises(ValueError, match='dim must equal heads'):
            TSViT(dim=128, heads=3, dim_head=32)

    def test_invalid_input_dimensions(self, model: TSViT) -> None:
        """Test validation of the input tensor rank."""
        with pytest.raises(ValueError, match='Expected input shape'):
            model(torch.rand(1, 60, 24, 24))

    def test_invalid_spatial_size(self, model: TSViT) -> None:
        """Test validation of the input spatial size."""
        with pytest.raises(ValueError, match='Expected spatial size'):
            model(torch.rand(1, 60, 20, 20, 11))

    def test_invalid_channels(self, model: TSViT) -> None:
        """Test validation of the number of input channels."""
        with pytest.raises(ValueError, match='Expected 11 input channels'):
            model(torch.rand(1, 60, 24, 24, 10))

    def test_too_many_frames(self, model: TSViT) -> None:
        """Test validation of the temporal sequence length."""
        with pytest.raises(ValueError, match='Expected at most 60'):
            model(torch.rand(1, 61, 24, 24, 11))

    def test_weights(self) -> None:
        """Test TSViT pretrained weights."""
        weights = TSViT_Weights.TSVIT_PASTIS24

        assert weights.url.endswith('tsvit_pastis24_fold1-026e8447.pth')
        assert weights.meta['dataset'] == 'PASTIS24'
        assert weights.meta['model'] == 'TSViT'
        assert weights.meta['publication'] == 'https://arxiv.org/abs/2301.04944'
        assert weights.meta['repo'] == 'https://github.com/michaeltrs/DeepSatModels'
        assert isinstance(weights.transforms, torch.nn.Identity)

    def test_factory(self) -> None:
        """Test TSViT model factory."""
        model = tsvit()

        assert isinstance(model, TSViT)

    def test_factory_with_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test TSViT model factory with pretrained weights."""
        expected = TSViT()

        def get_state_dict(**kwargs: object) -> dict[str, torch.Tensor]:
            return expected.state_dict()

        monkeypatch.setattr(
            TSViT_Weights.TSVIT_PASTIS24, 'get_state_dict', get_state_dict
        )

        model = tsvit(weights=TSViT_Weights.TSVIT_PASTIS24)

        assert isinstance(model, TSViT)
        assert model.state_dict().keys() == expected.state_dict().keys()

        for key in expected.state_dict():
            assert torch.equal(model.state_dict()[key], expected.state_dict()[key])
