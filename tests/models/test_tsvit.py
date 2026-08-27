# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch

from torchgeo.models.tsvit import TSViT, convert_tsvit_checkpoint, load_tsvit_checkpoint


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

    @staticmethod
    def _make_legacy_state_dict(model: TSViT) -> dict[str, torch.Tensor]:
        """Create a checkpoint using the original TSViT parameter names."""
        current = model.state_dict()
        legacy = {}

        direct_keys = [
            'temporal_token',
            'space_pos_embedding',
            'to_patch_embedding.1.weight',
            'to_patch_embedding.1.bias',
            'to_temporal_embedding_input.weight',
            'to_temporal_embedding_input.bias',
            'mlp_head.0.weight',
            'mlp_head.0.bias',
            'mlp_head.1.weight',
            'mlp_head.1.bias',
        ]

        for key in direct_keys:
            legacy[key] = current[key].clone()

        def unmap_transformer(prefix: str, depth: int) -> None:
            for i in range(depth):
                new = f'{prefix}.{i}'
                old = f'{prefix}.layers.{i}'

                mappings = {
                    f'{new}.norm1.weight': f'{old}.0.norm.weight',
                    f'{new}.norm1.bias': f'{old}.0.norm.bias',
                    f'{new}.attn.qkv.weight': f'{old}.0.fn.to_qkv.weight',
                    f'{new}.attn.proj.weight': f'{old}.0.fn.to_out.0.weight',
                    f'{new}.attn.proj.bias': f'{old}.0.fn.to_out.0.bias',
                    f'{new}.norm2.weight': f'{old}.1.norm.weight',
                    f'{new}.norm2.bias': f'{old}.1.norm.bias',
                    f'{new}.mlp.fc1.weight': f'{old}.1.fn.net.0.weight',
                    f'{new}.mlp.fc1.bias': f'{old}.1.fn.net.0.bias',
                    f'{new}.mlp.fc2.weight': f'{old}.1.fn.net.3.weight',
                    f'{new}.mlp.fc2.bias': f'{old}.1.fn.net.3.bias',
                }

                for new_key, old_key in mappings.items():
                    legacy[old_key] = current[new_key].clone()

            for suffix in ('weight', 'bias'):
                legacy[f'{prefix}.norm.{suffix}'] = current[
                    f'{prefix}.{depth}.{suffix}'
                ].clone()

        unmap_transformer('temporal_transformer', model.temporal_depth)
        unmap_transformer('space_transformer', model.spatial_depth)

        return legacy

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

    def test_checkpoint_conversion_roundtrip(self, model: TSViT) -> None:
        """Test conversion from original checkpoint key names."""
        current = model.state_dict()
        legacy = self._make_legacy_state_dict(model)

        converted = convert_tsvit_checkpoint(legacy, model)

        assert converted.keys() == current.keys()
        for key in current:
            assert torch.equal(converted[key], current[key])

    def test_missing_checkpoint_key(self, model: TSViT) -> None:
        """Test missing direct checkpoint parameters."""
        legacy = self._make_legacy_state_dict(model)
        legacy.pop('temporal_token')

        with pytest.raises(KeyError, match='Missing checkpoint key'):
            convert_tsvit_checkpoint(legacy, model)

    def test_missing_transformer_checkpoint_key(self, model: TSViT) -> None:
        """Test missing Transformer checkpoint parameters."""
        legacy = self._make_legacy_state_dict(model)
        legacy.pop('temporal_transformer.layers.0.0.fn.to_qkv.weight')

        with pytest.raises(KeyError, match='Missing checkpoint key'):
            convert_tsvit_checkpoint(legacy, model)

    def test_checkpoint_shape_mismatch(self, model: TSViT) -> None:
        """Test validation of converted checkpoint tensor shapes."""
        legacy = self._make_legacy_state_dict(model)
        legacy['temporal_token'] = torch.empty(1, 19, 1)

        with pytest.raises(ValueError, match='Shape mismatch'):
            convert_tsvit_checkpoint(legacy, model)

    def test_load_checkpoint(self, model: TSViT, tmp_path: Path) -> None:
        """Test loading a checkpoint through the public loader."""
        legacy = self._make_legacy_state_dict(model)
        checkpoint_path = tmp_path / 'tsvit.pth'
        torch.save(legacy, checkpoint_path)

        load_tsvit_checkpoint(model, checkpoint_path)
