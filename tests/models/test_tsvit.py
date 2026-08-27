# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models.tsvit import TSViT, convert_tsvit_checkpoint


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

    def test_checkpoint_conversion_roundtrip(self, model: TSViT) -> None:
        """Test conversion from original checkpoint key names.

        This creates a synthetic checkpoint with the original TSViT key layout
        from the TorchGeo model parameters, then converts it back and verifies
        that all tensors are recovered exactly.
        """
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

        converted = convert_tsvit_checkpoint(legacy, model)

        assert converted.keys() == current.keys()
        for key in current:
            assert torch.equal(converted[key], current[key])
