# Copyright (c) TorchGeo Contributors
# All rights reserved.
# Licensed under the MIT License.

"""Tests for blending utilities."""

import numpy as np
import pytest
import torch
from affine import Affine

from torchgeo.inference.blending import (
    _build_spatial_index,
    _query_spatial_index,
    _reconstruct_scene_from_patches,
    get_blend_mask,
)


class TestReconstructSceneFromPatches:
    """Tests for _reconstruct_scene_from_patches."""

    def test_single_patch(self) -> None:
        """Test reconstruction with single patch."""
        meta = [
            {
                'patch_id': 0,
                'bbox': (0, 0, 64, 64),
                'transform': torch.tensor([10.0, 0, 0, 0, -10.0, 1000]),
            }
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, delta=0)

        assert shape == (64, 64)
        assert transform == Affine(10.0, 0, 0, 0, -10.0, 1000)

    def test_two_patches_horizontal(self) -> None:
        """Test reconstruction with two horizontal patches."""
        meta = [
            {
                'patch_id': 0,
                'bbox': (0, 0, 64, 64),
                'transform': torch.tensor([1.0, 0, 100.0, 0, -1.0, 200.0]),
            },
            {
                'patch_id': 1,
                'bbox': (32, 0, 96, 64),
                'transform': torch.tensor([1.0, 0, 100.0, 0, -1.0, 200.0]),
            },
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, delta=0)

        assert shape == (64, 96)
        assert transform == Affine(1.0, 0, 100.0, 0, -1.0, 200.0)

    def test_inconsistent_resolutions_raises(self) -> None:
        """Test error on inconsistent resolutions."""
        meta = [
            {
                'patch_id': 0,
                'bbox': (0, 0, 64, 64),
                'transform': torch.tensor([1.0, 0, 0, 0, -1.0, 100]),
            },
            {
                'patch_id': 1,
                'bbox': (64, 0, 128, 64),
                'transform': torch.tensor([2.0, 0, 64, 0, -1.0, 100]),
            },
        ]

        with pytest.raises(ValueError, match='Inconsistent resolutions'):
            _reconstruct_scene_from_patches(meta, delta=0)


class TestGetBlendMask:
    """Tests for get_blend_mask."""

    def test_no_overlap(self) -> None:
        """Test blend mask with no overlap."""
        mask = get_blend_mask(64, overlap=0, delta=0, method='cosine')

        assert mask.shape == (64, 64)
        np.testing.assert_array_equal(mask, np.ones((64, 64)))

    def test_with_overlap_cosine(self) -> None:
        """Test cosine blend mask."""
        mask = get_blend_mask(64, overlap=8, delta=0, method='cosine')

        assert mask.shape == (64, 64)
        assert mask[0, 32] < mask[32, 32]
        assert mask[32, 32] == pytest.approx(1.0)


class TestSpatialIndexing:
    """Tests for spatial indexing utilities."""

    def test_build_and_query(self) -> None:
        """Test building and querying spatial index."""
        meta = [
            {'patch_id': 0, 'bbox': (0, 0, 64, 64)},
            {'patch_id': 1, 'bbox': (100, 100, 164, 164)},
        ]
        gdf = _build_spatial_index(meta)

        results = _query_spatial_index(
            gdf, chunk_y=0, chunk_x=0, chunk_h=64, chunk_w=64
        )

        assert len(results) == 1
        assert results[0]['patch_id'] == 0
