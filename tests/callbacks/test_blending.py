# Copyright (c) TorchGeo Contributors
# Licensed under the MIT License.

"""Tests for blending utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import Affine

from torchgeo.callbacks.blending import (
    _build_grid_index,
    _get_edge_deltas,
    _query_grid_index,
    _reconstruct_scene_from_patches,
    get_blend_mask,
    weighted_merge,
)


def _save_test_patch(
    path: Path, logits: torch.Tensor, transform: list[float], crs: Any = 'EPSG:32631'
) -> None:
    """Save test patch as GeoTIFF with one-hot encoded predictions.

    Args:
        path: Output file path.
        logits: Logits tensor of shape (num_classes, H, W).
        transform: Affine transform as list [a, b, c, d, e, f].
        crs: Coordinate reference system.
    """
    num_classes = logits.shape[0]
    class_predictions = logits.argmax(dim=0)
    one_hot = (
        torch.nn.functional.one_hot(class_predictions.long(), num_classes=num_classes)
        .permute(2, 0, 1)
        .to(torch.uint8)
        .numpy()
    )
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=one_hot.shape[1],
        width=one_hot.shape[2],
        count=one_hot.shape[0],
        dtype='uint8',
        transform=Affine(*transform),
        crs=crs,
    ) as dst:
        dst.write(one_hot)


class TestReconstructSceneFromPatches:
    """Tests for _reconstruct_scene_from_patches."""

    def test_single_patch(self) -> None:
        """Test reconstruction with single patch."""
        meta = [
            {
                'patch_id': 0,
                'geo_bbox': (0.0, 360.0, 640.0, 1000.0),
                'transform': [10.0, 0, 0, 0, -10.0, 1000],
            }
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

        assert shape == (64, 64)
        assert transform == Affine(10.0, 0, 0, 0, -10.0, 1000)
        assert meta[0]['bbox'] == (0, 0, 64, 64)

    def test_two_patches_horizontal(self) -> None:
        """Test reconstruction with two horizontal patches."""
        meta = [
            {
                'patch_id': 0,
                'geo_bbox': (100.0, 136.0, 164.0, 200.0),
                'transform': [1.0, 0, 100.0, 0, -1.0, 200.0],
            },
            {
                'patch_id': 1,
                'geo_bbox': (132.0, 136.0, 196.0, 200.0),
                'transform': [1.0, 0, 132.0, 0, -1.0, 200.0],
            },
        ]

        shape, transform = _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

        assert shape == (64, 96)
        assert transform == Affine(1.0, 0, 100.0, 0, -1.0, 200.0)
        assert meta[0]['bbox'] == (0, 0, 64, 64)
        assert meta[1]['bbox'] == (32, 0, 96, 64)

    def test_inconsistent_resolutions_raises(self) -> None:
        """Test error on inconsistent resolutions."""
        meta = [
            {
                'patch_id': 0,
                'geo_bbox': (0.0, 36.0, 64.0, 100.0),
                'transform': [1.0, 0, 0, 0, -1.0, 100],
            },
            {
                'patch_id': 1,
                'geo_bbox': (64.0, 36.0, 192.0, 100.0),
                'transform': [2.0, 0, 64, 0, -1.0, 100],
            },
        ]

        with pytest.raises(ValueError, match='Inconsistent resolutions'):
            _reconstruct_scene_from_patches(meta, (64, 64), delta=0)

    def test_empty_metadata_raises(self) -> None:
        """Test error on empty patch_metadata."""
        with pytest.raises(ValueError, match='patch_metadata is empty'):
            _reconstruct_scene_from_patches([], (64, 64), delta=0)


class TestGetEdgeDeltas:
    """Tests for _get_edge_deltas."""

    def test_corner_patch_two_edges_zero(self) -> None:
        """Corner patches get delta=0 on two boundary-touching edges."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = 1.0
        delta = 8

        top_left = (0.0, 36.0, 64.0, 100.0)
        result = _get_edge_deltas(top_left, scene_bounds, pixel_size, delta)
        assert result == (0, delta, 0, delta)

        bottom_right = (36.0, 0.0, 100.0, 64.0)
        result = _get_edge_deltas(bottom_right, scene_bounds, pixel_size, delta)
        assert result == (delta, 0, delta, 0)

    def test_interior_patch_all_edges_delta(self) -> None:
        """Interior patches get delta on all edges."""
        scene_bounds = (0.0, 0.0, 200.0, 200.0)
        interior_patch = (68.0, 68.0, 132.0, 132.0)
        pixel_size = 1.0
        delta = 8

        result = _get_edge_deltas(interior_patch, scene_bounds, pixel_size, delta)
        assert result == (delta, delta, delta, delta)

    def test_boundary_edge_single_edge_zero(self) -> None:
        """Patches on one boundary get delta=0 on that edge only."""
        scene_bounds = (0.0, 0.0, 200.0, 200.0)
        pixel_size = 1.0
        delta = 8

        top_edge = (68.0, 136.0, 132.0, 200.0)
        result = _get_edge_deltas(top_edge, scene_bounds, pixel_size, delta)
        assert result == (0, delta, delta, delta)

        left_edge = (0.0, 68.0, 64.0, 132.0)
        result = _get_edge_deltas(left_edge, scene_bounds, pixel_size, delta)
        assert result == (delta, delta, 0, delta)

    def test_tolerance_within_threshold(self) -> None:
        """Patches within 1.5 pixels of boundary count as boundary-touching."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = 1.0
        delta = 8

        almost_top = (32.0, 36.0, 68.0, 99.5)
        result = _get_edge_deltas(almost_top, scene_bounds, pixel_size, delta)
        assert result[0] == 0

    def test_tolerance_outside_threshold(self) -> None:
        """Patches beyond 1.5 pixels from boundary don't count as boundary."""
        scene_bounds = (0.0, 0.0, 100.0, 100.0)
        pixel_size = 1.0
        delta = 8

        not_at_top = (32.0, 36.0, 68.0, 98.0)
        result = _get_edge_deltas(not_at_top, scene_bounds, pixel_size, delta)
        assert result[0] == delta

    def test_single_patch_all_edges_zero(self) -> None:
        """Single patch covering entire scene has delta=0 on all edges."""
        scene_bounds = (0.0, 0.0, 64.0, 64.0)
        patch = (0.0, 0.0, 64.0, 64.0)
        pixel_size = 1.0
        delta = 8

        result = _get_edge_deltas(patch, scene_bounds, pixel_size, delta)
        assert result == (0, 0, 0, 0)


class TestGetBlendMask:
    """Tests for get_blend_mask."""

    def test_no_overlap(self) -> None:
        """Test blend mask with no overlap."""
        mask = get_blend_mask(64, overlap=0, delta=0, method='cosine')

        assert mask.shape == (64, 64)
        np.testing.assert_allclose(mask, np.ones((64, 64)), rtol=1e-5)

    def test_with_overlap_cosine(self) -> None:
        """Test cosine blend mask."""
        mask = get_blend_mask(64, overlap=8, delta=0, method='cosine')

        assert mask.shape == (64, 64)
        assert mask[0, 32] < mask[32, 32]
        assert mask[32, 32] == pytest.approx(1.0)

    def test_with_overlap_linear(self) -> None:
        """Test linear blend mask."""
        mask = get_blend_mask(64, overlap=8, delta=0, method='linear')

        assert mask.shape == (64, 64)
        assert mask[0, 32] < mask[32, 32]
        assert mask[32, 32] == pytest.approx(1.0)

    def test_invalid_method_raises(self) -> None:
        """Test invalid blend method raises error."""
        with pytest.raises(ValueError, match='Unknown blend method'):
            get_blend_mask(64, overlap=8, delta=0, method='invalid')

    def test_edge_deltas_suppresses_ramp(self) -> None:
        """Edges with delta=0 in edge_deltas get no blend ramp (weight=1)."""
        edge_deltas = (0, 8, 0, 8)
        mask = get_blend_mask(
            64, overlap=8, delta=8, method='cosine', edge_deltas=edge_deltas
        )

        assert mask.shape == (56, 56)
        assert mask[0, 28] == pytest.approx(1.0, rel=1e-4)
        assert mask[-1, 28] < 1.0

    def test_asymmetric_edge_deltas(self) -> None:
        """Different deltas on each edge produce asymmetric mask."""
        edge_deltas = (4, 8, 2, 6)
        mask = get_blend_mask(
            64, overlap=8, delta=8, method='cosine', edge_deltas=edge_deltas
        )

        expected_h = 64 - 4 - 8
        expected_w = 64 - 2 - 6
        assert mask.shape == (expected_h, expected_w)

    def test_edge_deltas_none_uses_uniform_delta(self) -> None:
        """When edge_deltas is None, uniform delta is used."""
        mask_uniform = get_blend_mask(64, overlap=8, delta=8, method='cosine')
        mask_explicit = get_blend_mask(
            64, overlap=8, delta=8, method='cosine', edge_deltas=(8, 8, 8, 8)
        )

        np.testing.assert_allclose(mask_uniform, mask_explicit)


class TestGridIndexing:
    """Tests for grid-based spatial indexing utilities."""

    def test_build_and_query(self) -> None:
        """Test building and querying grid index."""
        meta = [
            {'patch_id': 0, 'bbox': (0, 0, 64, 64)},
            {'patch_id': 1, 'bbox': (200, 200, 264, 264)},
        ]
        grid_size = 128
        grid = _build_grid_index(meta, grid_size)

        results = _query_grid_index(
            grid,
            meta,
            chunk_y=0,
            chunk_x=0,
            chunk_h=64,
            chunk_w=64,
            grid_size=grid_size,
        )

        assert len(results) == 1
        assert results[0]['patch_id'] == 0

    def test_query_multiple_patches(self) -> None:
        """Test querying returns multiple overlapping patches."""
        meta = [
            {'patch_id': 0, 'bbox': (0, 0, 64, 64)},
            {'patch_id': 1, 'bbox': (32, 0, 96, 64)},
            {'patch_id': 2, 'bbox': (200, 200, 264, 264)},
        ]
        grid_size = 128
        grid = _build_grid_index(meta, grid_size)

        results = _query_grid_index(
            grid,
            meta,
            chunk_y=0,
            chunk_x=0,
            chunk_h=100,
            chunk_w=100,
            grid_size=grid_size,
        )

        patch_ids = {r['patch_id'] for r in results}
        assert 0 in patch_ids
        assert 1 in patch_ids
        assert 2 not in patch_ids

    def test_query_returns_non_overlapping_in_same_cell(self) -> None:
        """Test grid query returns patches in same cell even if no pixel overlap.

        This tests the scenario where patches in the same grid cell are returned
        but don't actually overlap with the chunk at pixel level.
        """
        meta = [
            {'patch_id': 0, 'bbox': (0, 0, 32, 32)},
            {'patch_id': 1, 'bbox': (96, 96, 128, 128)},
        ]
        grid_size = 128
        grid = _build_grid_index(meta, grid_size)

        results = _query_grid_index(
            grid,
            meta,
            chunk_y=0,
            chunk_x=0,
            chunk_h=64,
            chunk_w=64,
            grid_size=grid_size,
        )

        patch_ids = {r['patch_id'] for r in results}
        assert 0 in patch_ids
        assert 1 in patch_ids


class TestExtentMismatch:
    """Tests for output extent accuracy (Issue 1)."""

    def test_extent_matches_input_bounds_3x3_grid(self) -> None:
        """Verify output extent matches original input extent exactly.

        Creates a 3x3 grid of overlapping patches covering exactly 160x160 pixels
        at 1m resolution. The output shape and transform should match exactly.
        """
        patch_size = 64
        stride = 48

        origin_x, origin_y = 1000.0, 2000.0
        res = 1.0

        meta = []
        for row in range(3):
            for col in range(3):
                geo_xmin = origin_x + col * stride * res
                geo_ymax = origin_y - row * stride * res
                geo_xmax = geo_xmin + patch_size * res
                geo_ymin = geo_ymax - patch_size * res

                meta.append(
                    {
                        'patch_id': row * 3 + col,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': [res, 0, geo_xmin, 0, -res, geo_ymax],
                    }
                )

        shape, transform = _reconstruct_scene_from_patches(
            meta, (patch_size, patch_size)
        )

        expected_width = 2 * stride + patch_size
        expected_height = 2 * stride + patch_size
        assert shape == (expected_height, expected_width)
        assert transform.c == pytest.approx(origin_x)
        assert transform.f == pytest.approx(origin_y)


class TestBlackBorder:
    """Tests for edge blending artifacts (Issue 2)."""

    def test_no_black_border_at_edges(self, tmp_path: Path) -> None:
        """Verify edge pixels have valid values after blending.

        Creates a 2x2 grid of patches where all patches predict class 1.
        After blending, all pixels including edges/corners should be class 1.
        This verifies that edge patches are correctly processed without artifacts.
        """
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 3
        expected_class = 1
        stride = patch_size - 2 * overlap

        origin_x, origin_y = 0.0, 128.0
        res = 1.0

        patch_metadata = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = origin_x + col * stride * res
                geo_ymax = origin_y - row * stride * res
                geo_xmax = geo_xmin + patch_size * res
                geo_ymin = geo_ymax - patch_size * res

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'patch_{patch_id:06d}.tif'
                transform = [res, 0, geo_xmin, 0, -res, geo_ymax]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            crs='EPSG:32631',
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        assert data[0, 0] == expected_class, (
            f'Top-left corner: {data[0, 0]} != {expected_class}'
        )
        assert data[0, -1] == expected_class, (
            f'Top-right corner: {data[0, -1]} != {expected_class}'
        )
        assert data[-1, 0] == expected_class, (
            f'Bottom-left corner: {data[-1, 0]} != {expected_class}'
        )
        assert data[-1, -1] == expected_class, (
            f'Bottom-right corner: {data[-1, -1]} != {expected_class}'
        )

        assert np.all(data[0, :] == expected_class), 'Top edge has wrong values'
        assert np.all(data[-1, :] == expected_class), 'Bottom edge has wrong values'
        assert np.all(data[:, 0] == expected_class), 'Left edge has wrong values'
        assert np.all(data[:, -1] == expected_class), 'Right edge has wrong values'

        assert np.all(data == expected_class), (
            f'Not all pixels are class {expected_class}'
        )


class TestNonOverlappingPatches:
    """Tests for handling patches that don't overlap with chunks."""

    def test_weighted_merge_skips_non_overlapping_patches(self, tmp_path: Path) -> None:
        """Test weighted_merge skips patches that don't overlap with current chunk.

        Creates two patches far apart but in the same grid cell (grid_size=1024).
        When processing chunks, the grid query returns both patches, but the
        overlap calculation should skip patches that don't actually overlap.
        """
        patch_size = 64
        num_classes = 2
        res = 1.0
        origin_y = 564.0

        patch_metadata = []
        positions = [(0, 0), (500, 500)]
        for patch_id, (col_offset, row_offset) in enumerate(positions):
            geo_xmin = float(col_offset)
            geo_ymax = origin_y - row_offset
            geo_xmax = geo_xmin + patch_size * res
            geo_ymin = geo_ymax - patch_size * res

            logits = torch.zeros(num_classes, patch_size, patch_size)
            logits[1] = 1.0

            patch_file = tmp_path / f'patch_{patch_id:06d}.tif'
            transform = [res, 0, geo_xmin, 0, -res, geo_ymax]
            _save_test_patch(patch_file, logits, transform)

            patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_file,
                    'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                    'transform': transform,
                }
            )

        output_path = tmp_path / 'output_sparse.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=0,
            delta=0,
            blend_method='cosine',
            crs='EPSG:32631',
            output_path=output_path,
            chunk_size=128,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert data[0, 0] == 1
            assert data[500, 500] == 1


class TestDatasetBoundsMode:
    """Tests for weighted_merge with dataset_bounds parameter."""

    def test_weighted_merge_with_dataset_bounds(self, tmp_path: Path) -> None:
        """Test weighted_merge uses dataset_bounds for output extent."""
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 3
        expected_class = 1

        dataset_bounds = (0.0, 0.0, 128.0, 128.0)
        dataset_res = 1.0

        patch_metadata = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = col * 32.0
                geo_ymax = 128.0 - row * 32.0
                geo_xmax = geo_xmin + patch_size
                geo_ymin = geo_ymax - patch_size

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'patch_{patch_id:06d}.tif'
                transform = [1.0, 0, geo_xmin, 0, -1.0, geo_ymax]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'output_with_bounds.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            crs='EPSG:32631',
            output_path=output_path,
            chunk_size=256,
            dataset_bounds=dataset_bounds,
            dataset_res=dataset_res,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert src.width == 128
            assert src.height == 128
            assert data.shape == (128, 128)

    def test_dataset_bounds_edge_coverage(self, tmp_path: Path) -> None:
        """Using dataset_bounds still produces full edge coverage with delta > 0."""
        patch_size = 64
        overlap = 16
        delta = 8
        num_classes = 2
        expected_class = 1

        dataset_bounds = (0.0, 0.0, 96.0, 96.0)
        dataset_res = 1.0

        patch_metadata = []
        for row in range(2):
            for col in range(2):
                patch_id = row * 2 + col
                geo_xmin = col * 32.0
                geo_ymax = 96.0 - row * 32.0
                geo_xmax = geo_xmin + patch_size
                geo_ymin = geo_ymax - patch_size

                logits = torch.zeros(num_classes, patch_size, patch_size)
                logits[expected_class] = 1.0

                patch_file = tmp_path / f'patch_bounds_{patch_id:06d}.tif'
                transform = [1.0, 0, geo_xmin, 0, -1.0, geo_ymax]
                _save_test_patch(patch_file, logits, transform)

                patch_metadata.append(
                    {
                        'patch_id': patch_id,
                        'file': patch_file,
                        'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                        'transform': transform,
                    }
                )

        output_path = tmp_path / 'output_bounds_edge.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=overlap,
            delta=delta,
            blend_method='cosine',
            crs='EPSG:32631',
            output_path=output_path,
            chunk_size=256,
            dataset_bounds=dataset_bounds,
            dataset_res=dataset_res,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        assert np.all(data[0, :] == expected_class), 'Top edge has wrong values'
        assert np.all(data[-1, :] == expected_class), 'Bottom edge has wrong values'
        assert np.all(data[:, 0] == expected_class), 'Left edge has wrong values'
        assert np.all(data[:, -1] == expected_class), 'Right edge has wrong values'


class TestSinglePatchScene:
    """Tests for single-patch scenes."""

    def test_single_patch_no_black_border(self, tmp_path: Path) -> None:
        """Single patch covering entire scene has no black borders with delta > 0."""
        patch_size = 64
        delta = 8
        num_classes = 2
        expected_class = 1
        res = 1.0

        geo_xmin = 0.0
        geo_ymax = 64.0
        geo_xmax = 64.0
        geo_ymin = 0.0

        logits = torch.zeros(num_classes, patch_size, patch_size)
        logits[expected_class] = 1.0

        patch_file = tmp_path / 'single_patch.tif'
        transform = [res, 0, geo_xmin, 0, -res, geo_ymax]
        _save_test_patch(patch_file, logits, transform)

        patch_metadata = [
            {
                'patch_id': 0,
                'file': patch_file,
                'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                'transform': transform,
            }
        ]

        output_path = tmp_path / 'single_patch_output.tif'
        weighted_merge(
            patch_metadata=patch_metadata,
            num_classes=num_classes,
            overlap=0,
            delta=delta,
            blend_method='cosine',
            crs='EPSG:32631',
            output_path=output_path,
            chunk_size=256,
        )

        with rasterio.open(output_path) as src:
            data = src.read(1)

        assert data.shape == (64, 64)
        assert np.all(data == expected_class), 'Single patch has wrong values'
        assert data[0, 0] == expected_class, 'Top-left corner wrong'
        assert data[0, -1] == expected_class, 'Top-right corner wrong'
        assert data[-1, 0] == expected_class, 'Bottom-left corner wrong'
        assert data[-1, -1] == expected_class, 'Bottom-right corner wrong'
