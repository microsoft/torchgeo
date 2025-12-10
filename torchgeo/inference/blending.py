# Copyright (c) TorchGeo Contributors
# All rights reserved.
# Licensed under the MIT License.

"""Blending utilities for tiled inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from affine import Affine
from shapely.geometry import box


def _reconstruct_scene_from_patches(
    patch_metadata: list[dict[str, Any]], delta: int = 0
) -> tuple[tuple[int, int], Affine]:
    """Reconstruct scene-level transform and shape from per-patch transforms.

    This leverages per-patch transforms to reconstruct the full scene metadata
    without needing upfront dataset information.

    Args:
        patch_metadata: List of dicts with 'bbox' and 'transform'.
            bbox is (x_start, y_start, x_stop, y_stop) in pixel coordinates.
            transform is Tensor [a, b, c, d, e, f] representing affine:
                | a  b  c |   where c, f are the origin
                | d  e  f |   and a, e are x_res, y_res
                | 0  0  1 |
        delta: Pixels to crop from edges (affects output shape).

    Returns:
        output_shape: (height, width) of full scene.
        scene_transform: Affine transform for the full scene.

    Raises:
        ValueError: If patches have inconsistent resolutions or metadata is empty.

    .. versionadded:: 0.7
    """
    if not patch_metadata:
        raise ValueError('patch_metadata is empty')

    all_x_starts = [meta['bbox'][0] for meta in patch_metadata]
    all_y_starts = [meta['bbox'][1] for meta in patch_metadata]
    all_x_stops = [meta['bbox'][2] for meta in patch_metadata]
    all_y_stops = [meta['bbox'][3] for meta in patch_metadata]

    min_x = min(all_x_starts)
    min_y = min(all_y_starts)
    max_x = max(all_x_stops)
    max_y = max(all_y_stops)

    first_transform = patch_metadata[0]['transform']
    x_res = first_transform[0].item()
    y_res = first_transform[4].item()

    for meta in patch_metadata[1:]:
        t = meta['transform']
        if abs(t[0].item() - x_res) > 1e-6 or abs(t[4].item() - y_res) > 1e-6:
            raise ValueError(
                f'Inconsistent resolutions: first patch has ({x_res}, {y_res}), '
                f'but patch {meta["patch_id"]} has ({t[0].item()}, {t[4].item()})'
            )

    first_geo_x = first_transform[2].item()
    first_geo_y = first_transform[5].item()

    scene_geo_x = first_geo_x + (min_x * x_res)
    scene_geo_y = first_geo_y + (min_y * y_res)

    scene_transform = Affine(x_res, 0, scene_geo_x, 0, y_res, scene_geo_y)

    output_width = max_x - min_x - 2 * delta
    output_height = max_y - min_y - 2 * delta

    return (output_height, output_width), scene_transform


def get_blend_mask(
    patch_size: int | tuple[int, int], overlap: int, delta: int, method: str = 'cosine'
) -> np.ndarray:
    """Generate blend mask for weighted patch merging.

    Args:
        patch_size: Size of patch (H, W) or single int.
        overlap: Overlap in pixels on each side.
        delta: Pixels to crop from edges.
        method: Blending method ('cosine' or 'linear').

    Returns:
        Blend mask of shape (H-2*delta, W-2*delta) with values in [0, 1].

    Raises:
        ValueError: If method is not 'cosine' or 'linear'.

    .. versionadded:: 0.7
    """
    if isinstance(patch_size, int):
        h = w = patch_size
    else:
        h, w = patch_size

    h_crop = h - 2 * delta
    w_crop = w - 2 * delta

    def weight_1d(size: int, overlap: int, method: str) -> np.ndarray:
        """Create 1D weight function."""
        weights = np.ones(size, dtype=np.float32)

        if method == 'cosine':
            if overlap > 0:
                ramp = (1 - np.cos(np.linspace(0, np.pi, overlap))) / 2
                weights[:overlap] = ramp
                weights[-overlap:] = ramp[::-1]
        elif method == 'linear':
            if overlap > 0:
                ramp = np.linspace(0, 1, overlap, dtype=np.float32)
                weights[:overlap] = ramp
                weights[-overlap:] = ramp[::-1]
        else:
            raise ValueError(f'Unknown blend method: {method}')

        return weights

    weights_y = weight_1d(h_crop, overlap, method)
    weights_x = weight_1d(w_crop, overlap, method)
    mask = np.outer(weights_y, weights_x)

    return mask


def _build_spatial_index(patch_metadata: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Build GeoPandas-based spatial index using Shapely STRtree.

    Args:
        patch_metadata: List of dicts with 'bbox' and other metadata.
            bbox is (x_start, y_start, x_stop, y_stop).

    Returns:
        GeoDataFrame with spatial index ready for querying.

    .. versionadded:: 0.7
    """
    geometries = [
        box(meta['bbox'][0], meta['bbox'][1], meta['bbox'][2], meta['bbox'][3])
        for meta in patch_metadata
    ]

    gdf = gpd.GeoDataFrame(patch_metadata, geometry=geometries)
    return gdf


def _query_spatial_index(
    gdf: gpd.GeoDataFrame, chunk_y: int, chunk_x: int, chunk_h: int, chunk_w: int
) -> list[dict[str, Any]]:
    """Query spatial index for patches overlapping chunk.

    Args:
        gdf: GeoDataFrame with patch metadata.
        chunk_y: Top-left y coordinate of chunk.
        chunk_x: Top-left x coordinate of chunk.
        chunk_h: Height of chunk.
        chunk_w: Width of chunk.

    Returns:
        List of patch metadata dicts overlapping the chunk.

    .. versionadded:: 0.7
    """
    query_box = box(chunk_x, chunk_y, chunk_x + chunk_w, chunk_y + chunk_h)

    possible_matches_idx = list(gdf.sindex.intersection(query_box.bounds))
    possible_matches = gdf.iloc[possible_matches_idx]

    precise_matches = possible_matches[possible_matches.intersects(query_box)]

    return list(precise_matches.to_dict('records'))


def weighted_merge(
    patch_metadata: list[dict[str, Any]],
    num_classes: int,
    overlap: int,
    delta: int,
    blend_method: str = 'cosine',
    crs: Any = None,
    output_path: str | Path | None = None,
    chunk_size: int = 4096,
    cog_config: dict[str, Any] | None = None,
) -> None:
    """Merge patches from disk with weighted blending.

    Uses chunked processing with spatial indexing for memory-efficient
    merging of arbitrarily large scenes.

    Args:
        patch_metadata: List of dicts with 'file', 'bbox', 'transform'.
        num_classes: Number of classes.
        overlap: Overlap in pixels.
        delta: Pixels to crop from edges.
        blend_method: 'cosine' or 'linear'.
        crs: Coordinate reference system.
        output_path: Where to save GeoTIFF.
        chunk_size: Size of chunks for processing.
        cog_config: COG configuration.

    .. versionadded:: 0.7
    """
    from torchgeo.inference.writer import GeoTIFFWriter

    output_shape, scene_transform = _reconstruct_scene_from_patches(
        patch_metadata, delta
    )

    gdf = _build_spatial_index(patch_metadata)

    first_patch = torch.load(patch_metadata[0]['file'])
    patch_h, patch_w = first_patch['logits'].shape[-2:]
    blend_mask = get_blend_mask((patch_h, patch_w), overlap, delta, blend_method)

    assert output_path is not None
    writer = GeoTIFFWriter(
        output_path=output_path,
        width=output_shape[1],
        height=output_shape[0],
        num_bands=1,
        crs=crs,
        transform=scene_transform,
        cog_config=cog_config,
    )

    height, width = output_shape
    with writer:
        for chunk_y in range(0, height, chunk_size):
            for chunk_x in range(0, width, chunk_size):
                chunk_h = min(chunk_size, height - chunk_y)
                chunk_w = min(chunk_size, width - chunk_x)

                chunk_output = np.zeros(
                    (num_classes, chunk_h, chunk_w), dtype=np.float32
                )
                chunk_weights = np.zeros((1, chunk_h, chunk_w), dtype=np.float32)

                overlapping = _query_spatial_index(
                    gdf, chunk_y, chunk_x, chunk_h, chunk_w
                )

                for meta in overlapping:
                    patch_data = torch.load(meta['file'])
                    logits = patch_data['logits'].numpy()
                    bounds_tensor = patch_data['bounds']

                    x_start = int(bounds_tensor[0].item())
                    y_start = int(bounds_tensor[3].item())

                    if delta > 0:
                        logits = logits[:, delta:-delta, delta:-delta]
                        x_start += delta
                        y_start += delta

                    overlap_x_start = max(0, x_start - chunk_x)
                    overlap_y_start = max(0, y_start - chunk_y)
                    overlap_x_end = min(chunk_w, x_start + logits.shape[2] - chunk_x)
                    overlap_y_end = min(chunk_h, y_start + logits.shape[1] - chunk_y)

                    if (
                        overlap_x_end <= overlap_x_start
                        or overlap_y_end <= overlap_y_start
                    ):
                        continue

                    patch_x_start = overlap_x_start - (x_start - chunk_x)
                    patch_y_start = overlap_y_start - (y_start - chunk_y)
                    patch_x_end = patch_x_start + (overlap_x_end - overlap_x_start)
                    patch_y_end = patch_y_start + (overlap_y_end - overlap_y_start)

                    patch_region = logits[
                        :, patch_y_start:patch_y_end, patch_x_start:patch_x_end
                    ]
                    mask_region = blend_mask[
                        patch_y_start:patch_y_end, patch_x_start:patch_x_end
                    ]

                    weighted = patch_region * mask_region
                    chunk_output[
                        :, overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end
                    ] += weighted
                    chunk_weights[
                        :, overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end
                    ] += mask_region

                chunk_output = chunk_output / (chunk_weights + 1e-8)

                chunk_labels = np.argmax(chunk_output, axis=0).astype(np.uint8)

                writer.write_chunk(chunk_labels, chunk_y, chunk_x)

    writer.finalize()
