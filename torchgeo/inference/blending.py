# Copyright (c) TorchGeo Contributors
# All rights reserved.
# Licensed under the MIT License.

"""Blending utilities for tiled inference."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from affine import Affine
from tqdm import tqdm


def _reconstruct_scene_from_patches(
    patch_metadata: list[dict[str, Any]], patch_size: tuple[int, int], delta: int = 0
) -> tuple[tuple[int, int], Affine]:
    """Reconstruct scene-level transform and shape from per-patch transforms.

    This leverages per-patch transforms to reconstruct the full scene metadata
    without needing upfront dataset information. Converts geo coordinates to
    pixel coordinates using a global reference frame.

    Args:
        patch_metadata: List of dicts with 'geo_bbox' and 'transform'.
            geo_bbox is (geo_xmin, geo_ymin, geo_xmax, geo_ymax) in geo coordinates.
            transform is Tensor [a, b, c, d, e, f] representing affine:
                | a  b  c |   where c, f are the origin
                | d  e  f |   and a, e are x_res, y_res
                | 0  0  1 |
        patch_size: Size of each patch as (height, width) in pixels.
        delta: Pixels to crop from edges (unused, kept for API compatibility).

    Returns:
        output_shape: (height, width) of full scene.
        scene_transform: Affine transform for the full scene.

    Raises:
        ValueError: If patches have inconsistent resolutions or metadata is empty.


    """
    if not patch_metadata:
        raise ValueError('patch_metadata is empty')

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

    all_geo_xmin = [meta['geo_bbox'][0] for meta in patch_metadata]
    all_geo_ymax = [meta['geo_bbox'][3] for meta in patch_metadata]

    global_geo_xmin = min(all_geo_xmin)
    global_geo_ymax = max(all_geo_ymax)

    patch_h, patch_w = patch_size

    for meta in patch_metadata:
        geo_bbox = meta['geo_bbox']
        patch_col_start = round((geo_bbox[0] - global_geo_xmin) / x_res)
        patch_row_start = round((global_geo_ymax - geo_bbox[3]) / abs(y_res))

        meta['bbox'] = (
            patch_col_start,
            patch_row_start,
            patch_col_start + patch_w,
            patch_row_start + patch_h,
        )

    all_x_starts = [meta['bbox'][0] for meta in patch_metadata]
    all_y_starts = [meta['bbox'][1] for meta in patch_metadata]
    all_x_stops = [meta['bbox'][2] for meta in patch_metadata]
    all_y_stops = [meta['bbox'][3] for meta in patch_metadata]

    min_x = min(all_x_starts)
    min_y = min(all_y_starts)
    max_x = max(all_x_stops)
    max_y = max(all_y_stops)

    scene_transform = Affine(x_res, 0, global_geo_xmin, 0, y_res, global_geo_ymax)

    output_width = max_x - min_x
    output_height = max_y - min_y

    return (output_height, output_width), scene_transform


def get_blend_mask(
    patch_size: int | tuple[int, int], overlap: int, delta: int, method: str = 'cosine'
) -> np.ndarray:
    """Generate blend mask for weighted patch merging.

    Uses the same formula as habitalp2 for proven compatibility.

    Args:
        patch_size: Size of patch (H, W) or single int.
        overlap: Overlap in pixels on each side.
        delta: Pixels to crop from edges.
        method: Blending method ('cosine' or 'linear').

    Returns:
        Blend mask of shape (H-2*delta, W-2*delta) with values in [0, 1].

    Raises:
        ValueError: If method is not 'cosine' or 'linear'.
    """
    if isinstance(patch_size, int):
        h = w = patch_size
    else:
        h, w = patch_size

    h_crop = h - 2 * delta
    w_crop = w - 2 * delta

    if method == 'cosine':
        y = np.ones(h_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)) / 2 + 0.5
            y[:overlap] = ramp[::-1]
            y[-overlap:] = ramp

        x = np.ones(w_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)) / 2 + 0.5
            x[:overlap] = ramp[::-1]
            x[-overlap:] = ramp

        mask = y[:, None] * x[None, :]
    elif method == 'linear':
        y = np.ones(h_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            y[:overlap] = ramp
            y[-overlap:] = ramp[::-1]

        x = np.ones(w_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            x[:overlap] = ramp
            x[-overlap:] = ramp[::-1]

        mask = y[:, None] * x[None, :]
    else:
        raise ValueError(f'Unknown blend method: {method}')

    mask += 1e-6

    return mask


def _build_grid_index(
    patch_metadata: list[dict[str, Any]], grid_size: int
) -> dict[tuple[int, int], list[int]]:
    """Build simple grid-based spatial index for fast patch lookup.

    Args:
        patch_metadata: List of dicts with 'bbox' (x_start, y_start, x_stop, y_stop).
        grid_size: Size of grid cells in pixels.

    Returns:
        Dict mapping (grid_row, grid_col) to list of patch indices.


    """
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, meta in enumerate(patch_metadata):
        bbox = meta['bbox']
        x_start, y_start, x_stop, y_stop = bbox

        grid_col_start = x_start // grid_size
        grid_col_end = (x_stop - 1) // grid_size
        grid_row_start = y_start // grid_size
        grid_row_end = (y_stop - 1) // grid_size

        for gr in range(grid_row_start, grid_row_end + 1):
            for gc in range(grid_col_start, grid_col_end + 1):
                grid[(gr, gc)].append(idx)

    return dict(grid)


def _query_grid_index(
    grid: dict[tuple[int, int], list[int]],
    patch_metadata: list[dict[str, Any]],
    chunk_y: int,
    chunk_x: int,
    chunk_h: int,
    chunk_w: int,
    grid_size: int,
) -> list[dict[str, Any]]:
    """Query grid index for patches overlapping chunk.

    Args:
        grid: Grid index from _build_grid_index.
        patch_metadata: Original patch metadata list.
        chunk_y: Top-left y coordinate of chunk.
        chunk_x: Top-left x coordinate of chunk.
        chunk_h: Height of chunk.
        chunk_w: Width of chunk.
        grid_size: Size of grid cells in pixels.

    Returns:
        List of patch metadata dicts that may overlap the chunk.

    """
    grid_col_start = chunk_x // grid_size
    grid_col_end = (chunk_x + chunk_w - 1) // grid_size
    grid_row_start = chunk_y // grid_size
    grid_row_end = (chunk_y + chunk_h - 1) // grid_size

    candidate_indices: set[int] = set()
    for gr in range(grid_row_start, grid_row_end + 1):
        for gc in range(grid_col_start, grid_col_end + 1):
            candidate_indices.update(grid.get((gr, gc), []))

    return [patch_metadata[idx] for idx in candidate_indices]


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


    """
    from torchgeo.inference.writer import GeoTIFFWriter

    first_patch = torch.load(patch_metadata[0]['file'])
    patch_h, patch_w = first_patch['logits'].shape[-2:]

    output_shape, scene_transform = _reconstruct_scene_from_patches(
        patch_metadata, (patch_h, patch_w), delta
    )

    x_res = scene_transform.a
    y_res = scene_transform.e
    global_geo_xmin = scene_transform.c
    global_geo_ymax = scene_transform.f

    grid_size = chunk_size * 2
    grid = _build_grid_index(patch_metadata, grid_size)

    effective_overlap = max(0, overlap - 2 * delta)
    blend_mask = get_blend_mask(
        (patch_h, patch_w), effective_overlap, delta, blend_method
    )

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
        chunk_iter = [
            (cy, cx)
            for cy in range(0, height, chunk_size)
            for cx in range(0, width, chunk_size)
        ]
        for chunk_y, chunk_x in tqdm(chunk_iter, desc='Merging chunks'):
            chunk_h = min(chunk_size, height - chunk_y)
            chunk_w = min(chunk_size, width - chunk_x)

            chunk_output = np.zeros((num_classes, chunk_h, chunk_w), dtype=np.float32)
            chunk_weights = np.zeros((chunk_h, chunk_w), dtype=np.float32)

            overlapping = _query_grid_index(
                grid, patch_metadata, chunk_y, chunk_x, chunk_h, chunk_w, grid_size
            )
            for meta in overlapping:
                patch_data = torch.load(meta['file'])
                logits = patch_data['logits'].numpy()

                geo_bbox = meta['geo_bbox']
                patch_col_start = int((geo_bbox[0] - global_geo_xmin) / x_res)
                patch_row_start = int((global_geo_ymax - geo_bbox[3]) / abs(y_res))

                if delta > 0:
                    logits = logits[:, delta:-delta, delta:-delta]
                    patch_col_start += delta
                    patch_row_start += delta

                current_patch_h, current_patch_w = logits.shape[1], logits.shape[2]

                overlap_col_start = max(0, patch_col_start - chunk_x)
                overlap_row_start = max(0, patch_row_start - chunk_y)
                overlap_col_end = min(
                    chunk_w, patch_col_start + current_patch_w - chunk_x
                )
                overlap_row_end = min(
                    chunk_h, patch_row_start + current_patch_h - chunk_y
                )

                if (
                    overlap_col_end <= overlap_col_start
                    or overlap_row_end <= overlap_row_start
                ):
                    continue

                patch_col_start_local = max(0, chunk_x - patch_col_start)
                patch_row_start_local = max(0, chunk_y - patch_row_start)
                patch_col_end_local = patch_col_start_local + (
                    overlap_col_end - overlap_col_start
                )
                patch_row_end_local = patch_row_start_local + (
                    overlap_row_end - overlap_row_start
                )

                patch_region = logits[
                    :,
                    patch_row_start_local:patch_row_end_local,
                    patch_col_start_local:patch_col_end_local,
                ]
                mask_region = blend_mask[
                    patch_row_start_local:patch_row_end_local,
                    patch_col_start_local:patch_col_end_local,
                ]

                chunk_output[
                    :,
                    overlap_row_start:overlap_row_end,
                    overlap_col_start:overlap_col_end,
                ] += patch_region * mask_region[None, :, :]
                chunk_weights[
                    overlap_row_start:overlap_row_end, overlap_col_start:overlap_col_end
                ] += mask_region

            chunk_output = chunk_output / (chunk_weights[None, :, :] + 1e-8)
            chunk_labels = np.argmax(chunk_output, axis=0).astype(np.uint8)

            writer.write_chunk(chunk_labels, chunk_y, chunk_x)

    writer.finalize()
