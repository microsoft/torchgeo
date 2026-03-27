# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Blending utilities for tiled inference."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm


class PatchMetadata(TypedDict):
    """Metadata for a single prediction patch."""

    patch_id: int
    file: Path
    geo_bbox: tuple[float, float, float, float]
    transform: list[float]
    bbox: NotRequired[tuple[int, int, int, int]]
    edge_deltas: NotRequired[tuple[int, int, int, int]]


def _get_edge_deltas(
    geo_bbox: tuple[float, float, float, float],
    scene_bounds: tuple[float, float, float, float],
    pixel_size: float,
    delta: int,
) -> tuple[int, int, int, int]:
    """Compute per-edge crop amounts based on boundary proximity.

    Patches touching scene boundaries preserve their edge pixels (delta=0 on that
    edge) to avoid black borders. Interior edges are cropped normally to remove
    neural network edge artifacts.

    A tolerance of 1.5 * pixel_size is used for boundary detection to handle
    floating-point imprecision. This means patches within 1.5 pixels of a scene
    boundary are treated as boundary-touching.

    Args:
        geo_bbox: Patch bounds as (xmin, ymin, xmax, ymax) in geo coordinates.
        scene_bounds: Scene bounds as (minx, miny, maxx, maxy) in geo coordinates.
        pixel_size: Size of one pixel in geo units.
        delta: Default pixels to crop from edges.

    Returns:
        Tuple of (top, bottom, left, right) crop amounts in pixels.
    """
    minx, miny, maxx, maxy = scene_bounds
    tolerance = abs(pixel_size) * 1.5

    top = 0 if abs(geo_bbox[3] - maxy) < tolerance else delta
    bottom = 0 if abs(geo_bbox[1] - miny) < tolerance else delta
    left = 0 if abs(geo_bbox[0] - minx) < tolerance else delta
    right = 0 if abs(geo_bbox[2] - maxx) < tolerance else delta

    return top, bottom, left, right


def _reconstruct_scene_from_patches(
    patch_metadata: list[PatchMetadata], patch_size: tuple[int, int], delta: int = 0
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
        delta: Pixels to crop from patch edges before blending.

    Returns:
        output_shape: (height, width) of full scene.
        scene_transform: Affine transform for the full scene.

    Raises:
        ValueError: If patches have inconsistent resolutions or metadata is empty.


    """
    if not patch_metadata:
        raise ValueError('patch_metadata is empty')

    first_transform = patch_metadata[0]['transform']
    x_res = first_transform[0]
    y_res = first_transform[4]

    for meta in patch_metadata[1:]:
        t = meta['transform']
        if abs(t[0] - x_res) > 1e-6 or abs(t[4] - y_res) > 1e-6:
            raise ValueError(
                f'Inconsistent resolutions: first patch has ({x_res}, {y_res}), '
                f'but patch {meta["patch_id"]} has ({t[0]}, {t[4]})'
            )

    all_geo_xmin = [meta['geo_bbox'][0] for meta in patch_metadata]
    all_geo_xmax = [meta['geo_bbox'][2] for meta in patch_metadata]
    all_geo_ymin = [meta['geo_bbox'][1] for meta in patch_metadata]
    all_geo_ymax = [meta['geo_bbox'][3] for meta in patch_metadata]

    scene_bounds = (
        min(all_geo_xmin),
        min(all_geo_ymin),
        max(all_geo_xmax),
        max(all_geo_ymax),
    )
    global_geo_xmin, _, _, global_geo_ymax = scene_bounds

    patch_h, patch_w = patch_size

    for meta in patch_metadata:
        geo_bbox = meta['geo_bbox']
        top, bottom, left, right = _get_edge_deltas(
            geo_bbox, scene_bounds, x_res, delta
        )
        meta['edge_deltas'] = (top, bottom, left, right)

        patch_geo_xmin = geo_bbox[0] + left * x_res
        patch_geo_ymax = geo_bbox[3] - top * abs(y_res)
        effective_patch_w = patch_w - left - right
        effective_patch_h = patch_h - top - bottom

        patch_col_start = round((patch_geo_xmin - global_geo_xmin) / x_res)
        patch_row_start = round((global_geo_ymax - patch_geo_ymax) / abs(y_res))

        meta['bbox'] = (
            patch_col_start,
            patch_row_start,
            patch_col_start + effective_patch_w,
            patch_row_start + effective_patch_h,
        )

    all_x_starts = [meta['bbox'][0] for meta in patch_metadata]
    all_y_starts = [meta['bbox'][1] for meta in patch_metadata]
    all_x_stops = [meta['bbox'][2] for meta in patch_metadata]
    all_y_stops = [meta['bbox'][3] for meta in patch_metadata]

    min_x = min(all_x_starts)
    min_y = min(all_y_starts)
    max_x = max(all_x_stops)
    max_y = max(all_y_stops)

    for meta in patch_metadata:
        bbox = meta['bbox']
        meta['bbox'] = (
            bbox[0] - min_x,
            bbox[1] - min_y,
            bbox[2] - min_x,
            bbox[3] - min_y,
        )

    scene_geo_xmin = global_geo_xmin + min_x * x_res
    scene_geo_ymax = global_geo_ymax + min_y * y_res

    scene_transform = Affine(x_res, 0, scene_geo_xmin, 0, y_res, scene_geo_ymax)

    output_width = max_x - min_x
    output_height = max_y - min_y

    return (output_height, output_width), scene_transform


def get_blend_mask(
    patch_size: int | tuple[int, int],
    overlap: int,
    delta: int,
    method: str = 'cosine',
    edge_deltas: tuple[int, int, int, int] | None = None,
) -> np.typing.NDArray[np.floating[Any]]:
    """Generate blend mask for weighted patch merging.

    Uses the same formula as habitalp2 for proven compatibility.

    Args:
        patch_size: Size of patch (H, W) or single int.
        overlap: Overlap in pixels on each side.
        delta: Default pixels to crop from edges.
        method: Blending method ('cosine' or 'linear').
        edge_deltas: Optional per-edge crop amounts (top, bottom, left, right).
            If provided, only applies blend ramps on edges where delta > 0.

    Returns:
        Blend mask with values in [0, 1].

    Raises:
        ValueError: If method is not 'cosine' or 'linear'.
    """
    if isinstance(patch_size, int):
        h = w = patch_size
    else:
        h, w = patch_size

    if edge_deltas is not None:
        top, bottom, left, right = edge_deltas
        apply_top_ramp = top > 0
        apply_bottom_ramp = bottom > 0
        apply_left_ramp = left > 0
        apply_right_ramp = right > 0
    else:
        top = bottom = left = right = delta
        apply_top_ramp = apply_bottom_ramp = apply_left_ramp = apply_right_ramp = True

    h_crop = h - top - bottom
    w_crop = w - left - right

    if h_crop <= 0 or w_crop <= 0:
        raise ValueError('delta crops away the entire patch')
    if overlap > min(h_crop, w_crop):
        raise ValueError('overlap exceeds cropped patch dimensions')

    if method == 'cosine':
        y = np.ones(h_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)) / 2 + 0.5
            if apply_top_ramp:
                y[:overlap] = ramp[::-1]
            if apply_bottom_ramp:
                y[-overlap:] = ramp

        x = np.ones(w_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.cos(np.pi * (np.arange(overlap) + 1) / (overlap + 1)) / 2 + 0.5
            if apply_left_ramp:
                x[:overlap] = ramp[::-1]
            if apply_right_ramp:
                x[-overlap:] = ramp

        mask = y[:, None] * x[None, :]
    elif method == 'linear':
        y = np.ones(h_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            if apply_top_ramp:
                y[:overlap] = ramp
            if apply_bottom_ramp:
                y[-overlap:] = ramp[::-1]

        x = np.ones(w_crop, dtype=np.float32)
        if overlap > 0:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            if apply_left_ramp:
                x[:overlap] = ramp
            if apply_right_ramp:
                x[-overlap:] = ramp[::-1]

        mask = y[:, None] * x[None, :]
    else:
        raise ValueError(f'Unknown blend method: {method}')

    return mask


def _build_grid_index(
    patch_metadata: list[PatchMetadata], grid_size: int
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
    patch_metadata: list[PatchMetadata],
    chunk_y: int,
    chunk_x: int,
    chunk_h: int,
    chunk_w: int,
    grid_size: int,
) -> list[PatchMetadata]:
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
    patch_metadata: list[PatchMetadata],
    num_classes: int,
    overlap: int,
    delta: int,
    blend_method: str = 'cosine',
    crs: str | None = None,
    output_path: str | Path | None = None,
    chunk_size: int = 4096,
    cog_config: dict[str, Any] | None = None,
    dataset_bounds: tuple[float, float, float, float] | None = None,
    dataset_res: float | None = None,
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
        dataset_bounds: Original dataset bounds (minx, miny, maxx, maxy).
        dataset_res: Original dataset resolution.


    """
    from torchgeo.callbacks.writer import GeoTIFFWriter

    with rasterio.open(patch_metadata[0]['file']) as src:
        patch_h, patch_w = src.height, src.width

    if dataset_bounds is not None and dataset_res is not None:
        minx, miny, maxx, maxy = dataset_bounds
        res = dataset_res[0] if not isinstance(dataset_res, float) else dataset_res
        output_width = round((maxx - minx) / res)
        output_height = round((maxy - miny) / res)
        output_shape = (output_height, output_width)
        scene_transform = Affine(res, 0, minx, 0, -res, maxy)
        scene_bounds = dataset_bounds

        first_transform = patch_metadata[0]['transform']
        x_res = first_transform[0]
        y_res = abs(first_transform[4])

        for meta in patch_metadata:
            geo_bbox = meta['geo_bbox']
            top, bottom, left, right = _get_edge_deltas(
                geo_bbox, scene_bounds, x_res, delta
            )
            meta['edge_deltas'] = (top, bottom, left, right)

            patch_geo_xmin = geo_bbox[0] + left * x_res
            patch_geo_ymax = geo_bbox[3] - top * y_res
            effective_patch_w = patch_w - left - right
            effective_patch_h = patch_h - top - bottom

            patch_col_start = round((patch_geo_xmin - minx) / x_res)
            patch_row_start = round((maxy - patch_geo_ymax) / y_res)

            meta['bbox'] = (
                patch_col_start,
                patch_row_start,
                patch_col_start + effective_patch_w,
                patch_row_start + effective_patch_h,
            )
    else:
        output_shape, scene_transform = _reconstruct_scene_from_patches(
            patch_metadata, (patch_h, patch_w), delta
        )

    grid_size = chunk_size * 2
    grid = _build_grid_index(patch_metadata, grid_size)

    effective_overlap = max(0, overlap - 2 * delta)
    mask_cache: dict[
        tuple[int, int, int, int], np.typing.NDArray[np.floating[Any]]
    ] = {}

    if output_path is None:
        raise ValueError('output_path is required')
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
                with rasterio.open(meta['file']) as src:
                    logits = src.read().astype(np.float32)

                edge_deltas = meta.get('edge_deltas', (delta, delta, delta, delta))
                top, bottom, left, right = edge_deltas

                bottom_slice = -bottom if bottom > 0 else None
                right_slice = -right if right > 0 else None
                logits = logits[:, top:bottom_slice, left:right_slice]

                if edge_deltas not in mask_cache:
                    mask_cache[edge_deltas] = get_blend_mask(
                        (patch_h, patch_w),
                        effective_overlap,
                        delta,
                        blend_method,
                        edge_deltas=edge_deltas,
                    )
                blend_mask = mask_cache[edge_deltas]

                bbox = meta['bbox']
                patch_col_start = bbox[0]
                patch_row_start = bbox[1]

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

            min_weight = 1e-6
            chunk_weights = np.maximum(chunk_weights, min_weight)
            chunk_output = chunk_output / chunk_weights[None, :, :]
            chunk_labels = np.argmax(chunk_output, axis=0).astype(np.uint8)

            writer.write_chunk(chunk_labels, chunk_y, chunk_x)

    writer.finalize()
