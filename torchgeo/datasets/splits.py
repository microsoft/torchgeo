# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset splitting utilities."""

import itertools
from collections.abc import Sequence
from copy import deepcopy
from itertools import accumulate
from math import floor, isclose
from typing import cast

import shapely
from rtree.index import Index, Property
from shapely import LineString
from torch import Generator, default_generator, randint, randperm

from ..datasets import GeoDataset
from .utils import BoundingBox


def _fractions_to_lengths(fractions: Sequence[float], total: int) -> Sequence[int]:
    """Utility to divide a number into a list of integers according to fractions.

    Implementation based on :meth:`torch.utils.data.random_split`.

    Args:
        fractions: list of fractions
        total: total to be divided

    Returns:
        List of lengths.

    .. versionadded:: 0.5
    """
    lengths = [floor(frac * total) for frac in fractions]
    remainder = int(total - sum(lengths))
    # Add 1 to all the lengths in round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(lengths)
        lengths[idx_to_add_at] += 1
    return lengths


def random_bbox_assignment(
    dataset: GeoDataset,
    lengths: Sequence[float],
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    """Split a GeoDataset randomly assigning its index's BoundingBoxes.

    This function will go through each BoundingBox in the GeoDataset's index and
    randomly assign it to new GeoDatasets.

    Args:
        dataset: dataset to be split
        lengths: lengths or fractions of splits to be produced
        generator: (optional) generator used for the random permutation

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    if not (isclose(sum(lengths), 1) or isclose(sum(lengths), len(dataset))):
        raise ValueError(
            "Sum of input lengths must equal 1 or the length of dataset's index."
        )

    if any(n <= 0 for n in lengths):
        raise ValueError('All items in input lengths must be greater than 0.')

    if isclose(sum(lengths), 1):
        lengths = _fractions_to_lengths(lengths, len(dataset))
    lengths = cast(Sequence[int], lengths)

    indices = randperm(sum(lengths), generator=generator)

    new_datasets = []
    for offset, length in zip(itertools.accumulate(lengths), lengths):
        ds = deepcopy(dataset)
        ds.index = dataset.index.iloc[indices[offset - length : offset]]
        new_datasets.append(ds)

    return new_datasets


def random_bbox_splitting(
    dataset: GeoDataset,
    fractions: Sequence[float],
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    """Split a GeoDataset randomly splitting its index's BoundingBoxes.

    This function will go through each BoundingBox in the GeoDataset's index,
    split it in a random direction and assign the resulting BoundingBoxes to
    new GeoDatasets.

    Args:
        dataset: dataset to be split
        fractions: fractions of splits to be produced
        generator: generator used for the random permutation

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    if not isclose(sum(fractions), 1):
        raise ValueError('Sum of input fractions must equal 1.')

    if any(n <= 0 for n in fractions):
        raise ValueError('All items in input fractions must be greater than 0.')

    new_datasets = [deepcopy(dataset) for _ in fractions]

    for i in range(len(dataset)):
        geometry_remaining = dataset.index.geometry.iloc[i]
        fraction_remaining = 1.0

        # Randomly choose the split direction
        horizontal, flip = randint(0, 2, (2,), generator=generator)
        for j, fraction in enumerate(fractions):
            if isclose(fraction_remaining, fraction):
                # For the last fraction, no need to split again
                new_geometry = geometry_remaining
            else:
                # Create a new_geometry from geometry_remaining
                minx, miny, maxx, maxy = geometry_remaining.bounds

                if flip:
                    frac = fraction_remaining - fraction
                else:
                    frac = fraction

                if horizontal:
                    splity = miny + (maxy - miny) * frac / fraction_remaining
                    line = LineString([(minx, splity), (maxx, splity)])
                else:
                    splitx = minx + (maxx - minx) * frac / fraction_remaining
                    line = LineString([(splitx, miny), (splitx, maxy)])

                geom1, geom2 = shapely.ops.split(geometry_remaining, line).geoms
                if horizontal:
                    if flip:
                        if geom1.centroid.y < splity:
                            geometry_remaining, new_geometry = geom1, geom2
                        else:
                            new_geometry, geometry_remaining = geom1, geom2
                    else:
                        if geom1.centroid.y < splity:
                            new_geometry, geometry_remaining = geom1, geom2
                        else:
                            geometry_remaining, new_geometry = geom1, geom2
                else:
                    if flip:
                        if geom1.centroid.x < splitx:
                            geometry_remaining, new_geometry = geom1, geom2
                        else:
                            new_geometry, geometry_remaining = geom1, geom2
                    else:
                        if geom1.centroid.x < splitx:
                            new_geometry, geometry_remaining = geom1, geom2
                        else:
                            geometry_remaining, new_geometry = geom1, geom2

            new_datasets[j].index.iloc[i].geometry = new_geometry

            fraction_remaining -= fraction
            horizontal = not horizontal

    return new_datasets


def random_grid_cell_assignment(
    dataset: GeoDataset,
    fractions: Sequence[float],
    grid_size: int = 6,
    generator: Generator | None = default_generator,
) -> list[GeoDataset]:
    """Overlays a grid over a GeoDataset and randomly assigns cells to new GeoDatasets.

    This function will go through each BoundingBox in the GeoDataset's index, overlay
    a grid over it, and randomly assign each cell to new GeoDatasets.

    Args:
        dataset: dataset to be split
        fractions: fractions of splits to be produced
        grid_size: number of rows and columns for the grid
        generator: generator used for the random permutation

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    if not isclose(sum(fractions), 1):
        raise ValueError('Sum of input fractions must equal 1.')

    if any(n <= 0 for n in fractions):
        raise ValueError('All items in input fractions must be greater than 0.')

    if grid_size < 2:
        raise ValueError('Input grid_size must be greater than 1.')

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in fractions
    ]

    lengths = _fractions_to_lengths(fractions, len(dataset) * grid_size**2)

    cells = []

    # Generate the grid's cells for each bbox in index
    for i, hit in enumerate(
        dataset.index.intersection(dataset.index.bounds, objects=True)
    ):
        minx, maxx, miny, maxy, mint, maxt = hit.bounds

        stridex = (maxx - minx) / grid_size
        stridey = (maxy - miny) / grid_size

        cells.extend(
            [
                (
                    (
                        minx + x * stridex,
                        minx + (x + 1) * stridex,
                        miny + y * stridey,
                        miny + (y + 1) * stridey,
                        mint,
                        maxt,
                    ),
                    hit.object,
                )
                for x in range(grid_size)
                for y in range(grid_size)
            ]
        )

    # Randomly assign cells to each new index
    cells = [cells[i] for i in randperm(len(cells), generator=generator)]

    for i, length in enumerate(lengths):
        for j in range(length):
            cell = cells.pop()
            new_indexes[i].insert(j, cell[0], cell[1])

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets


def roi_split(dataset: GeoDataset, rois: Sequence[BoundingBox]) -> list[GeoDataset]:
    """Split a GeoDataset intersecting it with a ROI for each desired new GeoDataset.

    Args:
        dataset: dataset to be split
        rois: regions of interest of splits to be produced

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in rois
    ]

    for i, roi in enumerate(rois):
        if any(roi.intersects(x) and (roi & x).area > 0 for x in rois[i + 1 :]):
            raise ValueError("ROIs in input rois can't overlap.")

        j = 0
        for hit in dataset.index.intersection(tuple(roi), objects=True):
            box = BoundingBox(*hit.bounds)
            new_box = box & roi
            if new_box.area > 0:
                new_indexes[i].insert(j, tuple(new_box), hit.object)
                j += 1

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets


def time_series_split(
    dataset: GeoDataset, lengths: Sequence[float | tuple[float, float]]
) -> list[GeoDataset]:
    """Split a GeoDataset on its time dimension to create non-overlapping GeoDatasets.

    Args:
        dataset: dataset to be split
        lengths: lengths, fractions or pairs of timestamps (start, end) of splits
            to be produced

    Returns:
        A list of the subset datasets.

    .. versionadded:: 0.5
    """
    minx, maxx, miny, maxy, mint, maxt = dataset.bounds

    totalt = maxt - mint

    if not all(isinstance(x, tuple) for x in lengths):
        lengths = cast(Sequence[float], lengths)

        if not (isclose(sum(lengths), 1) or isclose(sum(lengths), totalt)):
            raise ValueError(
                "Sum of input lengths must equal 1 or the dataset's time length."
            )

        if any(n <= 0 for n in lengths):
            raise ValueError('All items in input lengths must be greater than 0.')

        if isclose(sum(lengths), 1):
            lengths = [totalt * f for f in lengths]

        lengths = [
            (mint + offset - length, mint + offset)  # type: ignore[operator]
            for offset, length in zip(accumulate(lengths), lengths)
        ]

    lengths = cast(Sequence[tuple[float, float]], lengths)

    new_indexes = [
        Index(interleaved=False, properties=Property(dimension=3)) for _ in lengths
    ]

    _totalt = 0.0
    for i, (start, end) in enumerate(lengths):
        if start >= end:
            raise ValueError(
                'Pairs of timestamps in lengths must have end greater than start.'
            )

        if start < mint or end > maxt:
            raise ValueError(
                "Pairs of timestamps in lengths can't be out of dataset's time bounds."
            )

        if any(start < x < end or start < y < end for x, y in lengths[i + 1 :]):
            raise ValueError("Pairs of timestamps in lengths can't overlap.")

        # Remove one microsecond from each BoundingBox's maxt to avoid overlapping
        offset = 0 if i == len(lengths) - 1 else 1e-6
        roi = BoundingBox(minx, maxx, miny, maxy, start, end - offset)
        j = 0
        for hit in dataset.index.intersection(tuple(roi), objects=True):
            box = BoundingBox(*hit.bounds)
            new_box = box & roi
            if new_box.volume > 0:
                new_indexes[i].insert(j, tuple(new_box), hit.object)
                j += 1

        _totalt += end - start

    if not isclose(_totalt, totalt):
        raise ValueError(
            "Pairs of timestamps in lengths must cover dataset's time bounds."
        )

    new_datasets = []
    for index in new_indexes:
        ds = deepcopy(dataset)
        ds.index = index
        new_datasets.append(ds)

    return new_datasets
