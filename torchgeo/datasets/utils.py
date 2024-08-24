# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common dataset utilities."""

# https://github.com/sphinx-doc/sphinx/issues/11327
from __future__ import annotations

import collections
import contextlib
import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypeAlias, TypedDict, cast, overload

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from torch import Tensor
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
    extract_archive,
)
from torchvision.utils import draw_segmentation_masks

from .errors import DependencyNotFoundError

# Only include import redirects
__all__ = (
    'check_integrity',
    'download_and_extract_archive',
    'download_url',
    'extract_archive',
)


Path: TypeAlias = str | os.PathLike[str]


class Sample(TypedDict, total=False):
    """A single dataset sample.

    .. versionadded:: 0.6
    """

    # Designed to match kornia.constants.DataKey
    image: Tensor
    mask: Tensor
    bbox: Tensor
    bbox_xyxy: Tensor
    bbox_xywh: Tensor
    keypoints: Tensor
    label: Tensor

    # Additional common keys for TorchGeo datasets
    prediction: Tensor
    bounds: BoundingBox
    crs: CRS

    # TODO: Additional dataset-specific keys that should be subclasses
    images: Tensor
    input: Tensor
    boxes: Tensor
    bboxes: Tensor
    masks: Tensor
    labels: Tensor
    prediction_masks: Tensor
    prediction_boxes: Tensor
    prediction_labels: Tensor
    prediction_label: Tensor
    prediction_scores: Tensor
    audio: Tensor
    points: Tensor
    x: Tensor
    y: Tensor
    relative_time: Tensor
    ocean: Tensor
    array: Tensor
    chm: Tensor
    hsi: Tensor
    las: Tensor
    image1: Tensor
    image2: Tensor
    crs1: Tensor
    crs2: Tensor
    magnitude: Tensor
    agb: Tensor
    key: Tensor
    patch: Tensor
    geometry: Tensor
    properties: Tensor
    id: int
    centroid_lat: Tensor
    centroid_lon: Tensor
    content: Tensor
    year: Tensor
    ndvi: Tensor
    filename: str
    category: str
    field_ids: Tensor
    tile_index: Tensor
    transform: Tensor
    src: Tensor
    dst: Tensor
    input_size: Tensor
    output_size: Tensor
    index: BoundingBox


class Batch(Sample):
    """A batch of samples.

    .. versionadded:: 0.6
    """

    # For now, identical to Sample until we can type check tensor shapes


@dataclass(frozen=True)
class BoundingBox:
    """Data class for indexing spatiotemporal data."""

    #: western boundary
    minx: float
    #: eastern boundary
    maxx: float
    #: southern boundary
    miny: float
    #: northern boundary
    maxy: float
    #: earliest boundary
    mint: float
    #: latest boundary
    maxt: float

    def __post_init__(self) -> None:
        """Validate the arguments passed to :meth:`__init__`.

        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)

        .. versionadded:: 0.2
        """
        if self.minx > self.maxx:
            raise ValueError(
                f"Bounding box is invalid: 'minx={self.minx}' > 'maxx={self.maxx}'"
            )
        if self.miny > self.maxy:
            raise ValueError(
                f"Bounding box is invalid: 'miny={self.miny}' > 'maxy={self.maxy}'"
            )
        if self.mint > self.maxt:
            raise ValueError(
                f"Bounding box is invalid: 'mint={self.mint}' > 'maxt={self.maxt}'"
            )

    @overload
    def __getitem__(self, key: int) -> float:
        pass

    @overload
    def __getitem__(self, key: slice) -> list[float]:
        pass

    def __getitem__(self, key: int | slice) -> float | list[float]:
        """Index the (minx, maxx, miny, maxy, mint, maxt) tuple.

        Args:
            key: integer or slice object

        Returns:
            the value(s) at that index

        Raises:
            IndexError: if key is out of bounds
        """
        return [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt][key]

    def __iter__(self) -> Iterator[float]:
        """Container iterator.

        Returns:
            iterator object that iterates over all objects in the container
        """
        yield from [self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt]

    def __contains__(self, other: BoundingBox) -> bool:
        """Whether or not other is within the bounds of this bounding box.

        Args:
            other: another bounding box

        Returns:
            True if other is within this bounding box, else False

        .. versionadded:: 0.2
        """
        return (
            (self.minx <= other.minx <= self.maxx)
            and (self.minx <= other.maxx <= self.maxx)
            and (self.miny <= other.miny <= self.maxy)
            and (self.miny <= other.maxy <= self.maxy)
            and (self.mint <= other.mint <= self.maxt)
            and (self.mint <= other.maxt <= self.maxt)
        )

    def __or__(self, other: BoundingBox) -> BoundingBox:
        """The union operator.

        Args:
            other: another bounding box

        Returns:
            the minimum bounding box that contains both self and other

        .. versionadded:: 0.2
        """
        return BoundingBox(
            min(self.minx, other.minx),
            max(self.maxx, other.maxx),
            min(self.miny, other.miny),
            max(self.maxy, other.maxy),
            min(self.mint, other.mint),
            max(self.maxt, other.maxt),
        )

    def __and__(self, other: BoundingBox) -> BoundingBox:
        """The intersection operator.

        Args:
            other: another bounding box

        Returns:
            the intersection of self and other

        Raises:
            ValueError: if self and other do not intersect

        .. versionadded:: 0.2
        """
        try:
            return BoundingBox(
                max(self.minx, other.minx),
                min(self.maxx, other.maxx),
                max(self.miny, other.miny),
                min(self.maxy, other.maxy),
                max(self.mint, other.mint),
                min(self.maxt, other.maxt),
            )
        except ValueError:
            raise ValueError(f'Bounding boxes {self} and {other} do not overlap')

    @property
    def area(self) -> float:
        """Area of bounding box.

        Area is defined as spatial area.

        Returns:
            area

        .. versionadded:: 0.3
        """
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    @property
    def volume(self) -> float:
        """Volume of bounding box.

        Volume is defined as spatial area times temporal range.

        Returns:
            volume

        .. versionadded:: 0.3
        """
        return self.area * (self.maxt - self.mint)

    def intersects(self, other: BoundingBox) -> bool:
        """Whether or not two bounding boxes intersect.

        Args:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
            and self.mint <= other.maxt
            and self.maxt >= other.mint
        )

    def split(
        self, proportion: float, horizontal: bool = True
    ) -> tuple[BoundingBox, BoundingBox]:
        """Split BoundingBox in two.

        Args:
            proportion: split proportion in range (0,1)
            horizontal: whether the split is horizontal or vertical

        Returns:
            A tuple with the resulting BoundingBoxes

        .. versionadded:: 0.5
        """
        if not (0.0 < proportion < 1.0):
            raise ValueError('Input proportion must be between 0 and 1.')

        if horizontal:
            w = self.maxx - self.minx
            splitx = self.minx + w * proportion
            bbox1 = BoundingBox(
                self.minx, splitx, self.miny, self.maxy, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                splitx, self.maxx, self.miny, self.maxy, self.mint, self.maxt
            )
        else:
            h = self.maxy - self.miny
            splity = self.miny + h * proportion
            bbox1 = BoundingBox(
                self.minx, self.maxx, self.miny, splity, self.mint, self.maxt
            )
            bbox2 = BoundingBox(
                self.minx, self.maxx, splity, self.maxy, self.mint, self.maxt
            )

        return bbox1, bbox2


class Executable:
    """Command-line executable.

    .. versionadded:: 0.6
    """

    def __init__(self, name: Path) -> None:
        """Initialize a new Executable instance.

        Args:
            name: Command name.
        """
        self.name = name

    def __call__(self, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        """Run the command.

        Args:
            args: Arguments to pass to the command.
            kwargs: Keyword arguments to pass to :func:`subprocess.run`.

        Returns:
            The completed process.
        """
        kwargs['check'] = True
        return subprocess.run((self.name, *args), **kwargs)


def disambiguate_timestamp(date_str: str, format: str) -> tuple[float, float]:
    """Disambiguate partial timestamps.

    TorchGeo stores the timestamp of each file in a spatiotemporal R-tree. If the full
    timestamp isn't known, a file could represent a range of time. For example, in the
    CDL dataset, each mask spans an entire year. This method returns the maximum
    possible range of timestamps that ``date_str`` could belong to. It does this by
    parsing ``format`` to determine the level of precision of ``date_str``.

    Args:
        date_str: string representing date and time of a data point
        format: format codes accepted by :meth:`datetime.datetime.strptime`

    Returns:
        (mint, maxt) tuple for indexing
    """
    mint = datetime.strptime(date_str, format)
    format = format.replace('%%', '')

    # TODO: May have issues with time zones, UTC vs. local time, and DST
    # TODO: This is really tedious, is there a better way to do this?

    if not any([f'%{c}' in format for c in 'yYcxG']):
        # No temporal info
        return 0, sys.maxsize
    elif not any([f'%{c}' in format for c in 'bBmjUWcxV']):
        # Year resolution
        maxt = datetime(mint.year + 1, 1, 1)
    elif not any([f'%{c}' in format for c in 'aAwdjcxV']):
        # Month resolution
        if mint.month == 12:
            maxt = datetime(mint.year + 1, 1, 1)
        else:
            maxt = datetime(mint.year, mint.month + 1, 1)
    elif not any([f'%{c}' in format for c in 'HIcX']):
        # Day resolution
        maxt = mint + timedelta(days=1)
    elif not any([f'%{c}' in format for c in 'McX']):
        # Hour resolution
        maxt = mint + timedelta(hours=1)
    elif not any([f'%{c}' in format for c in 'ScX']):
        # Minute resolution
        maxt = mint + timedelta(minutes=1)
    elif not any([f'%{c}' in format for c in 'f']):
        # Second resolution
        maxt = mint + timedelta(seconds=1)
    else:
        # Microsecond resolution
        maxt = mint + timedelta(microseconds=1)

    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()


@contextlib.contextmanager
def working_dir(dirname: Path, create: bool = False) -> Iterator[None]:
    """Context manager for changing directories.

    Args:
        dirname: directory to temporarily change to
        create: if True, create the destination directory
    """
    if create:
        os.makedirs(dirname, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(dirname)

    try:
        yield
    finally:
        os.chdir(cwd)


def _list_dict_to_dict_list(
    samples: Iterable[Mapping[Any, Any]],
) -> dict[Any, list[Any]]:
    """Convert a list of dictionaries to a dictionary of lists.

    Args:
        samples: a list of dictionaries

    Returns:
        a dictionary of lists

    .. versionadded:: 0.2
    """
    collated = collections.defaultdict(list)
    for sample in samples:
        for key, value in sample.items():
            collated[key].append(value)
    return collated


def _dict_list_to_list_dict(
    sample: Mapping[Any, Sequence[Any]],
) -> list[dict[Any, Any]]:
    """Convert a dictionary of lists to a list of dictionaries.

    Args:
        sample: a dictionary of lists

    Returns:
        a list of dictionaries

    .. versionadded:: 0.2
    """
    uncollated: list[dict[Any, Any]] = [
        {} for _ in range(max(map(len, sample.values())))
    ]
    for key, values in sample.items():
        for i, value in enumerate(values):
            uncollated[i][key] = value
    return uncollated


def stack_samples(samples: Iterable[Sample]) -> Batch:
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: Batch = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.stack(value)
    return collated


def concat_samples(samples: Iterable[Sample]) -> Batch:
    """Concatenate a list of samples along an existing axis.

    Useful for joining samples in a :class:`torchgeo.datasets.IntersectionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: Batch = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.cat(value)
        else:
            collated[key] = value[0]
    return collated


def merge_samples(samples: Iterable[Sample]) -> Batch:
    """Merge a list of samples.

    Useful for joining samples in a :class:`torchgeo.datasets.UnionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: Batch = {}
    for sample in samples:
        for key, value in sample.items():
            if key in collated and isinstance(value, Tensor):
                # Take the maximum so that nodata values (zeros) get replaced
                # by data values whenever possible
                collated[key] = torch.maximum(collated[key], value)
            else:
                collated[key] = value
    return collated


def unbind_samples(batch: Batch) -> list[Sample]:
    """Reverse of :func:`stack_samples`.

    Useful for turning a mini-batch of samples into a list of samples. These individual
    samples can then be plotted using a dataset's ``plot`` method.

    Args:
        batch: a mini-batch of samples

    Returns:
         list of samples

    .. versionadded:: 0.2
    """
    for key, values in batch.items():
        if isinstance(values, Tensor):
            batch[key] = torch.unbind(values)
    return _dict_list_to_list_dict(batch)


def rasterio_loader(path: Path) -> np.typing.NDArray[np.int_]:
    """Load an image file using rasterio.

    Args:
        path: path to the image to be loaded

    Returns:
        the image
    """
    with rasterio.open(path) as f:
        array: np.typing.NDArray[np.int_] = f.read().astype(np.int32)
        # NonGeoClassificationDataset expects images returned with channels last (HWC)
        array = array.transpose(1, 2, 0)
    return array


def sort_sentinel2_bands(x: Path) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split('_')[-1]
    x = os.path.splitext(x)[0]
    if x == 'B8A':
        x = 'B08A'
    return x


def draw_semantic_segmentation_masks(
    image: Tensor,
    mask: Tensor,
    alpha: float = 0.5,
    colors: Sequence[str | tuple[int, int, int]] | None = None,
) -> np.typing.NDArray[np.uint8]:
    """Overlay a semantic segmentation mask onto an image.

    Args:
        image: tensor of shape (3, h, w) and dtype uint8
        mask: tensor of shape (h, w) with pixel values representing the classes and
            dtype bool
        alpha: alpha blend factor
        colors: list of RGB int tuples, or color strings e.g. red, #FF00FF

    Returns:
        a version of ``image`` overlaid with the colors given by ``mask`` and
            ``colors``
    """
    classes = torch.from_numpy(np.arange(len(colors) if colors else 0, dtype=np.uint8))
    class_masks = mask == classes[:, None, None]
    img = draw_segmentation_masks(
        image=image.byte(), masks=class_masks, alpha=alpha, colors=colors
    )
    img = img.permute((1, 2, 0)).numpy().astype(np.uint8)
    return cast('np.typing.NDArray[np.uint8]', img)


def rgb_to_mask(
    rgb: np.typing.NDArray[np.uint8], colors: Sequence[tuple[int, int, int]]
) -> np.typing.NDArray[np.uint8]:
    """Converts an RGB colormap mask to a integer mask.

    Args:
        rgb: array mask of coded with RGB tuples
        colors: list of RGB tuples to convert to integer indices

    Returns:
        integer array mask
    """
    assert len(colors) <= 256  # we currently return a uint8 array, so the largest value
    # we can map is 255

    h, w = rgb.shape[:2]
    mask: np.typing.NDArray[np.uint8] = np.zeros(shape=(h, w), dtype=np.uint8)
    for i, c in enumerate(colors):
        cmask = rgb == c
        # Only update mask if class is present in mask
        if isinstance(cmask, np.ndarray):
            mask[cmask.all(axis=-1)] = i
    return mask


def percentile_normalization(
    img: np.typing.NDArray[np.int_],
    lower: float = 2,
    upper: float = 98,
    axis: int | Sequence[int] | None = None,
) -> np.typing.NDArray[np.int_]:
    """Applies percentile normalization to an input image.

    Specifically, this will rescale the values in the input such that values <= the
    lower percentile value will be 0 and values >= the upper percentile value will be 1.
    Using the 2nd and 98th percentile usually results in good visualizations.

    Args:
        img: image to normalize
        lower: lower percentile in range [0,100]
        upper: upper percentile in range [0,100]
        axis: Axis or axes along which the percentiles are computed. The default
            is to compute the percentile(s) along a flattened version of the array.

    Returns:
        normalized version of ``img``

    .. versionadded:: 0.2
    """
    assert lower < upper
    lower_percentile = np.percentile(img, lower, axis=axis)
    upper_percentile = np.percentile(img, upper, axis=axis)
    img_normalized: np.typing.NDArray[np.int_] = np.clip(
        (img - lower_percentile) / (upper_percentile - lower_percentile + 1e-5), 0, 1
    )
    return img_normalized


def path_is_vsi(path: Path) -> bool:
    """Checks if the given path is pointing to a Virtual File System.

    .. note::
       Does not check if the path exists, or if it is a dir or file.

    VSI can for instance be Cloud Storage Blobs or zip-archives.
    They will start with a prefix indicating this.
    For examples of these, see references for the two accepted syntaxes.

    * https://gdal.org/user/virtual_file_systems.html
    * https://rasterio.readthedocs.io/en/latest/topics/datasets.html

    Args:
        path: a directory or file

    Returns:
        True if path is on a virtual file system, else False

    .. versionadded:: 0.6
    """
    return '://' in str(path) or str(path).startswith('/vsi')


def array_to_tensor(array: np.typing.NDArray[Any]) -> Tensor:
    """Converts a :class:`numpy.ndarray` to :class:`torch.Tensor`.

    :func:`torch.from_tensor` rejects numpy types like uint16 that are not supported
    in pytorch. This function instead casts uint16 and uint32 numpy arrays to an
    appropriate pytorch type without loss of precision.

    For example, a uint32 array becomes an int64 tensor. uint64 arrays will continue
    to raise errors since there is no suitable torch dtype.

    The returned tensor is a copy.

    Args:
        array: a :class:`numpy.ndarray`.

    Returns:
        A :class:`torch.Tensor` with the same dtype as array unless array is uint16 or
        uint32, in which case an int32 or int64 Tensor is returned, respectively.

    .. versionadded:: 0.6
    """
    if array.dtype == np.uint16:
        array = array.astype(np.int32)
    elif array.dtype == np.uint32:
        array = array.astype(np.int64)
    return torch.tensor(array)


def lazy_import(name: str) -> Any:
    """Lazy import of *name*.

    Args:
        name: Name of module to import.

    Returns:
        Module import.

    Raises:
        DependencyNotFoundError: If *name* is not installed.

    .. versionadded:: 0.6
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        # Map from import name to package name on PyPI
        name = name.split('.')[0].replace('_', '-')
        module_to_pypi: dict[str, str] = collections.defaultdict(lambda: name)
        module_to_pypi |= {'cv2': 'opencv-python', 'skimage': 'scikit-image'}
        name = module_to_pypi[name]
        msg = f"""\
{name} is not installed and is required to use this dataset. Either run:

$ pip install {name}

to install just this dependency, or:

$ pip install torchgeo[datasets]

to install all optional dataset dependencies."""
        raise DependencyNotFoundError(msg) from None


def which(name: Path) -> Executable:
    """Search for executable *name*.

    Args:
        name: Name of executable to search for.

    Returns:
        Callable executable instance.

    Raises:
        DependencyNotFoundError: If *name* is not installed.

    .. versionadded:: 0.6
    """
    if cmd := shutil.which(name):
        return Executable(cmd)
    else:
        msg = f'{name} is not installed and is required to use this dataset.'
        raise DependencyNotFoundError(msg) from None
