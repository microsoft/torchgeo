# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common dataset utilities."""

import bz2
import collections
import contextlib
import gzip
import lzma
import os
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import rasterio
import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity, download_url
from torchvision.utils import draw_segmentation_masks

__all__ = (
    "check_integrity",
    "download_url",
    "download_and_extract_archive",
    "extract_archive",
    "BoundingBox",
    "disambiguate_timestamp",
    "working_dir",
    "stack_samples",
    "concat_samples",
    "merge_samples",
    "unbind_samples",
    "rasterio_loader",
    "sort_sentinel2_bands",
    "draw_semantic_segmentation_masks",
    "rgb_to_mask",
    "percentile_normalization",
)


class _rarfile:
    class RarFile:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> Any:
            try:
                import rarfile
            except ImportError:
                raise ImportError(
                    "rarfile is not installed and is required to extract this dataset"
                )

            # TODO: catch exception for when rarfile is installed but not
            # unrar/unar/bsdtar
            return rarfile.RarFile(*self.args, **self.kwargs)

        def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
            pass


class _zipfile:
    class ZipFile:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self) -> Any:
            try:
                # Supports normal zip files, proprietary deflate64 compression algorithm
                import zipfile_deflate64 as zipfile
            except ImportError:
                # Only supports normal zip files
                # https://github.com/python/mypy/issues/1153
                import zipfile  # type: ignore[no-redef]

            return zipfile.ZipFile(*self.args, **self.kwargs)

        def __exit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
            pass


def extract_archive(src: str, dst: Optional[str] = None) -> None:
    """Extract an archive.

    Args:
        src: file to be extracted
        dst: directory to extract to (defaults to dirname of ``src``)

    Raises:
        RuntimeError: if src file has unknown archival/compression scheme
    """
    if dst is None:
        dst = os.path.dirname(src)

    suffix_and_extractor: List[Tuple[Union[str, Tuple[str, ...]], Any]] = [
        (".rar", _rarfile.RarFile),
        (
            (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".tbz", ".txz"),
            tarfile.open,
        ),
        (".zip", _zipfile.ZipFile),
    ]

    for suffix, extractor in suffix_and_extractor:
        if src.endswith(suffix):
            with extractor(src, "r") as f:
                f.extractall(dst)
            return

    suffix_and_decompressor: List[Tuple[str, Any]] = [
        (".bz2", bz2.open),
        (".gz", gzip.open),
        (".xz", lzma.open),
    ]

    for suffix, decompressor in suffix_and_decompressor:
        if src.endswith(suffix):
            dst = os.path.join(dst, os.path.basename(src).replace(suffix, ""))
            with decompressor(src, "rb") as sf, open(dst, "wb") as df:
                df.write(sf.read())
            return

    raise RuntimeError("src file has unknown archival/compression scheme")


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download and extract an archive.

    Args:
        url: URL to download
        download_root: directory to download to
        extract_root: directory to extract to (defaults to ``download_root``)
        filename: download filename (defaults to basename of ``url``)
        md5: checksum for download verification
    """
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root)


def download_radiant_mlhub_dataset(
    dataset_id: str, download_root: str, api_key: Optional[str] = None
) -> None:
    """Download a dataset from Radiant Earth.

    Args:
        dataset_id: the ID of the dataset to fetch
        download_root: directory to download to
        api_key: the API key to use for all requests from the session. Can also be
            passed in via the ``MLHUB_API_KEY`` environment variable, or configured in
            ``~/.mlhub/profiles``.
    """
    try:
        import radiant_mlhub
    except ImportError:
        raise ImportError(
            "radiant_mlhub is not installed and is required to download this dataset"
        )

    dataset = radiant_mlhub.Dataset.fetch(dataset_id, api_key=api_key)
    dataset.download(output_dir=download_root, api_key=api_key)


def download_radiant_mlhub_collection(
    collection_id: str, download_root: str, api_key: Optional[str] = None
) -> None:
    """Download a collection from Radiant Earth.

    Args:
        collection_id: the ID of the collection to fetch
        download_root: directory to download to
        api_key: the API key to use for all requests from the session. Can also be
            passed in via the ``MLHUB_API_KEY`` environment variable, or configured in
            ``~/.mlhub/profiles``.
    """
    try:
        import radiant_mlhub
    except ImportError:
        raise ImportError(
            "radiant_mlhub is not installed and is required to download this collection"
        )

    collection = radiant_mlhub.Collection.fetch(collection_id, api_key=api_key)
    collection.download(output_dir=download_root, api_key=api_key)


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

    # https://github.com/PyCQA/pydocstyle/issues/525
    @overload
    def __getitem__(self, key: int) -> float:  # noqa: D105
        pass

    @overload
    def __getitem__(self, key: slice) -> List[float]:  # noqa: D105
        pass

    def __getitem__(self, key: Union[int, slice]) -> Union[float, List[float]]:
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

    def __contains__(self, other: "BoundingBox") -> bool:
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

    def __or__(self, other: "BoundingBox") -> "BoundingBox":
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

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
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
            raise ValueError(f"Bounding boxes {self} and {other} do not overlap")

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

    def intersects(self, other: "BoundingBox") -> bool:
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


def disambiguate_timestamp(date_str: str, format: str) -> Tuple[float, float]:
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

    # TODO: This doesn't correctly handle literal `%%` characters in format
    # TODO: May have issues with time zones, UTC vs. local time, and DST
    # TODO: This is really tedious, is there a better way to do this?

    if not any([f"%{c}" in format for c in "yYcxG"]):
        # No temporal info
        return 0, sys.maxsize
    elif not any([f"%{c}" in format for c in "bBmjUWcxV"]):
        # Year resolution
        maxt = datetime(mint.year + 1, 1, 1)
    elif not any([f"%{c}" in format for c in "aAwdjcxV"]):
        # Month resolution
        if mint.month == 12:
            maxt = datetime(mint.year + 1, 1, 1)
        else:
            maxt = datetime(mint.year, mint.month + 1, 1)
    elif not any([f"%{c}" in format for c in "HIcX"]):
        # Day resolution
        maxt = mint + timedelta(days=1)
    elif not any([f"%{c}" in format for c in "McX"]):
        # Hour resolution
        maxt = mint + timedelta(hours=1)
    elif not any([f"%{c}" in format for c in "ScX"]):
        # Minute resolution
        maxt = mint + timedelta(minutes=1)
    elif not any([f"%{c}" in format for c in "f"]):
        # Second resolution
        maxt = mint + timedelta(seconds=1)
    else:
        # Microsecond resolution
        maxt = mint + timedelta(microseconds=1)

    mint -= timedelta(microseconds=1)
    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()


@contextlib.contextmanager
def working_dir(dirname: str, create: bool = False) -> Iterator[None]:
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


def _list_dict_to_dict_list(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
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


def _dict_list_to_list_dict(sample: Dict[Any, Sequence[Any]]) -> List[Dict[Any, Any]]:
    """Convert a dictionary of lists to a list of dictionaries.

    Args:
        sample: a dictionary of lists

    Returns:
        a list of dictionaries

    .. versionadded:: 0.2
    """
    uncollated: List[Dict[Any, Any]] = [
        {} for _ in range(max(map(len, sample.values())))
    ]
    for key, values in sample.items():
        for i, value in enumerate(values):
            uncollated[i][key] = value
    return uncollated


def stack_samples(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Stack a list of samples along a new axis.

    Useful for forming a mini-batch of samples to pass to
    :class:`torch.utils.data.DataLoader`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: Dict[Any, Any] = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.stack(value)
    return collated


def concat_samples(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Concatenate a list of samples along an existing axis.

    Useful for joining samples in a :class:`torchgeo.datasets.IntersectionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: Dict[Any, Any] = _list_dict_to_dict_list(samples)
    for key, value in collated.items():
        if isinstance(value[0], Tensor):
            collated[key] = torch.cat(value)
        else:
            collated[key] = value[0]
    return collated


def merge_samples(samples: Iterable[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Merge a list of samples.

    Useful for joining samples in a :class:`torchgeo.datasets.UnionDataset`.

    Args:
        samples: list of samples

    Returns:
        a single sample

    .. versionadded:: 0.2
    """
    collated: Dict[Any, Any] = {}
    for sample in samples:
        for key, value in sample.items():
            if key in collated and isinstance(value, Tensor):
                # Take the maximum so that nodata values (zeros) get replaced
                # by data values whenever possible
                collated[key] = torch.maximum(collated[key], value)
            else:
                collated[key] = value
    return collated


def unbind_samples(sample: Dict[Any, Sequence[Any]]) -> List[Dict[Any, Any]]:
    """Reverse of :func:`stack_samples`.

    Useful for turning a mini-batch of samples into a list of samples. These individual
    samples can then be plotted using a dataset's ``plot`` method.

    Args:
        sample: a mini-batch of samples

    Returns:
         list of samples

    .. versionadded:: 0.2
    """
    for key, values in sample.items():
        if isinstance(values, Tensor):
            sample[key] = torch.unbind(values)
    return _dict_list_to_list_dict(sample)


def rasterio_loader(path: str) -> "np.typing.NDArray[np.int_]":
    """Load an image file using rasterio.

    Args:
        path: path to the image to be loaded

    Returns:
        the image
    """
    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.int_]" = f.read().astype(np.int32)
        # VisionClassificationDataset expects images returned with channels last (HWC)
        array = array.transpose(1, 2, 0)
    return array


def sort_sentinel2_bands(x: str) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split("_")[-1]
    x = os.path.splitext(x)[0]
    if x == "B8A":
        x = "B08A"
    return x


def draw_semantic_segmentation_masks(
    image: Tensor,
    mask: Tensor,
    alpha: float = 0.5,
    colors: Optional[Sequence[Union[str, Tuple[int, int, int]]]] = None,
) -> "np.typing.NDArray[np.uint8]":
    """Overlay a semantic segmentation mask onto an image.

    Args:
        image: tensor of shape (3, h, w) and dtype uint8
        mask: tensor of shape (h, w) with pixel values representing the classes and
            dtype bool
        alpha: alpha blend factor
        colors: list of RGB int tuples, or color strings e.g. red, #FF00FF

    Returns:
        a version of ``image`` overlayed with the colors given by ``mask`` and
            ``colors``
    """
    classes = torch.unique(mask)
    classes = classes[1:]
    class_masks = mask == classes[:, None, None]
    img = draw_segmentation_masks(
        image=image, masks=class_masks, alpha=alpha, colors=colors
    )
    img = img.permute((1, 2, 0)).numpy().astype(np.uint8)
    return cast("np.typing.NDArray[np.uint8]", img)


def rgb_to_mask(
    rgb: "np.typing.NDArray[np.uint8]", colors: List[Tuple[int, int, int]]
) -> "np.typing.NDArray[np.uint8]":
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
    mask: "np.typing.NDArray[np.uint8]" = np.zeros(shape=(h, w), dtype=np.uint8)
    for i, c in enumerate(colors):
        cmask = rgb == c
        # Only update mask if class is present in mask
        if isinstance(cmask, np.ndarray):
            mask[cmask.all(axis=-1)] = i
    return mask


def percentile_normalization(
    img: "np.typing.NDArray[np.int_]",
    lower: float = 2,
    upper: float = 98,
    axis: Optional[Union[int, Sequence[int]]] = None,
) -> "np.typing.NDArray[np.int_]":
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

    Returns
        normalized version of ``img``

    .. versionadded:: 0.2
    """
    assert lower < upper
    lower_percentile = np.percentile(img, lower, axis=axis)
    upper_percentile = np.percentile(img, upper, axis=axis)
    img_normalized: "np.typing.NDArray[np.int_]" = np.clip(
        (img - lower_percentile) / (upper_percentile - lower_percentile), 0, 1
    )
    return img_normalized
