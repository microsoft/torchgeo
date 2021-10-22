# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common dataset utilities."""

import bz2
import contextlib
import gzip
import lzma
import os
import sys
import tarfile
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets.utils import check_integrity, download_url

__all__ = (
    "check_integrity",
    "download_url",
    "download_and_extract_archive",
    "extract_archive",
    "BoundingBox",
    "disambiguate_timestamp",
    "working_dir",
    "collate_dict",
    "rasterio_loader",
    "dataset_split",
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
        (".zip", zipfile.ZipFile),
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


class BoundingBox(Tuple[float, float, float, float, float, float]):
    """Data class for indexing spatiotemporal data.

    Attributes:
        minx (float): western boundary
        maxx (float): eastern boundary
        miny (float): southern boundary
        maxy (float): northern boundary
        mint (float): earliest boundary
        maxt (float): latest boundary
    """

    def __new__(
        cls,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        mint: float,
        maxt: float,
    ) -> "BoundingBox":
        """Create a new instance of BoundingBox.

        Args:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary

        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)
        """
        if minx > maxx:
            raise ValueError(f"Bounding box is invalid: 'minx={minx}' > 'maxx={maxx}'")
        if miny > maxy:
            raise ValueError(f"Bounding box is invalid: 'miny={miny}' > 'maxy={maxy}'")
        if mint > maxt:
            raise ValueError(f"Bounding box is invalid: 'mint={mint}' > 'maxt={maxt}'")

        # Using super() doesn't work with mypy, see:
        # https://stackoverflow.com/q/60611012/5828163
        return tuple.__new__(cls, [minx, maxx, miny, maxy, mint, maxt])

    def __init__(
        self,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        mint: float,
        maxt: float,
    ) -> None:
        """Initialize a new instance of BoundingBox.

        Args:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary
        """
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.mint = mint
        self.maxt = maxt

    def __getnewargs__(self) -> Tuple[float, float, float, float, float, float]:
        """Values passed to the ``__new__()`` method upon unpickling.

        Returns:
            tuple of bounds
        """
        return self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt

    def __repr__(self) -> str:
        """Return the formal string representation of the object.

        Returns:
            formal string representation
        """
        return (
            f"{self.__class__.__name__}(minx={self.minx}, maxx={self.maxx}, "
            f"miny={self.miny}, maxy={self.maxy}, mint={self.mint}, maxt={self.maxt})"
        )

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


def collate_dict(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a list of samples to form a mini-batch of Tensors.

    Args:
        samples: list of samples

    Returns:
        a single sample
    """
    collated = {}
    for key, value in samples[0].items():
        if isinstance(value, Tensor):
            collated[key] = torch.stack([sample[key] for sample in samples])
        else:
            collated[key] = [
                sample[key] for sample in samples
            ]  # type: ignore[assignment]
    return collated


def rasterio_loader(path: str) -> np.ndarray:  # type: ignore[type-arg]
    """Load an image file using rasterio.

    Args:
        path: path to the image to be loaded

    Returns:
        the image
    """
    with rasterio.open(path) as f:
        array: np.ndarray = f.read().astype(np.int32)  # type: ignore[type-arg]
        # VisionClassificationDataset expects images returned with channels last (HWC)
        array = array.transpose(1, 2, 0)
    return array


def dataset_split(
    dataset: Dataset[Any], val_pct: float, test_pct: Optional[float] = None
) -> List[Subset[Any]]:
    """Split a torch Dataset into train/val/test sets.

    If ``test_pct`` is not set then only train and validation splits are returned.

    Args:
        dataset: dataset to be split into train/val or train/val/test subsets
        val_pct: percentage of samples to be in validation set
        test_pct: (Optional) percentage of samples to be in test set
    Returns:
        a list of the subset datasets. Either [train, val] or [train, val, test]
    """
    if test_pct is None:
        val_length = int(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        return random_split(dataset, [train_length, val_length])
    else:
        val_length = int(len(dataset) * val_pct)
        test_length = int(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(dataset, [train_length, val_length, test_length])
