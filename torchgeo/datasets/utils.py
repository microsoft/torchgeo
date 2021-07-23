"""Common dataset utilities."""

import bz2
import contextlib
import gzip
import lzma
import os
import tarfile
import zipfile
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity, download_url

__all__ = (
    "check_integrity",
    "download_url",
    "download_and_extract_archive",
    "extract_archive",
    "BoundingBox",
    "working_dir",
    "collate_dict",
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


def extract_archive(src: str, dst: Optional[str] = None) -> None:
    """Extract an archive.

    Args:
        src: file to be extracted
        dst: directory to extract to (defaults to dirname of ``src``)
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


def download_radiant_mlhub(
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
    """Merge a list of samples for form a mini-batch of Tensors.

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
