# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` datasets."""

import abc
import functools
import glob
import math
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.merge
import torch
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader

from .utils import BoundingBox, disambiguate_timestamp

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Dataset.__module__ = "torch.utils.data"


class GeoDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like:

    * coordinates (latitude, longitude)
    * :term:`coordinate reference system (CRS)`
    * resolution

    These kind of datasets are special because they can be combined. For example:

    * Combine Landsat8 and CDL to train a model for crop classification
    * Combine NAIP and Chesapeake to train a model for land cover mapping

    This isn't true for :class:`VisionDataset`, where the lack of geospatial information
    prohibits swapping image sources or target labels.
    """

    #: :term:`coordinate reference system (CRS)` for the dataset.
    crs: CRS

    #: Resolution of the dataset in units of CRS.
    res: float

    def __init__(
        self, transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

    def __add__(self, other: "GeoDataset") -> "ZipDataset":  # type: ignore[override]
        """Merge two GeoDatasets.

        Args:
            other: another dataset

        Returns:
            a single dataset

        Raises:
            ValueError: if other is not a GeoDataset, or if datasets do not overlap,
                or if datasets do not have the same
                :term:`coordinate reference system (CRS)`
        """
        return ZipDataset([self, other])

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns:
            length of the dataset
        """
        count: int = self.index.count(self.index.bounds)
        return count

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: GeoDataset
    bbox: {self.bounds}
    size: {len(self)}"""

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)


class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files."""

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

    #: Regular expression used to extract date from filename.
    #:
    #: The expression should use named groups. The expression may contain any number of
    #: groups. The following groups are specifically searched for by the base class:
    #:
    #: * ``date``: used to calculate ``mint`` and ``maxt`` for ``index`` insertion
    #:
    #: When :attr:`separate_files`` is True, the following additional groups are
    #: searched for to find other files:
    #:
    #: * ``band``: replaced with requested band name
    #: * ``resolution``: replaced with a glob character
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = "%Y%m%d"

    #: True if dataset contains imagery, False if dataset contains mask
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Names of all available bands in the dataset
    all_bands: List[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: List[str] = []

    #: If True, stretch the image from the 2nd percentile to the 98th percentile,
    #: used for plotting
    stretch = False

    #: Color map for the dataset, used for plotting
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root: str,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.root = root
        self.cache = cache

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        self.crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: List[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in getattr(self, "bands", self.all_bands):
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "date" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                        if "resolution" in match.groupdict():
                            start = match.start("resolution")
                            end = match.end("resolution")
                            filename = filename[:start] + "*" + filename[end:]
                    filepath = glob.glob(os.path.join(directory, filename))[0]
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)  # type: ignore[attr-defined]
        else:
            data = self._merge_files(filepaths, query)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = int(round((query.maxx - query.minx) / self.res))
            out_height = int(round((query.maxy - query.miny) / self.res))
            out_shape = (src.count, out_height, out_width)
            dest = src.read(
                out_shape=out_shape, window=from_bounds(*bounds, src.transform)
            )
        else:
            dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res)
        dest = dest.astype(np.int32)

        tensor: Tensor = torch.tensor(dest)  # type: ignore[attr-defined]
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src

    def plot(self, data: Tensor) -> None:
        """Plot a data sample.

        Args:
            data: the data to plot

        Raises:
            AssertionError: if ``is_image`` is True and ``data`` has a different number
                of channels than expected
        """
        array = data.squeeze().numpy()

        if self.is_image:
            bands = getattr(self, "bands", self.all_bands)
            assert array.shape[0] == len(bands)

            # Only plot RGB bands
            if bands and self.rgb_bands:
                indices = np.array([bands.index(band) for band in self.rgb_bands])
                array = array[indices]

            # Convert from CxHxW to HxWxC
            array = np.rollaxis(array, 0, 3)

        if self.cmap:
            # Convert from class labels to RGBA values
            cmap = np.array([self.cmap[i] for i in range(len(self.cmap))])
            array = cmap[array]

        if self.stretch:
            # Stretch to the range of 2nd to 98th percentile
            per02 = np.percentile(array, 2)  # type: ignore[no-untyped-call]
            per98 = np.percentile(array, 98)  # type: ignore[no-untyped-call]
            array = (array - per02) / (per98 - per02)
            array = np.clip(array, 0, 1)

        # Plot the data
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()


class VectorDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as vector files."""

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: float = 0.0001,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.root = root
        self.res = res

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        for filepath in glob.iglob(pathname, recursive=True):
            try:
                with fiona.open(filepath) as src:
                    if crs is None:
                        crs = CRS.from_dict(src.crs)

                    minx, miny, maxx, maxy = src.bounds
                    (minx, maxx), (miny, maxy) = fiona.transform.transform(
                        src.crs, crs.to_dict(), [minx, maxx], [miny, maxy]
                    )
            except fiona.errors.FionaValueError:
                # Skip files that fiona is unable to read
                continue
            else:
                mint = 0
                maxt = sys.maxsize
                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        self.crs = crs

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = []
        for filepath in filepaths:
            with fiona.open(filepath) as src:
                # We need to know the bounding box of the query in the source CRS
                (minx, maxx), (miny, maxy) = fiona.transform.transform(
                    self.crs.to_dict(),
                    src.crs,
                    [query.minx, query.maxx],
                    [query.miny, query.maxy],
                )

                # Filter geometries to those that intersect with the bounding box
                for feature in src.filter(bbox=(minx, miny, maxx, maxy)):
                    # Warp geometries to requested CRS
                    shape = fiona.transform.transform_geom(
                        src.crs, self.crs.to_dict(), feature["geometry"]
                    )
                    shapes.append(shape)

        # Rasterize geometries
        width = (query.maxx - query.minx) / self.res
        height = (query.maxy - query.miny) / self.res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        masks = rasterio.features.rasterize(
            shapes, out_shape=(int(height), int(width)), transform=transform
        )

        sample = {
            "mask": torch.tensor(masks),  # type: ignore[attr-defined]
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(self, data: Tensor) -> None:
        """Plot a data sample.

        Args:
            data: the data to plot
        """
        array = data.squeeze().numpy()

        # Plot the image
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()


class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):  # type: ignore[misc]
    """Abstract base class for classification datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Initialize a new VisionClassificationDataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array
            is_valid_file: A function that takes the path of an Image file and checks if
                the file is a valid file
        """
        # When transform & target_transform are None, ImageFolder.__getitem__(index)
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load a single image and it's class label.

        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        array = np.array(img)
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        label = torch.tensor(label)  # type: ignore[attr-defined]
        return tensor, label


class ZipDataset(GeoDataset):
    """Dataset for merging two or more GeoDatasets.

    For example, this allows you to combine an image source like Landsat8 with a target
    label like CDL.
    """

    def __init__(self, datasets: Sequence[GeoDataset]) -> None:
        """Initialize a new Dataset instance.

        Args:
            datasets: list of datasets to merge

        Raises:
            ValueError: if datasets contains non-GeoDatasets, do not overlap, are not in
                the same :term:`coordinate reference system (CRS)`, or do not have the
                same resolution
        """
        for ds in datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError("ZipDataset only supports GeoDatasets")

        crs = datasets[0].crs
        res = datasets[0].res
        for ds in datasets:
            if ds.crs != crs:
                raise ValueError("Datasets must be in the same CRS")
            if not math.isclose(ds.res, res):
                # TODO: relax this constraint someday
                raise ValueError("Datasets must have the same resolution")

        self.datasets = datasets
        self.crs = crs
        self.res = res

        # Make sure datasets have overlap
        try:
            self.bounds
        except ValueError:
            raise ValueError("Datasets have no overlap")

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # TODO: use collate_dict here to concatenate instead of replace.
        # For example, if using Landsat + Sentinel + CDL, don't want to remove Landsat
        # images and replace with Sentinel images.
        sample = {}
        for ds in self.datasets:
            sample.update(ds[query])
        return sample

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns:
            length of the dataset
        """
        return sum(map(len, self.datasets))

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: ZipDataset
    bbox: {self.bounds}
    size: {len(self)}"""

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        # We want to compute the intersection of all dataset bounds, not the union
        minx = max([ds.bounds[0] for ds in self.datasets])
        maxx = min([ds.bounds[1] for ds in self.datasets])
        miny = max([ds.bounds[2] for ds in self.datasets])
        maxy = min([ds.bounds[3] for ds in self.datasets])
        mint = max([ds.bounds[4] for ds in self.datasets])
        maxt = min([ds.bounds[5] for ds in self.datasets])

        return BoundingBox(minx, maxx, miny, maxy, mint, maxt)
