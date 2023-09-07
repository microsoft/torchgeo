# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
import random
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from typing import Callable, Optional, Union

import torch
from dateutil.relativedelta import relativedelta
from dateutil.rrule import DAILY, HOURLY, MONTHLY, WEEKLY, YEARLY, rrule
from rtree.index import Index, Property
from torch.utils.data import Sampler

from torchgeo.datasets import BoundingBox, GeoDataset

from .constants import Units
from .utils import _to_tuple, get_random_bounding_box, tile_to_chips


class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: Optional[BoundingBox] = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)
        else:
            self.index = Index(interleaved=False, properties=Property(dimension=3))
            hits = dataset.index.intersection(tuple(roi), objects=True)
            for hit in hits:
                bbox = BoundingBox(*hit.bounds) & roi
                self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        .. versionchanged:: 0.4
           ``length`` parameter is now optional, a reasonable default will be used

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class TimeWindowGeoSampler(GeoSampler):
    """Samples geospatial time instances for time-series tasks.

    This is particularly useful for time-series datasets where you would
    like to sample sequential inputs of specified length or time duration
    and predict targets beyond such a time frame.

    .. versionadded:: 0.5.0
    """

    allowed_time_units = ["hours", "days", "weeks", "months", "years"]
    rrule_mappings = {
        "hours": HOURLY,
        "days": DAILY,
        "weeks": WEEKLY,
        "months": MONTHLY,
        "years": YEARLY,
    }

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        encoder_length: int,
        prediction_length: int,
        time_unit: str,
        time_delta: int = 0,
        consecutive: bool = True,
        time_range: Optional[tuple[datetime, datetime]] = None,
        roi: Optional[BoundingBox] = None,
        size_units: Units = Units.PIXELS,
        max_samples_per_geolocation: int = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch` in the desired time-series sample
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset each with a randomly chosen sequential time-series)
            encoder_length: the number of ``time_units`` that should form a time window
                for a single sample for the input sequence
            prediction_length: the number of ``time_units`` that should form a time
                window for a single sample for the target sequence that follows
                the ``encoder_length``
            time_unit: unit of time, accepting 'hours', 'days', 'weeks', 'months',
                 'years'
            time_delta: the number of ``time_units`` that specify the horizon
                between the last time step of the encoding sequence and the first
                time step of the prediction sequence
            consecutive: whether or not the target sequence should follow the input
                sequence. If "False", the target sequence will be the last
                ``prediction_length`` number of time steps from the input sequence
            time_range: beginning and end string timestamp to consider
                drawing samples from, needs to follow the date format convention of
                ``dataset`` files
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``bounds)
            size_units: defines if ``size`` is in pixel or CRS units
            max_samples_per_geolocation: max number of time-series sequences for any
                randomly sampled geo patch
        """
        # TODO handle encoder length of 1
        # roi should be overwritten with time_range specifications
        if roi is None:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)

        # by default take roi of dataset but in case time range
        # is specified overwrite default roi
        if time_range is not None:
            start_date, end_date = time_range
            assert end_date > start_date, "End time needs to be later than start time."
            roi = BoundingBox(
                minx=roi.minx,
                maxx=roi.maxx,
                miny=roi.miny,
                maxy=roi.maxy,
                mint=start_date.timestamp(),
                maxt=end_date.timestamp(),
            )

        assert time_delta >= 0, "Time delta has to be non-negative."
        self.time_delta = time_delta

        if not consecutive:
            assert (
                encoder_length > prediction_length
            ), "Prediction length has to be less than encoder length."

        super().__init__(dataset, roi)

        self.size = _to_tuple(size)

        # check time unit
        assert (
            time_unit in self.allowed_time_units
        ), f"Currently, only supporting one of {self.allowed_time_units} as time unit."
        self.time_unit = time_unit
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.consecutive = consecutive

        if size_units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.length = 0

        self.hits: list[list[float]] = []
        areas: list[float] = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                if bounds.area > 0:
                    rows, cols = tile_to_chips(bounds, self.size)
                    self.length += rows * cols
                else:
                    self.length += 1
                self.hits.append(hit)
                areas.append(bounds.area)
        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

        # get all possible subsequence within the time range
        possible_input_time_ranges = self._compute_subsequences()

        # if there are a lot of possible subsequences
        # limit them
        random.shuffle(possible_input_time_ranges)
        if max_samples_per_geolocation is not None:
            possible_input_time_ranges = possible_input_time_ranges[
                :max_samples_per_geolocation
            ]

        self.possible_input_time_ranges = possible_input_time_ranges

    def __iter__(self) -> Iterator[tuple[BoundingBox]]:
        """Return the index of a dataset.

        Returns:
            Tuple of (minx, maxx, miny, maxy, mint, maxt) coordinates to query a dataset
            sequentially for input and target sequence respectively
        """
        # outer loop: pick a location, inner loop sliding window across time
        # this will give you the same location with different time horizons

        # use a count to respect the specified length
        count: int = 0
        # in the outer loop pick a geospatial location
        # while count < len(self):
        for _ in range(len(self)):
            if count == len(self):
                break
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            # in the inner loop slide across time to yield locations with different
            # time horizons
            for input_sequence in self.possible_input_time_ranges:
                if count == len(self):
                    break

                # adjust the bounding box for the candidate time sequence
                input_time_bbox = BoundingBox(
                    minx=bounding_box.minx,
                    maxx=bounding_box.maxx,
                    miny=bounding_box.miny,
                    maxy=bounding_box.maxy,
                    mint=input_sequence[0].timestamp(),
                    maxt=input_sequence[1].timestamp(),
                )
                # generate subsequence target
                input_time_bbox, target_time_bbox = self._retrieve_sequential_query(
                    input_time_bbox
                )

                # check if they can be found in index
                input_hits = list(self.index.intersection(tuple(input_time_bbox)))
                target_hits = list(self.index.intersection(tuple(target_time_bbox)))

                if input_hits and target_hits:
                    count += 1
                    yield (input_time_bbox, target_time_bbox)

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length

    def _compute_subsequences(self) -> list[list[datetime]]:
        """Compute the possible subsequences within the time-horizon.

        Returns:
            all subsequences in the time dimension
        """
        # dates incremented by 1 time unit up to end_time_range - self.prediction_length
        # because target window will be sequentially defined in
        # _retrieve_sequential_query
        all_timestamps_within_time_range = [
            dt
            for dt in rrule(
                freq=self.rrule_mappings[self.time_unit],
                dtstart=datetime.fromtimestamp(self.roi.mint),
                until=datetime.fromtimestamp(self.roi.maxt)
                - relativedelta(**{self.time_unit: self.prediction_length}),
            )
        ]
        # list of list with all timestamps within a time frame
        subsequences: list[list[datetime]] = list(
            map(
                list,
                zip(
                    *(
                        all_timestamps_within_time_range[i:]
                        for i in range(self.encoder_length)
                    )
                ),
            )
        )
        # only keep start and end date to define of a a time frame
        subsequences = [[sub[0], sub[-1]] for sub in subsequences]

        return subsequences

    def _retrieve_sequential_query(
        self, query: BoundingBox
    ) -> tuple[BoundingBox, BoundingBox]:
        """Get a sequential query based on *encoder_length* and *prediction_length*.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates

        Returns:
            Tuple of sequential bounding boxes in time with same location for input
            and target
        """
        # select a time range covering both sample and target window
        input_query = BoundingBox(
            minx=query.minx,
            maxx=query.maxx,
            miny=query.miny,
            maxy=query.maxy,
            mint=query.mint,
            maxt=(
                datetime.fromtimestamp(query.maxt)
                - timedelta(
                    microseconds=1
                )  # subtract otherwise there is overlap to target
            ).timestamp(),
        )

        if self.consecutive:
            target_query = BoundingBox(
                minx=query.minx,
                maxx=query.maxx,
                miny=query.miny,
                maxy=query.maxy,
                mint=input_query.maxt,
                maxt=(
                    datetime.fromtimestamp(input_query.maxt)
                    + relativedelta(**{self.time_unit: self.prediction_length})
                    + relativedelta(**{self.time_unit: self.time_delta})
                ).timestamp(),
            )
        else:
            target_query = BoundingBox(
                minx=query.minx,
                maxx=query.maxx,
                miny=query.miny,
                maxy=query.maxy,
                mint=(
                    datetime.fromtimestamp(query.maxt)
                    - relativedelta(**{self.time_unit: self.prediction_length})
                    - timedelta(microseconds=1)
                ).timestamp(),
                maxt=(
                    datetime.fromtimestamp(query.maxt) - timedelta(microseconds=1)
                ).timestamp(),
            )

        return (input_query, target_query)


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(hit)

        self.length = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


class PreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.NonGeoDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Optional[BoundingBox] = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.3

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
        """
        super().__init__(dataset, roi)
        self.shuffle = shuffle

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            self.hits.append(hit)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        for idx in generator(len(self)):
            yield BoundingBox(*self.hits[idx].bounds)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.hits)
