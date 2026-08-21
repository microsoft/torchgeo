torchgeo.samplers
=================

.. module:: torchgeo.samplers

.. toctree::
   :maxdepth: 0
   :hidden:
   :glob:

   samplers/*

Samplers
--------

Samplers are used to index a dataset, retrieving a single query at a time. For :class:`~torchgeo.datasets.NonGeoDataset`, dataset objects can be indexed with integers, and PyTorch's builtin samplers are sufficient. For :class:`~torchgeo.datasets.GeoDataset`, dataset objects require a bounding box for indexing. For this reason, we define our own :class:`GeoSampler` implementations below. These can be used like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomPatchSampler

   dataset = Landsat(...)
   sampler = RandomPatchSampler(dataset, size=256, length=10000)
   dataloader = DataLoader(dataset, sampler=sampler)


This data loader will return 256x256 px images, and has an epoch length of 10,000.

Some datasets have static mosaics, and only spatial sampling is important. Other datasets include time series observations, with no spatial component. Finally, many datasets for satellite image time series (SITS) include both. TorchGeo provides a number of spatial and temporal sampling strategies, which can be combined using the ``@`` operator:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import GriddedPatchSampler, SequentialPeriodSampler

   dataset = Landsat(..., time_series=True)
   spatial_sampler = GriddedPatchSampler(dataset, size=256, stride=128)
   temporal_sampler = SequentialPeriodSampler(dataset, freq='Y')
   spatiotemporal_sampler = spatial_sampler @ temporal_sampler
   dataloader = DataLoader(dataset, sampler=spatiotemporal_sampler)


This data loader will iterate over all valid locations and all valid times, with annual frequency, returning a data cube for each sample.

The majority of spatial and temporal samplers have both random and sequential variants. Random variants are recommended at training time to maximize the diversity of the dataset, while sequential variants are recommended at inference time to ensure complete coverage of the dataset.

Spatial Samplers
^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   RandomPatchSampler
   GriddedPatchSampler

Temporal Samplers
^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   RandomTimestampSampler
   SequentialTimestampSampler
   RandomTimedeltaSampler
   SequentialTimedeltaSampler
   RandomPeriodSampler
   SequentialPeriodSampler

Base Classes
------------

If you want to write your own custom sampler, you can extend one of these abstract base classes.

.. autosummary::
   :nosignatures:

   GeoSampler
   SpatialSampler
   TemporalSampler
   SpatioTemporalSampler

Units
-----

By default, the ``size`` parameter specifies the size of the image in *pixel* units. If you would instead like to specify the size in *CRS* units, you can change the ``units`` parameter like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomPatchSampler, Units

   dataset = Landsat(...)
   sampler = RandomPatchSampler(dataset, size=256 * 30, length=10000, units=Units.CRS)
   dataloader = DataLoader(dataset, sampler=sampler)


Assuming that each pixel in the CRS is 30 m, this data loader will return 256x256 px images, and has an epoch length of 10,000.

.. autoclass:: Units
