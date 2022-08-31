torchgeo.samplers
=================

.. module:: torchgeo.samplers

Samplers
--------

Samplers are used to index a dataset, retrieving a single query at a time. For :class:`~torchgeo.datasets.NonGeoDataset`, dataset objects can be indexed with integers, and PyTorch's builtin samplers are sufficient. For :class:`~torchgeo.datasets.GeoDataset`, dataset objects require a bounding box for indexing. For this reason, we define our own :class:`GeoSampler` implementations below. These can be used like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomGeoSampler

   dataset = Landsat(...)
   sampler = RandomGeoSampler(dataset, size=256, length=10000)
   dataloader = DataLoader(dataset, sampler=sampler)


This data loader will return 256x256 px images, and has an epoch length of 10,000.

Random Geo Sampler
^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomGeoSampler

Grid Geo Sampler
^^^^^^^^^^^^^^^^

.. autoclass:: GridGeoSampler

Pre-chipped Geo Sampler
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PreChippedGeoSampler

Batch Samplers
--------------

When working with large tile-based datasets, randomly sampling patches from each tile can be extremely time consuming. It's much more efficient to choose a tile, load it, warp it to the appropriate :term:`coordinate reference system (CRS)` and resolution, and then sample random patches from that tile to construct a mini-batch of data. For this reason, we define our own :class:`BatchGeoSampler` implementations below. These can be used like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomBatchGeoSampler

   dataset = Landsat(...)
   sampler = RandomBatchGeoSampler(dataset, size=256, batch_size=128, length=10000)
   dataloader = DataLoader(dataset, batch_sampler=sampler)


This data loader will return 256x256 px images, and has a batch size of 128 and an epoch length of 10,000.

Random Batch Geo Sampler
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomBatchGeoSampler

Base Classes
------------

If you want to write your own custom sampler, you can extend one of these abstract base classes.

Geo Sampler
^^^^^^^^^^^

.. autoclass:: GeoSampler

Batch Geo Sampler
^^^^^^^^^^^^^^^^^

.. autoclass:: BatchGeoSampler

Units
-----

By default, the ``size`` parameter specifies the size of the image in *pixel* units. If you would instead like to specify the size in *CRS* units, you can change the ``units`` parameter like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomGeoSampler, Units

   dataset = Landsat(...)
   sampler = RandomGeoSampler(dataset, size=256 * 30, length=10000, units=Units.CRS)
   dataloader = DataLoader(dataset, sampler=sampler)


Assuming that each pixel in the CRS is 30 m, this data loader will return 256x256 px images, and has an epoch length of 10,000.

.. autoclass:: Units
