torchgeo.samplers
=================

.. module:: torchgeo.samplers

Constants and Enums
--------

When sampling imagery, the Samplers below can sample tiles in one of two units: 1) the units specified by the CRS of the rasters, or 2) pixels.

The Units enum within :module:`~torchgeo.samplers.constants` specifies these two sampling modes. By default, the Samplers use Units.PIXELS.


Samplers
--------

Samplers are used to index a dataset, retrieving a single query at a time. For :class:`~torchgeo.datasets.VisionDataset`, dataset objects can be indexed with integers, and PyTorch's builtin samplers are sufficient. For :class:`~torchgeo.datasets.GeoDataset`, dataset objects require a bounding box for indexing. For this reason, we define our own :class:`GeoSampler` implementations below. These can be used like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomGeoSampler
   from torchgeo.samplers.constants import Units

   dataset = Landsat(...)
   sampler = RandomGeoSampler(dataset, size=1000, length=100, units=Units.PIXELS)
   dataloader = DataLoader(dataset, sampler=sampler)


Random Geo Sampler
^^^^^^^^^^^^^^^^^^

.. autoclass:: RandomGeoSampler

Grid Geo Sampler
^^^^^^^^^^^^^^^^

.. autoclass:: GridGeoSampler

Batch Samplers
--------------

When working with large tile-based datasets, randomly sampling patches from each tile can be extremely time consuming. It's much more efficient to choose a tile, load it, warp it to the appropriate :term:`coordinate reference system (CRS)` and resolution, and then sample random patches from that tile to construct a mini-batch of data. For this reason, we define our own :class:`BatchGeoSampler` implementations below. These can be used like so:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.datasets import Landsat
   from torchgeo.samplers import RandomBatchGeoSampler
   from torchgeo.samplers.constants import Units

   dataset = Landsat(...)
   sampler = RandomBatchGeoSampler(dataset, size=1000, batch_size=10, length=100, units=Units.PIXELS)
   dataloader = DataLoader(dataset, batch_sampler=sampler)


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
