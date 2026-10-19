Frequently Asked Questions
==========================

Why does my data loader fail with an integer index error?
---------------------------------------------------------

Errors such as ``TypeError: object of type 'int' has no len()`` occur because, unlike :class:`~torchgeo.datasets.NonGeoDataset`, :class:`~torchgeo.datasets.GeoDataset` is indexed by spatiotemporal slices instead of integers. PyTorch's default sampler returns integer indices, so use a :class:`~torchgeo.samplers.GeoSampler` implementation instead:

.. code-block:: python

   from torch.utils.data import DataLoader

   from torchgeo.samplers import RandomPatchSampler

   sampler = RandomPatchSampler(dataset, size=256, length=10000)
   dataloader = DataLoader(dataset, sampler=sampler)

The default PyTorch collation function supports standard TorchGeo samples, so :func:`~torchgeo.datasets.stack_samples` is usually unnecessary.

How can I speed up geospatial sampling?
---------------------------------------

Preprocessing and storage choices can significantly affect sampling performance:

* Warp all files to the same coordinate reference system (CRS) and resolution.
* Use target-aligned pixels (TAP) and Cloud Optimized GeoTIFFs (COGs).
* Experiment with compression, file formats, and data types.
* Use a block size that is a multiple of the patch size.
* Use the largest patch size and batch size that fit in memory.
* Experiment with ``GDAL_CACHEMAX``.
* Follow the `Pangeo COG best practices <https://github.com/pangeo-data/cog-best-practices>`_.

See the :doc:`../tutorials/geospatial` tutorial for preprocessing examples.
