.. _STAC:

STAC
====

:class:`~torchgeo.datasets.STACDataset` turns a STAC GeoParquet item table into a
TorchGeo raster dataset. It discovers items once, when the dataset is created, then
reads pixels lazily as samplers query it, so training and inference make no live
STAC API calls. Point it at a local file, ``file://`` URI, HTTP(S) URL, or an
``fsspec``-supported URI such as ``s3://``, ``gs://``, or ``abfs://``.

The input is a STAC *item* GeoParquet table: it must carry ``geometry``, ``assets``,
and a time column (``datetime``, or ``start_datetime`` and ``end_datetime``). A plain
GeoParquet index without these STAC fields will not work.

A few things worth knowing the first time:

* ``intersects_bbox`` and ``time_range`` choose which *items* enter the dataset; they
  do not crop pixels. Pixel extents come from your sampler or slice. Use them, with
  ``filters``, to keep the index small before TorchGeo builds its in-memory index.
  ``max_index_items`` guards against loading a huge table by accident (set it to
  ``None`` to index everything).
* Sampling uses the CRS and resolution of the first selected raster asset unless
  ``crs`` or ``res`` is supplied. For remote datasets, pass both when possible to
  avoid an extra network open during construction.
* For private buckets or signed URLs, pass ``storage_options`` for the GeoParquet
  table and ``sign_href`` for raster assets. ``storage_options`` are passed to
  ``fsspec``; HTTP(S) options without a ``headers`` key are treated as headers.
  Cloud URIs require the matching ``fsspec`` backend, such as ``s3fs``, ``gcsfs``,
  or ``adlfs``. ``dataset.files`` shows the unsigned, canonical hrefs; signing
  happens at read time.

Usage
-----

.. code-block:: python

   from torchgeo.datasets import STACDataset

   dataset = STACDataset(
       index_path='s3://bucket/path/items.parquet',
       asset_keys=('B04', 'B08'),
       intersects_bbox=(8.7, 41.7, 8.8, 41.8),
       time_range=('2015-07-04T00:00:00Z', '2015-07-05T23:59:59Z'),
       filters=[('eo:cloud_cover', '<', 5)],
   )

``filters`` accepts PyArrow-style filter tuples passed through to
``geopandas.read_parquet``. Pass ``time_series=True`` to stack intersecting items
chronologically instead of merging them into a mosaic.

.. currentmodule:: torchgeo.datasets
.. autoclass:: STACDataset
