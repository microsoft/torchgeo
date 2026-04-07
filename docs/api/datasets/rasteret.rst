.. _Rasteret:

Rasteret
========

.. currentmodule:: torchgeo.datasets

``RasteretDataset`` is an experimental raster dataset entry point for
Rasteret-backed collections.

Scope
-----

``RasteretDataset`` adds a new dataset class. It does **not** change how
existing TorchGeo raster datasets work, and it does **not** auto-convert every
TorchGeo dataset into Rasteret.

The scope is:

1. Prepare or load a Rasteret ``Collection``.
2. Pass that collection into ``RasteretDataset``.
3. Continue with normal TorchGeo samplers, transforms, and dataloaders.

Quick Start
-----------

.. code-block:: python

   import rasteret
   from torchgeo.datasets import RasteretDataset
   from torchgeo.samplers import RandomGeoSampler

   collection = rasteret.build(
       "earthsearch/sentinel-2-l2a",
       name="s2-bangalore",
       bbox=(77.5, 12.9, 77.7, 13.1),
       date_range=("2024-01-01", "2024-06-30"),
   )

   dataset = RasteretDataset(
       collection=collection,
       bands=["B04", "B03", "B02"],
   )
   sampler = RandomGeoSampler(dataset, size=256, length=100)

How Users Get A Collection
--------------------------

Rasteret starts from a ``Collection`` which helps in images discovery and read
planning based on COG metadata and other properties of image level properties rather
than file-path globbing.

Common entry points are:

1. ``rasteret.build(...)`` for datasets already registered in Rasteret's catalog.
2. ``rasteret.build_from_stac(...)`` for custom STAC APIs.
3. ``rasteret.build_from_table(...)`` for Parquet/GeoParquet tables that already
   contain COG URLs.
4. ``rasteret.load(...)`` for an existing shared or previously-built Rasteret
   collection.

For raw local or cloud COG files without an existing STAC or Parquet record
table, Rasteret's current workflow is to first create a Parquet record table
(``id``, ``datetime``, ``geometry``, ``assets``) and then run
``build_from_table(..., enrich_cog=True)``.

This backend is an experimental entry point for Rasteret-compatible raster
collections, not a transparent backend swap for every existing TorchGeo dataset
subclass.

Mental Model
------------

- Native ``RasterDataset`` subclasses are usually path-first and read via
  rasterio/GDAL.
- ``RasteretDataset`` is collection-first and is a TorchGeo-facing wrapper
  over Rasteret's existing ``collection.to_torchgeo_dataset(...)``
  integration.
- After dataset construction, downstream TorchGeo usage is standard.

TorchGeo Surface That Stays The Same
------------------------------------

After ``RasteretDataset`` is created, normal TorchGeo usage remains the same:

- ``RandomGeoSampler``, ``GridGeoSampler``, and dataloaders
- transforms and collation
- sample dict structure (``image``/``mask``, ``bounds``, ``transform``)
- dataset composition (``&`` / ``|``)

In other words, the difference is primarily **how the dataset is constructed**
and **how pixels are fetched**, not how TorchGeo code downstream interacts with
the dataset object.

Why It Is Different
-------------------

Rasteret is not just a TorchGeo adapter. Its core abstraction is a reusable
``Collection`` that can also serve xarray, NumPy, GeoPandas, and point-sampling
workflows. TorchGeo is one consumer of that collection.

That is why this integration is intentionally collection-first:

- collection preparation remains in Rasteret's own public API
- TorchGeo gains a thin experimental dataset entry point
- TorchGeo does not need to absorb Rasteret's ingest, cloud, or metadata logic

Remaining Differences vs Native RasterDataset
---------------------------------------------

The main remaining differences are:

Rasteret ``0.3.7+`` aligns with native ``RasterDataset`` semantics for
time-series temporal filtering and overlapping-record mosaicking.

1. **Construction model**

   ``RasteretDataset`` requires a Rasteret ``Collection`` object instead of
   filesystem paths.

2. **CRS override timing**

   Pass ``crs=...`` when constructing ``RasteretDataset``. Published
   Rasteret ``0.3.7+`` delegates bind read-time CRS at construction, so
   post-init ``dataset.crs = ...`` is not supported here.

3. **Multi-CRS default behavior**

   If records span multiple CRS zones, Rasteret keeps one EPSG by default and
   drops rows from other EPSG zones unless ``target_crs`` is provided.

4. **Band-resolution handling**

   Requested bands must share resolution unless ``allow_resample=True``.

5. **Metadata contract**

   Rasteret dataset creation depends on collection metadata and per-band
   metadata columns. Invalid rows are skipped, and creation fails if no valid
   rows remain.

6. **Image dtype behavior**

   Rasteret follows TorchGeo's ``array_to_tensor`` integer casting rules
   instead of unconditionally converting imagery to floating point at read
   time.

7. **``label_field``**

   ``label_field`` is an extra Rasteret-specific option that adds
   ``sample["label"]`` from a collection column. TorchGeo's generic sample
   collation path already tolerates extra keys.

AOI vs Sampling Domain
----------------------

Rasteret's ``geometries=...`` argument (passed through to
``collection.to_torchgeo_dataset(geometries=...)``) is a **record filter**: it
controls which tiles/scenes are included in ``dataset.index``.

TorchGeo samplers still sample over the dataset index bounds unless you pass
``roi=...``. For AOI-only sampling, pass ``roi=<AOI polygon in dataset CRS>``
to ``GridGeoSampler`` or ``RandomGeoSampler``.

.. autoclass:: RasteretDataset
