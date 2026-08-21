Glossary
========

A collection of common terms used in :mod:`torchgeo` that may be unfamiliar to either:

1. Deep learning researchers who don't know remote sensing
2. Remote sensing researchers who don't know deep learning

|


.. glossary::

   area of interest (AOI)
       Synonym for :term:`region of interest (ROI)`. A particular spatial area to focus on.

   chip
       Synonym for :term:`patch`. A smaller image sampled from a larger :term:`tile`.

   classification
       A computer vision task that involves predicting the image class for an entire image or a specific bounding box.

   coordinate reference system (CRS)
       Synonym for :term:`spatial reference system (SRS)`. A system that defines how to locate geographic entities on a :term:`projected <projection>` surface.

   index
       The lookup table of a geospatial dataset (``GeoDataset.index``): a mapping from each file's spatiotemporal footprint to its path, used to find which files a :term:`query` overlaps. It is metadata, not pixel data. Unrelated to a :term:`spectral index`; and note that the ``index`` argument of ``__getitem__`` is, despite its name, a :term:`query`.

   instance segmentation
       A computer vision task that involves predicting labels for each pixel in an image such that each object has a unique label.

   object detection
       A computer vision task that involves predicting bounding boxes around each object in an image.

   patch
       Synonym for :term:`chip`. A smaller image sampled from a larger :term:`tile`.

   projection
       A geometric transformation for portraying the surface of a 3D Earth onto a 2D planar image.

   query
       The spatiotemporal bounding box passed to a geospatial dataset's ``__getitem__`` (a ``GeoSlice``), describing *where and when to sample*. Distinct from the dataset's :term:`index`, which is *how* files are found. When a query is read in a file's native :term:`coordinate reference system (CRS)`, its box may align to a different grid than the index.

   region of interest (ROI)
       Synonym for :term:`area of interest (AOI)`. A particular spatial region to focus on.

   regression
       A computer vision task that involves predicting a real valued number based on an image.

   semantic segmentation
       A computer vision task that involves predicting labels for each pixel in an image such that each class has a unique label.

   spatial reference system (SRS)
       Synonym for :term:`coordinate reference system (CRS)`. A system that defines how to locate geographic entities on a :term:`projected <projection>` surface.

   spectral index
       A per-pixel combination of spectral bands that highlights a phenomenon, such as NDVI (vegetation) or NDBI (built-up area). Unrelated to a dataset's :term:`index`.

   stitching
       Combining a collection of :term:`patches <patch>` into a single image. This is the reverse operation of :term:`tiling`.

   swath
       A set of :term:`tiles <tile>` along a satellite trajectory.

   tile
       A single image file taken by a remote sensor like a satellite.

   tiling
       Splitting a :term:`tile` into :term:`patches <patch>`. This is the reverse operation of :term:`stitching`.
