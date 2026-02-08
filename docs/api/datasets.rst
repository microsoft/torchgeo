torchgeo.datasets
=================

.. module:: torchgeo.datasets

In :mod:`torchgeo`, we define two types of datasets: :ref:`Geospatial Datasets` and :ref:`Non-geospatial Datasets`. These abstract base classes are documented in more detail in :ref:`Base Classes`.

.. _Geospatial Datasets:

Geospatial Datasets
-------------------

:class:`GeoDataset` is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using :class:`IntersectionDataset` and :class:`UnionDataset`.

.. csv-table::
   :widths: 30 15 20 36 20 15
   :header-rows: 1
   :align: center
   :file: datasets/geo_datasets.csv

.. toctree::
   :maxdepth: 1
   :hidden:

   datasets/aboveground-woody-biomass
   datasets/agrifieldnet
   datasets/airphen
   datasets/aster-global-dem
   datasets/canadian-building-footprints
   datasets/chesapeake
   datasets/copernicus-embed
   datasets/global-building-map
   datasets/global-mangrove-distribution
   datasets/google-satellite-embedding
   datasets/cropland-data-layer
   datasets/eddmaps
   datasets/enmap
   datasets/enviroatlas
   datasets/esri2020
   datasets/eu-dem
   datasets/eurocrops
   datasets/gbif
   datasets/globbiomass
   datasets/inaturalist
   datasets/io-bench
   datasets/l7-irish
   datasets/l8-biome
   datasets/landcover-ai-geo
   datasets/landsat
   datasets/mmflood
   datasets/naip
   datasets/nccm
   datasets/nlcd
   datasets/open-buildings
   datasets/openstreetmap
   datasets/presto-embeddings
   datasets/prisma
   datasets/sentinel
   datasets/south-africa-crop-type
   datasets/south-america-soybean
   datasets/tessera-embeddings

.. _Non-geospatial Datasets:

Non-geospatial Datasets
-----------------------

:class:`NonGeoDataset` is designed for datasets that lack geospatial information. These datasets can still be combined using :class:`ConcatDataset <torch.utils.data.ConcatDataset>`.

.. csv-table:: C = classification,  R = regression, S = semantic segmentation, I = instance segmentation, T = time series, CD = change detection, OD = object detection, IC = image captioning
   :widths: 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: datasets/non_geo_datasets.csv

.. toctree::
   :maxdepth: 1
   :hidden:

   datasets/advance
   datasets/benin-cashew-plantations
   datasets/bigearthnet
   datasets/biomassters
   datasets/bright
   datasets/cabuar
   datasets/caffe
   datasets/chabud
   datasets/clay-embeddings
   datasets/cloud-cover-detection
   datasets/copernicus-pretrain
   datasets/cowc
   datasets/cropharvest
   datasets/kenya-crop-type
   datasets/deepglobe-land-cover
   datasets/dfc2022
   datasets/dior
   datasets/digital-typhoon
   datasets/dl4gam
   datasets/dota
   datasets/earth-index-embeddings
   datasets/etci2021
   datasets/eurosat
   datasets/everwatch
   datasets/fair1m
   datasets/fields-of-the-world
   datasets/firerisk
   datasets/forest-damage
   datasets/geonrw
   datasets/gid15
   datasets/hyspecnet11k
   datasets/idtrees
   datasets/inria-aerial-image-labeling
   datasets/landcover-ai
   datasets/levircd
   datasets/levircd-plus
   datasets/loveda
   datasets/major-tom
   datasets/mapinwild
   datasets/mdas
   datasets/million-aid
   datasets/mmearth
   datasets/nasa-marine-debris
   datasets/oscd
   datasets/pastis
   datasets/patternnet
   datasets/potsdam
   datasets/quakeset
   datasets/reforestree
   datasets/resisc45
   datasets/rwanda-field-boundary
   datasets/satlas-pretrain
   datasets/seasonal-contrast
   datasets/seasonet
   datasets/sen12ms
   datasets/skippd
   datasets/skyscript
   datasets/so2sat
   datasets/solar-plants-brazil
   datasets/soda
   datasets/ssl4eo
   datasets/ssl4eo-l-benchmark
   datasets/substation
   datasets/sustainbench-crop-yield
   datasets/treesatai
   datasets/tropical-cyclone
   datasets/uc-merced
   datasets/usavars
   datasets/vaihingen
   datasets/vhr10
   datasets/western-usa-live-fuel-moisture
   datasets/xbd
   datasets/zuericrop

Copernicus-Bench
----------------

Copernicus-Bench is a comprehensive evaluation benchmark with 15 downstream tasks hierarchically organized across preprocessing (e.g., cloud removal), base applications (e.g., land cover classification), and specialized applications (e.g., air quality estimation). This benchmark enables systematic assessment of foundation model performances across various Sentinel missions on different levels of practical applications.

.. csv-table:: C = classification,  R = regression, S = semantic segmentation, T = time series, CD = change detection, E = embedding
   :widths: 5 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: datasets/copernicus_bench.csv

.. toctree::
   :maxdepth: 1

   datasets/copernicus-bench

SpaceNet
--------

The `SpaceNet Dataset <https://spacenet.ai/datasets/>`_ is hosted as an Amazon Web Services (AWS) `Public Dataset <https://registry.opendata.aws/spacenet/>`_. It contains ~67,000 square km of very high-resolution imagery, >11M building footprints, and ~20,000 km of road labels to ensure that there is adequate open source data available for geospatial machine learning research. SpaceNet Challenge Dataset's have a combination of very high resolution satellite imagery and high quality corresponding labels for foundational mapping features such as building footprints or road networks.

.. csv-table:: I = instance segmentation
   :widths: 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: datasets/spacenet.csv

.. toctree::
   :maxdepth: 1

   datasets/spacenet

.. _Base Classes:

Base Classes
------------

If you want to write your own custom dataset, you can extend one of these abstract base classes.

GeoDataset
^^^^^^^^^^

.. autoclass:: GeoDataset

RasterDataset
^^^^^^^^^^^^^

.. autoclass:: RasterDataset

VectorDataset
^^^^^^^^^^^^^

.. autoclass:: VectorDataset

NonGeoDataset
^^^^^^^^^^^^^

.. autoclass:: NonGeoDataset

NonGeoClassificationDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NonGeoClassificationDataset

IntersectionDataset
^^^^^^^^^^^^^^^^^^^

.. autoclass:: IntersectionDataset

UnionDataset
^^^^^^^^^^^^

.. autoclass:: UnionDataset

Utilities
---------

Collation Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: stack_samples
.. autofunction:: concat_samples
.. autofunction:: merge_samples
.. autofunction:: unbind_samples

Splitting Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: random_bbox_assignment
.. autofunction:: random_bbox_splitting
.. autofunction:: random_grid_cell_assignment
.. autofunction:: roi_split
.. autofunction:: time_series_split

Errors
------

.. autoclass:: DatasetNotFoundError
.. autoclass:: DependencyNotFoundError
.. autoclass:: RGBBandsMissingError
