torchgeo.datasets
=================

.. module:: torchgeo.datasets

In :mod:`torchgeo`, we define two types of datasets: :ref:`Geospatial Datasets` and :ref:`Non-geospatial Datasets`. These abstract base classes are documented in more detail in :ref:`Base Classes`.

.. _Geospatial Datasets:

Geospatial Datasets
-------------------

:class:`GeoDataset` is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using :class:`ZipDataset`.

Canadian Building Footprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CanadianBuildingFootprints

Chesapeake Bay High-Resolution Land Cover Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Chesapeake
.. autoclass:: Chesapeake7
.. autoclass:: Chesapeake13
.. autoclass:: ChesapeakeDC
.. autoclass:: ChesapeakeDE
.. autoclass:: ChesapeakeMD
.. autoclass:: ChesapeakeNY
.. autoclass:: ChesapeakePA
.. autoclass:: ChesapeakeVA
.. autoclass:: ChesapeakeWV
.. autoclass:: ChesapeakeCVPR
.. autoclass:: ChesapeakeCVPRDataModule

Cropland Data Layer (CDL)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CDL

Landsat
^^^^^^^

.. autoclass:: Landsat
.. autoclass:: Landsat9
.. autoclass:: Landsat8
.. autoclass:: Landsat7
.. autoclass:: Landsat5TM
.. autoclass:: Landsat5MSS
.. autoclass:: Landsat4TM
.. autoclass:: Landsat4MSS
.. autoclass:: Landsat3
.. autoclass:: Landsat2
.. autoclass:: Landsat1

National Agriculture Imagery Program (NAIP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NAIP
.. autoclass:: NAIPChesapeakeDataModule

Sentinel
^^^^^^^^

.. autoclass:: Sentinel
.. autoclass:: Sentinel2

.. _Non-geospatial Datasets:

Non-geospatial Datasets
-----------------------

:class:`VisionDataset` is designed for datasets that lack geospatial information. These datasets can still be combined using :class:`ConcatDataset <torch.utils.data.ConcatDataset>`.

ADVANCE (AuDio Visual Aerial sceNe reCognition datasEt)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ADVANCE

Smallholder Cashew Plantations in Benin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BeninSmallHolderCashews

BigEarthNet
^^^^^^^^^^^

.. autoclass:: BigEarthNet
.. autoclass:: BigEarthNetDataModule

Cars Overhead With Context (COWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: COWC
.. autoclass:: COWCCounting
.. autoclass:: COWCDetection
.. autoclass:: COWCCountingDataModule

CV4A Kenya Crop Type Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CV4AKenyaCropType

ETCI2021 Flood Detection
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ETCI2021
.. autoclass:: ETCI2021DataModule

EuroSAT
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EuroSAT
.. autoclass:: EuroSATDataModule

GID-15 (Gaofen Image Dataset)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GID15

LandCover.ai (Land Cover from Aerial Imagery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LandCoverAI
.. autoclass:: LandCoverAIDataModule

LEVIR-CD+ (LEVIR Change Detection +)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LEVIRCDPlus

OSCD (Onera Satellite Change Detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OSCD
.. autoclass:: OSCDDataModule

PatternNet
^^^^^^^^^^

.. autoclass:: PatternNet

Potsdam
^^^^^^^

.. autoclass:: Potsdam2D
.. autoclass:: Potsdam2DDataModule

RESISC45 (Remote Sensing Image Scene Classification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RESISC45
.. autoclass:: RESISC45DataModule

Seasonal Contrast
^^^^^^^^^^^^^^^^^

.. autoclass:: SeasonalContrastS2

SEN12MS
^^^^^^^

.. autoclass:: SEN12MS
.. autoclass:: SEN12MSDataModule

So2Sat
^^^^^^

.. autoclass:: So2Sat
.. autoclass:: So2SatDataModule

SpaceNet
^^^^^^^^

.. autoclass:: SpaceNet
.. autoclass:: SpaceNet1
.. autoclass:: SpaceNet2
.. autoclass:: SpaceNet4
.. autoclass:: SpaceNet5
.. autoclass:: SpaceNet7

Tropical Cyclone Wind Estimation Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TropicalCycloneWindEstimation
.. autoclass:: CycloneDataModule

Vaihingen
^^^^^^^^^

.. autoclass:: Vaihingen2D
.. autoclass:: Vaihingen2DDataModule

NWPU VHR-10
^^^^^^^^^^^

.. autoclass:: VHR10

UC Merced
^^^^^^^^^

.. autoclass:: UCMerced
.. autoclass:: UCMercedDataModule

xView2
^^^^^^

.. autoclass:: XView2
.. autoclass:: XView2DataModule

ZueriCrop
^^^^^^^^^

.. autoclass:: ZueriCrop

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

VisionDataset
^^^^^^^^^^^^^

.. autoclass:: VisionDataset

VisionClassificationDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: VisionClassificationDataset

ZipDataset
^^^^^^^^^^

.. autoclass:: ZipDataset

Utilities
---------

.. autoclass:: BoundingBox
.. autofunction:: collate_dict
