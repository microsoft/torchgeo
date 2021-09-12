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

Sentinel
^^^^^^^^

.. autoclass:: Sentinel
.. autoclass:: Sentinel2

Spacenet 1: Building Detection v1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Spacenet1

.. _Non-geospatial Datasets:

Non-geospatial Datasets
-----------------------

:class:`VisionDataset` is designed for datasets that lack geospatial information. These datasets can still be combined using :class:`ConcatDataset <torch.utils.data.ConcatDataset>`.

Smallholder Cashew Plantations in Benin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BeninSmallHolderCashews

Cars Overhead With Context (COWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: COWC
.. autoclass:: COWCCounting
.. autoclass:: COWCDetection

CV4A Kenya Crop Type Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CV4AKenyaCropType

ETCI2021 Flood Detection
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ETCI2021

GID-15 (Gaofen Image Dataset)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GID15

LandCover.ai (Land Cover from Aerial Imagery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LandCoverAI

LEVIR-CD+ (LEVIR Change Detection +)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LEVIRCDPlus

PatternNet
^^^^^^^^^^

.. autoclass:: PatternNet

RESISC45 (Remote Sensing Image Scene Classification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RESISC45

SEN12MS
^^^^^^^

.. autoclass:: SEN12MS

So2Sat
^^^^^^

.. autoclass:: So2Sat

Tropical Cyclone Wind Estimation Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TropicalCycloneWindEstimation

NWPU VHR-10
^^^^^^^^^^^

.. autoclass:: VHR10

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

ZipDataset
^^^^^^^^^^

.. autoclass:: ZipDataset

Utilities
---------

.. autoclass:: BoundingBox
.. autofunction:: collate_dict
