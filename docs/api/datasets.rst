torchgeo.datasets
=================

.. module:: torchgeo.datasets

In :mod:`torchgeo`, we define two types of datasets: :ref:`Geospatial Datasets` and :ref:`Non-geospatial Datasets`. These abstract base classes are documented in more detail in :ref:`Base Classes`.

.. _Geospatial Datasets:

Geospatial Datasets
-------------------

:class:`GeoDataset` is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using :class:`IntersectionDataset` and :class:`UnionDataset`.

Aboveground Live Woody Biomass Density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AbovegroundLiveWoodyBiomassDensity

Aster Global Digital Evaluation Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AsterGDEM

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

CMS Global Mangrove Canopy Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CMSGlobalMangroveCanopy

Cropland Data Layer (CDL)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CDL

EDDMapS
^^^^^^^

.. autoclass:: EDDMapS

EnviroAtlas
^^^^^^^^^^^

.. autoclass:: EnviroAtlas

Esri2020
^^^^^^^^

.. autoclass:: Esri2020

EU-DEM
^^^^^^

.. autoclass:: EUDEM

GBIF
^^^^

.. autoclass:: GBIF

GlobBiomass
^^^^^^^^^^^

.. autoclass:: GlobBiomass

iNaturalist
^^^^^^^^^^^

.. autoclass:: INaturalist

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

Open Buildings
^^^^^^^^^^^^^^

.. autoclass:: OpenBuildings

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

Cars Overhead With Context (COWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: COWC
.. autoclass:: COWCCounting
.. autoclass:: COWCDetection

CV4A Kenya Crop Type Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CV4AKenyaCropType

2022 IEEE GRSS Data Fusion Contest (DFC2022)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DFC2022

ETCI2021 Flood Detection
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ETCI2021

EuroSAT
^^^^^^^

.. autoclass:: EuroSAT

FAIR1M (Fine-grAined object recognItion in high-Resolution imagery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FAIR1M

Forest Damage
^^^^^^^^^^^^^

.. autoclass:: ForestDamage

GID-15 (Gaofen Image Dataset)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GID15

IDTReeS
^^^^^^^

.. autoclass:: IDTReeS

Inria Aerial Image Labeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: InriaAerialImageLabeling

LandCover.ai (Land Cover from Aerial Imagery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LandCoverAI

LEVIR-CD+ (LEVIR Change Detection +)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LEVIRCDPlus

LoveDA (Land-cOVEr Domain Adaptive semantic segmentation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LoveDA

NASA Marine Debris
^^^^^^^^^^^^^^^^^^

.. autoclass:: NASAMarineDebris

OSCD (Onera Satellite Change Detection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: OSCD

PatternNet
^^^^^^^^^^

.. autoclass:: PatternNet

Potsdam
^^^^^^^

.. autoclass:: Potsdam2D

RESISC45 (Remote Sensing Image Scene Classification)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RESISC45

Seasonal Contrast
^^^^^^^^^^^^^^^^^

.. autoclass:: SeasonalContrastS2

SEN12MS
^^^^^^^

.. autoclass:: SEN12MS

So2Sat
^^^^^^

.. autoclass:: So2Sat

SpaceNet
^^^^^^^^

.. autoclass:: SpaceNet
.. autoclass:: SpaceNet1
.. autoclass:: SpaceNet2
.. autoclass:: SpaceNet3
.. autoclass:: SpaceNet4
.. autoclass:: SpaceNet5
.. autoclass:: SpaceNet7

Tropical Cyclone Wind Estimation Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TropicalCycloneWindEstimation

UC Merced
^^^^^^^^^

.. autoclass:: UCMerced

USAVars
^^^^^^^

.. autoclass:: USAVars

Vaihingen
^^^^^^^^^

.. autoclass:: Vaihingen2D

NWPU VHR-10
^^^^^^^^^^^

.. autoclass:: VHR10

xView2
^^^^^^

.. autoclass:: XView2

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

IntersectionDataset
^^^^^^^^^^^^^^^^^^^

.. autoclass:: IntersectionDataset

UnionDataset
^^^^^^^^^^^^

.. autoclass:: UnionDataset

Utilities
---------

.. autoclass:: BoundingBox

Collation Functions
^^^^^^^^^^^^^^^^^^^

.. autofunction:: stack_samples
.. autofunction:: concat_samples
.. autofunction:: merge_samples
.. autofunction:: unbind_samples
