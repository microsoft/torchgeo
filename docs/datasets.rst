torchgeo.datasets
=================

.. module:: torchgeo.datasets

In :mod:`torchgeo`, we define two types of datasets: :ref:`Geospatial Datasets` and :ref:`Non-geospatial Datasets`. These abstract base classes are documented in more detail in :ref:`Base Classes`.

Geospatial Datasets
-------------------

:class:`GeoDataset` is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using :class:`ZipDataset`.

Cropland Data Layer (CDL)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CDL

Landsat
^^^^^^^

.. autoclass:: Landsat
.. autoclass:: Landsat8_9
.. autoclass:: Landsat7
.. autoclass:: Landsat4_5TM
.. autoclass:: Landsat4_5MSS
.. autoclass:: Landsat1_3

Sentinel
^^^^^^^^

.. autoclass:: Sentinel
.. autoclass:: Sentinel2

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

LandCover.ai (Land Cover from Aerial Imagery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LandCoverAI

SEN12MS
^^^^^^^

.. autoclass:: SEN12MS

Tropical Cyclone Wind Estimation Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TropicalCycloneWindEstimation

NWPU VHR-10
^^^^^^^^^^^

.. autoclass:: VHR10

Base Classes
------------

If you want to write your own custom dataset, you can extend one of these abstract base classes.

GeoDataset
^^^^^^^^^^

.. autoclass:: GeoDataset

VisionDataset
^^^^^^^^^^^^^

.. autoclass:: VisionDataset

ZipDataset
^^^^^^^^^^

.. autoclass:: ZipDataset

Utilities
---------

.. autoclass:: BoundingBox
