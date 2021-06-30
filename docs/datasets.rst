torchgeo.datasets
=================

.. module:: torchgeo.datasets

In :mod:`torchgeo`, we define two types of datasets: :ref:`Geospatial Datasets` and :ref:`Non-geospatial Datasets`. These abstract base classes are documented in more detail in :ref:`Base Classes`.

Geospatial Datasets
-------------------

:class:`GeoDataset` is designed for datasets that contain geospatial information, like latitude, longitude, coordinate system, and projection. Datasets containing this kind of information can be combined using :class:`ZipDataset`.

Smallholder Cashew Plantations in Benin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: BeninSmallHolderCashews

CV4A Kenya Crop Type Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CV4AKenyaCropType

SEN12MS
^^^^^^^

.. autoclass:: SEN12MS

Non-geospatial Datasets
-----------------------

:class:`VisionDataset` is designed for datasets that lack geospatial information. These datasets can still be combined using :class:`ConcatDataset <torch.utils.data.ConcatDataset>`.

Cars Overhead With Context (COWC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: COWC
.. autoclass:: COWCCounting
.. autoclass:: COWCDetection

Tropical Cyclone Wind Estimation Competition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TropicalCycloneWindEstimation

LandCover.ai (Land Cover from Aerial Imagery)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LandCoverAI

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
