torchgeo.models
===============

.. module:: torchgeo.models

This section provides an overview and links to the documentation for all models available in ``torchgeo.models``.

ChangeStar models
-----------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   models/changestar
   models/changestarfarseg
   models/changemixin


ResNet models
-------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   models/resnet18
   models/resnet50


Pretrained Weights
------------------

TorchGeo provides a number of pre-trained models and backbones, allowing you to perform transfer learning on small datasets without training a new model from scratch or relying on ImageNet weights. Depending on the satellite/sensor where your data comes from, choose from the following pre-trained weights based on which one has the best performance metrics.

Sensor-Agnostic
^^^^^^^^^^^^^^^

These weights can be used with imagery from any satellite/sensor. In addition to the usual performance metrics, there are also additional columns for dynamic spatial (resolution), temporal (time span), and/or spectral (wavelength) support, either via their training data (implicit) or via their model architecture (explicit).

.. csv-table::
   :widths: 45 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/agnostic.csv


Landsat
^^^^^^^

.. csv-table::
   :widths: 65 10 10 10 10 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/landsat.csv


NAIP
^^^^

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/naip.csv


Sentinel-1
^^^^^^^^^^

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/sentinel1.csv


Sentinel-2
^^^^^^^^^^

.. csv-table::
   :widths: 45 10 10 10 10 15 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/sentinel2.csv

