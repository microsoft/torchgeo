torchgeo.models
===============

.. module:: torchgeo.models

This section provides an overview of all models available in ``torchgeo.models``.

Model Architectures
-------------------

.. toctree::
   :maxdepth: 1

   models/aurora
   models/btc
   models/change-star
   models/changevit
   models/conv-lstm
   models/copernicus-fm
   models/croma
   models/dofa
   models/earthloc
   models/farseg
   models/fcn
   models/fc-siamese-networks
   models/l-tae
   models/mosaiks
   models/panopticon
   models/resnet
   models/scale-mae
   models/swin-transformer
   models/u-net
   models/vision-transformer

Utility Functions
-----------------

.. autofunction:: get_model
.. autofunction:: get_model_weights
.. autofunction:: get_weight
.. autofunction:: list_models

Pretrained Weights
------------------

TorchGeo provides a number of pre-trained models and backbones, allowing you to perform transfer learning on small datasets without training a new model from scratch or relying on ImageNet weights. Depending on the satellite/sensor where your data comes from, choose from the following pre-trained weights based on which one has the best performance metrics.

.. contents::
   :local:
   :depth: 2

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


Aerial
^^^^^^

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/aerial.csv


Atmospheric
^^^^^^^^^^^

.. csv-table:: N = Nowcasting, MWF = Medium-Range Weather Forecasting, S2S = Subseasonal to Seasonal, DS = Decadal Scale
   :widths: 45 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/atmospheric.csv

