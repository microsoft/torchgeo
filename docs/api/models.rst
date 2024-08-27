torchgeo.models
=================

.. module:: torchgeo.models

Change Star
^^^^^^^^^^^

.. autoclass:: ChangeStar
.. autoclass:: ChangeStarFarSeg
.. autoclass:: ChangeMixin

DOFA
^^^^

.. autoclass:: DOFA
.. autofunction:: dofa_small_patch16_224
.. autofunction:: dofa_base_patch16_224
.. autofunction:: dofa_large_patch16_224
.. autofunction:: dofa_huge_patch16_224
.. autoclass:: DOFABase16_Weights
.. autoclass:: DOFALarge16_Weights

FarSeg
^^^^^^

.. autoclass:: FarSeg

Fully-convolutional Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: FCN

FC Siamese Networks
^^^^^^^^^^^^^^^^^^^

.. autoclass:: FCSiamConc
.. autoclass:: FCSiamDiff

RCF Extractor
^^^^^^^^^^^^^

.. autoclass:: RCF

ResNet
^^^^^^

.. autofunction:: resnet18
.. autofunction:: resnet50
.. autofunction:: resnet152
.. autoclass:: ResNet18_Weights
.. autoclass:: ResNet50_Weights
.. autoclass:: ResNet152_Weights

Scale-MAE
^^^^^^^^^

.. autofunction:: ScaleMAE
.. autoclass:: ScaleMAELarge16_Weights

Swin Transformer
^^^^^^^^^^^^^^^^^^

.. autofunction:: swin_v2_t
.. autofunction:: swin_v2_b
.. autoclass:: Swin_V2_T_Weights
.. autoclass:: Swin_V2_B_Weights

Vision Transformer
^^^^^^^^^^^^^^^^^^

.. autofunction:: vit_small_patch16_224
.. autoclass:: ViTSmall16_Weights

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: get_model
.. autofunction:: get_model_weights
.. autofunction:: get_weight
.. autofunction:: list_models


Pretrained Weights
^^^^^^^^^^^^^^^^^^

TorchGeo provides a number of pre-trained models and backbones, allowing you to perform transfer learning on small datasets without training a new model from scratch or relying on ImageNet weights. Depending on the satellite/sensor where your data comes from, choose from the following pre-trained weights based on which one has the best performance metrics.

Sensor-Agnostic
---------------

These weights can be used with imagery from any satellite/sensor. In addition to the usual performance metrics, there are also additional columns for dynamic spatial (resolution), temporal (time span), and/or spectral (wavelength) support, either via their training data (implicit) or via their model architecture (explicit).

.. csv-table::
   :widths: 45 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/agnostic.csv


Landsat
-------

.. csv-table::
   :widths: 65 10 10 10 10 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/landsat.csv


NAIP
----

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/naip.csv


Sentinel-1
----------

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/sentinel1.csv


Sentinel-2
----------

.. csv-table::
   :widths: 45 10 10 10 10 15 10 10 10
   :header-rows: 1
   :align: center
   :file: weights/sentinel2.csv
