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
.. autoclass:: ResNet18_Weights
.. autoclass:: ResNet50_Weights

Scale-MAE
^^^^^^^^^

.. autofunction:: ScaleMAE
.. autoclass:: ScaleMAELarge16_Weights

Swin Transformer
^^^^^^^^^^^^^^^^^^

.. autofunction:: swin_v2_b
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

Sensor-Agnostic
---------------

These weights can be used with imagery from any satellite/sensor.

.. csv-table::
   :widths: 45 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: agnostic_pretrained_weights.csv


NAIP
----

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: naip_pretrained_weights.csv


Landsat
-------

.. csv-table::
   :widths: 65 10 10 10 10 10 10 10 10 10
   :header-rows: 1
   :align: center
   :file: landsat_pretrained_weights.csv


Sentinel-1
----------

.. csv-table::
   :widths: 45 10 10 10 10
   :header-rows: 1
   :align: center
   :file: sentinel1_pretrained_weights.csv


Sentinel-2
----------

.. csv-table::
   :widths: 45 10 10 10 10 15 10 10 10
   :header-rows: 1
   :align: center
   :file: sentinel2_pretrained_weights.csv

Other Data Sources
------------------

.. csv-table::
   :widths: 45 10 10 10 1
   :header-rows: 1
   :align: center
   :file: misc_pretrained_weights.csv
