torchgeo.models
=================

.. module:: torchgeo.models

Change Star
^^^^^^^^^^^

.. autoclass:: ChangeStar
.. autoclass:: ChangeStarFarSeg
.. autoclass:: ChangeMixin

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

.. csv-table::
   :widths: 40 10 10 10 15 15
   :header-rows: 1
   :align: center
   :file: resnet_pretrained_weights.csv

.. autofunction:: resnet18
.. autofunction:: resnet50
.. autoclass:: ResNet18_Weights
.. autoclass:: ResNet50_Weights

Vision Transformer
^^^^^^^^^^^^^^^^^^

.. csv-table::
   :widths: 40 10 10 10 15 15
   :header-rows: 1
   :align: center
   :file: vit_pretrained_weights.csv

.. autofunction:: vit_small_patch16_224
.. autoclass:: VITSmall16_Weights
