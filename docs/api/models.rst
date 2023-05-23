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

.. autofunction:: resnet18
.. autofunction:: resnet50
.. autoclass:: ResNet18_Weights
.. autoclass:: ResNet50_Weights

.. csv-table::
   :widths: 45 10 10 10 15 10 10 10
   :header-rows: 1
   :align: center
   :file: resnet_pretrained_weights.csv

Swin Transformer
^^^^^^^^^^^^^^^^^^

.. autofunction:: swin_v2_b
.. autoclass:: Swin_V2_B_Weights

.. csv-table::
   :widths: 45 10 10 10 15 10 10 10
   :header-rows: 1
   :align: center
   :file: swin_pretrained_weights.csv

Vision Transformer
^^^^^^^^^^^^^^^^^^

.. autofunction:: vit_small_patch16_224
.. autoclass:: ViTSmall16_Weights

.. csv-table::
   :widths: 45 10 10 10 15 10 10 10
   :header-rows: 1
   :align: center
   :file: vit_pretrained_weights.csv

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: get_model
.. autofunction:: get_model_weights
.. autofunction:: get_weight
.. autofunction:: list_models
