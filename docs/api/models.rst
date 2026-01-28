torchgeo.models
===============

.. module:: torchgeo.models

This section provides an overview of all models available in ``torchgeo.models``.

Model Architectures
-------------------

TorchGeo contains a number of model architectures depending on the task you are trying to solve and your model inputs.

1D Time Series (:math:`\scriptstyle B \times T \times C`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   models/l-tae
   models/tessera

2D Images (:math:`\scriptstyle B \times C \times H \times W`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   models/copernicus-fm
   models/croma
   models/dofa
   models/earthloc
   models/farseg
   models/fcn
   models/mosaiks
   models/panopticon
   models/resnet
   models/scale-mae
   models/swin-transformer
   models/tilenet
   models/u-net
   models/vision-transformer

TorchGeo also supports most `timm <https://huggingface.co/docs/timm/en/index>`__ encoders and `SMP <https://segmentation-modelspytorch.readthedocs.io/en/latest/>`__ decoders.

3D Change Detection (:math:`\scriptstyle B \times 2 \times C \times H \times W`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   models/btc
   models/changestar
   models/changevit
   models/fc-siamese-networks

See `torchange <https://github.com/Z-Zheng/pytorch-change-models>`__ for additional change detection architectures.

3D Image Time Series (:math:`\scriptstyle B \times T \times C \times H \times W`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   models/convlstm

4D Ocean and Atmosphere (:math:`\scriptstyle B \times T \times C \times Z \times Y \times X`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   models/aurora

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

