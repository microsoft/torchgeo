torchgeo.trainers
=================

.. module:: torchgeo.trainers

.. toctree::
   :maxdepth: 0
   :hidden:
   :glob:

   trainers/*

TorchGeo provides `LightningModules <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`__ for a number of common tasks in geospatial and geotemporal deep learning.

Supervised Learning
-------------------

Supervised learning tasks have both inputs and labeled outputs.

.. list-table:: Supervised Learning Tasks
   :header-rows: 1

   * - Input
     - Output
     - Task
   * - :math:`\mathbb{R}^{T \times C}`
     - :math:`\mathbb{R}^{T \times C}`
     - :ref:`TemporalRegressionTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{Z}` or :math:`\mathbb{Z}^C`
     - :ref:`ClassificationTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{R}` or :math:`\mathbb{R}^C`
     - :ref:`RegressionTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{Z}^{H \times W}`
     - :ref:`SemanticSegmentationTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{R}^{H \times W}`
     - :ref:`PixelwiseRegressionTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{R}^{O \times 4}`
     - :ref:`ObjectDetectionTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{Z}^{O \times H \times W}`
     - :ref:`InstanceSegmentationTask`
   * - :math:`\mathbb{R}^{2 \times C \times H \times W}`
     - :math:`\mathbb{Z}^{H \times W}`
     - :ref:`ChangeDetectionTask`
   * - :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :math:`\mathbb{Z}^{H \times W}`
     - :ref:`SpatioTemporalSegmentationTask`


Self-Supervised Learning
------------------------

Self-supervised learning (SSL) tasks have inputs and create their own labeled outputs.

.. list-table:: Self-Supervised Learning Tasks
   :header-rows: 1

   * - Input
     - Task
   * - :math:`\mathbb{R}^{C \times H \times W}` or :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :ref:`BYOLTask`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :ref:`MAETask`
   * - :math:`\mathbb{R}^{C \times H \times W}` or :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :ref:`MoCoTask`
   * - :math:`\mathbb{R}^{C \times H \times W}` or :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :ref:`SimCLRTask`

Non-Learning Tasks
------------------

Tasks that do not relate to learning.

.. list-table:: Non-Learning Tasks
   :header-rows: 1

   * - Input
     - Task
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :ref:`IOBenchTask`

Base Classes
------------

Abstract base classes that all other tasks inherit from.

.. list-table:: Base Classes
   :header-rows: 1

   * - Task
   * - :ref:`BaseTask`

Mixins
-------

`Mixins <https://en.wikipedia.org/wiki/Mixin>`__ that support code reuse across multiple tasks.

.. list-table:: Mixins
   :header-rows: 1

   * - Output
     - Mixin
   * - :math:`\mathbb{Z}` or :math:`\mathbb{Z}^C` or :math:`\mathbb{Z}^{H \times W}`
     - :ref:`ClassificationMixin`
   * - :math:`\mathbb{R}` or :math:`\mathbb{R}^C` or :math:`\mathbb{R}^{T \times C}` or :math:`\mathbb{R}^{H \times W}`
     - :ref:`RegressionMixin`
