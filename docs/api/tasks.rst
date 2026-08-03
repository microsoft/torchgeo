torchgeo.tasks
=================

.. module:: torchgeo.tasks

.. toctree::
   :maxdepth: 0
   :hidden:
   :glob:

   tasks/*

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
     - :ref:`TemporalRegression`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{N}` or :math:`\mathbb{N}^C`
     - :ref:`Classification`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{R}` or :math:`\mathbb{R}^C`
     - :ref:`Regression`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{N}^{H \times W}`
     - :ref:`SemanticSegmentation`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{R}^{H \times W}`
     - :ref:`PixelwiseRegression`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{R}^{O \times 4}`
     - :ref:`ObjectDetection`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :math:`\mathbb{N}^{O \times H \times W}`
     - :ref:`InstanceSegmentation`
   * - :math:`\mathbb{R}^{2 \times C \times H \times W}`
     - :math:`\mathbb{N}^{H \times W}`
     - :ref:`ChangeDetection`
   * - :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :math:`\mathbb{N}^{H \times W}`
     - :ref:`SpatioTemporalSegmentation`


Self-Supervised Learning
------------------------

Self-supervised learning (SSL) tasks have inputs and create their own labeled outputs.

.. list-table:: Self-Supervised Learning Tasks
   :header-rows: 1

   * - Input
     - Task
   * - :math:`\mathbb{R}^{C \times H \times W}` or :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :ref:`BYOL`
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :ref:`MAE`
   * - :math:`\mathbb{R}^{C \times H \times W}` or :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :ref:`MoCo`
   * - :math:`\mathbb{R}^{C \times H \times W}` or :math:`\mathbb{R}^{T \times C \times H \times W}`
     - :ref:`SimCLR`

Non-Learning Tasks
------------------

Tasks that do not relate to learning.

.. list-table:: Non-Learning Tasks
   :header-rows: 1

   * - Input
     - Task
   * - :math:`\mathbb{R}^{C \times H \times W}`
     - :ref:`IOBench`

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
   * - :math:`\mathbb{N}` or :math:`\mathbb{N}^C` or :math:`\mathbb{N}^{H \times W}`
     - :ref:`ClassificationMixin`
   * - :math:`\mathbb{R}` or :math:`\mathbb{R}^C` or :math:`\mathbb{R}^{T \times C}` or :math:`\mathbb{R}^{H \times W}`
     - :ref:`RegressionMixin`
