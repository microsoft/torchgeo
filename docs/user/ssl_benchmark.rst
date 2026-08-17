Self-Supervised Learning Tasks
==============================

TorchGeo ships several self-supervised learning (SSL) tasks. Because SSL has no labels, a training loss says very little about whether a task works: the losses are on different scales and measure different things, and a loss that falls steadily is entirely compatible with a representation that has collapsed. It is therefore non-trivial to determine whether an SSL training run is working at all without probing the learned representations over a dataset. A run that is silently producing a degenerate encoder could look, from the loss curve alone, much like a run that is working well. This matters when contributing a new SSL task to TorchGeo: a new task that trains without error, and whose loss decreases, has not yet been shown to do anything useful.

This page describes the tasks TorchGeo provides, defines a small benchmark for comparing them, and records the current numbers so that a new task can be judged against them on equal terms.

.. contents::
   :local:
   :depth: 1

Available tasks
---------------

All SSL tasks subclass :class:`~torchgeo.tasks.BaseTask` and take a ``model`` argument naming any `timm <https://huggingface.co/docs/timm/reference/models>`__ encoder, plus ``in_channels`` so they can be used with multispectral imagery.

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - Task
     - Approach
     - Notes
   * - :class:`~torchgeo.tasks.SimCLR`
     - Contrastive. Pulls two augmented views of the same image together and pushes apart views of different images, using NT-Xent over the batch.
     - ``version`` selects SimCLR v1 or v2. Needs large batches, since negatives come from within the batch.
   * - :class:`~torchgeo.tasks.MoCo`
     - Contrastive with a momentum-updated target encoder, so negatives can come from a queue rather than the current batch.
     - ``version`` selects MoCo v1, v2, or v3. v3 drops the queue and uses a predictor head.
   * - :class:`~torchgeo.tasks.BYOL`
     - Non-contrastive. Predicts the target network's projection of one view from the online network's projection of another, with no negatives.
     - Takes no ``size`` or augmentation argument; the input resolution is fixed at 224x224 internally.
   * - :class:`~torchgeo.tasks.MAE`
     - Generative. Masks most patches and reconstructs them.
     - Vision transformers only, since it operates on patch tokens.


Benchmarking
------------

Our evaluation follows `Corley et al. 2024, "Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters" <https://arxiv.org/abs/2305.13456>`_: pretrain on EuroSAT 13 band multispectral without labels, freeze the encoder, and score the frozen features with a k-nearest-neighbour classifier. kNN is used rather than a linear probe because it has no optimizer, no learning rate, and no regularization of its own, so it measures the representation rather than the tuning of the probe.

Hold all of the following fixed. Changing any of them makes a number incomparable to the table below.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Setting
     - Value
   * - Dataset
     - EuroSAT, all 13 Sentinel-2 bands, TorchGeo splits (16,200 train / 5,400 val / 5,400 test)
   * - Normalization
     - Per-band standardization using ``MEAN`` and ``STD`` from ``torchgeo.datamodules.eurosat``, which is what :class:`~torchgeo.datamodules.EuroSATDataModule` applies by default
   * - Input size
     - 224x224, produced by each task's ``RandomResizedCrop`` from the native 64x64 imagery
   * - Pretraining
     - 60 epochs, batch size 128, mixed precision, one GPU, seed 0
   * - Features
     - ``forward_head(forward_features(x), pre_logits=True)`` on the frozen encoder, evaluated on unaugmented images
   * - Probe
     - ``sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)``, Euclidean
   * - Scaling
     - Fit the probe on raw features and on ``StandardScaler`` features, and report whichever is better, as the reference paper does
   * - Selection
     - Choose the learning rate on validation accuracy, then read the test set once

Resizing to 224x224 matters more than it looks. The reference paper's central finding is that evaluating at the native 64x64 rather than 224x224 changes the ranking of pretrained models, and normalization has a comparable effect: in our runs, preprocessing alone moved a single ImageNet ResNet-50 across a 0.43 range of kNN accuracy.

Results
-------

EuroSAT test-set kNN-5 top-1 accuracy for the reference baselines. None of these involve SSL pretraining; they exist to bound what a new SSL task has to beat before it can be said to have learned anything.

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - Baseline
     - Encoder
     - Test acc
   * - Image statistics [#floor]_
     - none
     - 0.8937
   * - Random initialization
     - ResNet-50
     - 0.8622
   * - Random initialization
     - ViT-S/16
     - 0.8507
   * - Supervised ImageNet [#imagenet]_
     - ResNet-50
     - 0.8948
   * - Supervised ImageNet [#imagenet]_
     - ViT-S/16
     - 0.9178

Numbers for the SSL tasks themselves are deliberately omitted until they can be reproduced under this protocol, and will be added here as they are.

Reading the table:

* **Beat the floors, not just random init.** 52 hand-computed per-band image statistics, with no network at all, score 0.8937. A task that lands below that has not learned anything a mean and a standard deviation do not already capture. Supervised ImageNet transfer on a ViT is a harder floor at 0.9178.
* **Differences of a few thousandths are not meaningful.** Every number comes from a single seed, so treat gaps below roughly 0.005 as noise.

.. rubric:: Footnotes

.. [#floor] Per-band mean, standard deviation, minimum, and maximum of each image, concatenated into a 52-dimensional vector and fed to the same kNN probe. No network and no training.

.. [#imagenet] These encoders are ImageNet-pretrained but were built with ``in_chans=13``, so timm adapts the pretrained 3-channel stem rather than reinitializing it: ``timm.models.adapt_input_conv`` tiles the RGB filters ``ceil(13 / 3) = 5`` times, truncates to 13 channels, and rescales by ``3 / 13`` to preserve the activation magnitude. The remaining layers are the unmodified ImageNet weights.

Running the benchmark
---------------------

Pretraining runs from a configuration file using the shipped tasks and :class:`~torchgeo.datamodules.EuroSATDataModule`, whose default normalization is exactly the per-band standardization this protocol requires. For example, MoCo v3 on a ResNet-50:

.. code-block:: yaml

   seed_everything: 0
   trainer:
     accelerator: gpu
     devices: 1
     max_epochs: 60
     precision: 16-mixed
     benchmark: true
   model:
     class_path: MoCo
     init_args:
       model: resnet50
       in_channels: 13
       version: 3
       lr: 0.01
       size: 224
   data:
     class_path: EuroSATDataModule
     init_args:
       batch_size: 128
       num_workers: 8
     dict_kwargs:
       root: data/eurosat

.. code-block:: console

   $ python -m torchgeo fit --config moco_resnet50.yaml

The other tasks differ only in ``model.init_args``:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Configuration
     - ``model.init_args``
   * - ``moco_vit_small``
     - ``class_path: MoCo``, ``model: vit_small_patch16_224``, ``version: 3``, ``lr: 0.001``, ``size: 224``
   * - ``simclr_resnet50``
     - ``class_path: SimCLR``, ``model: resnet50``, ``version: 2``, ``lr: 1.5``, ``size: 224``, ``memory_bank_size: 0``
   * - ``simclr_vit_small``
     - ``class_path: SimCLR``, ``model: vit_small_patch16_224``, ``version: 2``, ``lr: 1.5``, ``size: 224``, ``memory_bank_size: 0``
   * - ``mae_vit_small``
     - ``class_path: MAE``, ``model: vit_small_patch16_224``, ``lr: 7.5e-05``, ``size: 224``, ``warmup_epochs: 10``

All configurations use ``in_channels: 13`` and the same ``trainer`` and ``data`` blocks as above. The learning rates shown are starting points from exploratory runs rather than validated results, so sweep them rather than trusting them.

Scoring is not part of ``torchgeo fit``: training writes a checkpoint, and the kNN probe is applied afterwards. Given a frozen encoder, the whole probe is:

.. code-block:: python

   import torch
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.preprocessing import StandardScaler


   @torch.no_grad()
   def features(backbone, loader, device):
       backbone.eval().to(device)
       out, targets = [], []
       for batch in loader:
           x = batch['image'].to(device)
           z = backbone.forward_head(backbone.forward_features(x), pre_logits=True)
           out.append(z.flatten(1).cpu())
           targets.append(batch['label'])
       return torch.cat(out).numpy(), torch.cat(targets).numpy()


   def knn_score(train, train_y, test, test_y):
       scores = []
       for scaler in (None, StandardScaler()):
           a, b = (train, test) if scaler is None else (
               scaler.fit_transform(train), scaler.transform(test)
           )
           probe = KNeighborsClassifier(n_neighbors=5).fit(a, train_y)
           scores.append(probe.score(b, test_y))
       return max(scores)

Features are extracted from unaugmented images, so pass the datamodule's validation or test loader rather than the training loader.

Adding a new task
-----------------

A new SSL task should arrive with the same pieces as any other TorchGeo task:

#. ``torchgeo/tasks/foo.py``, subclassing :class:`~torchgeo.tasks.BaseTask`.
#. An entry in ``torchgeo/tasks/__init__.py``.
#. Tests: TODO
#. ``docs/api/tasks.rst``.

Before claiming it works, sweep at least three or four learning rates and evaluate these checkpoints on the EuroSAT val set using the methodology described above. Do not borrow a library default: the shipped defaults follow the linear scaling rule at batch 4096, so MoCo v3's ``lr=9.6`` is ``0.6 x 4096 / 256`` and SimCLR's ``lr=4.8`` is ``0.3 x 4096 / 256``, both far too large at the batch size of 128 used here. Evaluate the best on additionally on the EuroSAT test split and compare the published numbers above. If it doesn't beat image-statistics, then something is likely wrong. Check that the representation has not collapsed: the standard deviation of the embeddings should stay well away from zero, and the mean pairwise cosine similarity well away from one. Report all of this in your PR!