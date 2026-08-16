Self-Supervised Learning Trainers
=================================

TorchGeo ships several self-supervised learning (SSL) trainers. Because SSL has no labels, a training loss says very little about whether a trainer works: the losses are on different scales and measure different things, and a loss that falls steadily is entirely compatible with a representation that has collapsed. In one run during this benchmark, SimCLR's loss fell by 30% while every sample in the batch mapped to an identical vector.

It is therefore non-trivial to determine whether an SSL training run is working at all without probing the learned representations over a dataset. A run that is silently producing a degenerate encoder looks, from the loss curve alone, much like a run that is working well. This matters most when contributing a new SSL task to TorchGeo: a new trainer that trains without error, and whose loss decreases, has not yet been shown to do anything useful.

This page describes the trainers TorchGeo provides, defines a small benchmark for comparing them, and records the current numbers so that a new trainer can be judged against them on equal terms.

.. contents::
   :local:
   :depth: 1

Available trainers
------------------

All SSL trainers subclass :class:`~torchgeo.tasks.BaseTask` and take a ``model`` argument naming any `timm <https://huggingface.co/docs/timm/reference/models>`__ encoder, plus ``in_channels`` so they can be used with multispectral imagery.

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - Trainer
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

Each trainer builds its own augmentation pipeline, and the argument used to override it differs: ``augmentations`` for SimCLR, ``augmentation1`` and ``augmentation2`` for MoCo, ``transform`` for MAE, and nothing at all for BYOL. Passing the wrong name is silently ignored, so verify that a custom pipeline is actually installed before attributing a result to it.

The benchmark
-------------

The evaluation follows `Corley et al. 2024, "Revisiting pre-trained remote sensing model benchmarks: resizing and normalization matters" <https://arxiv.org/abs/2305.13456>`_: pretrain on EuroSAT without labels, freeze the encoder, and score the frozen features with a k-nearest-neighbour classifier. kNN is used rather than a linear probe because it has no optimizer, no learning rate, and no regularization of its own, so it measures the representation rather than the tuning of the probe.

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
     - 224x224, produced by each trainer's ``RandomResizedCrop`` from the native 64x64 imagery
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

EuroSAT test-set kNN-5 top-1 accuracy, using each trainer's default augmentations. Every row is the best learning rate for that trainer and encoder, selected on validation.

.. list-table::
   :header-rows: 1
   :widths: 24 20 16 16 24

   * - Trainer
     - Encoder
     - Test acc
     - lr [#lr]_
     - Reproduce
   * - Image statistics [#floor]_
     - none
     - 0.8937
     - --
     - --
   * - Random initialization
     - ResNet-50
     - 0.8622
     - --
     - --
   * - Random initialization
     - ViT-S/16
     - 0.8507
     - --
     - --
   * - Supervised ImageNet
     - ResNet-50
     - 0.8948
     - --
     - --
   * - Supervised ImageNet
     - ViT-S/16
     - 0.9178
     - --
     - --
   * - MoCo v3
     - ResNet-50
     - **0.9476**
     - 1e-2
     - ``moco_resnet50``
   * - MoCo v3
     - ViT-S/16
     - 0.9396
     - 1e-3
     - ``moco_vit_small``
   * - FroSSL [#frossl]_
     - ViT-S/16
     - 0.9400
     - 2e-4
     - --
   * - FroSSL [#frossl]_
     - ResNet-50
     - 0.9156
     - 2e-3
     - --
   * - BYOL [#byolfix]_
     - ViT-S/16
     - 0.9398
     - 1e-4
     - --
   * - BYOL [#byolfix]_
     - ResNet-50
     - 0.9333
     - 3e-4
     - --
   * - SimCLR
     - ResNet-50
     - 0.9393
     - 1.5
     - ``simclr_resnet50``
   * - SimCLR
     - ViT-S/16
     - 0.9326
     - 1.5
     - ``simclr_vit_small``
   * - MAE
     - ViT-S/16
     - 0.8978
     - 7.5e-5
     - ``mae_vit_small``

Reading the table:

* **Beat the floors, not just random init.** 52 hand-computed per-band image statistics, with no network at all, score 0.8937. A trainer that lands below that has not learned anything a mean and a standard deviation do not already capture. Supervised ImageNet transfer on a ViT is a harder floor at 0.9178.
* **The learning rate is a property of the trainer and encoder together.** The best rates span four orders of magnitude, and the optimum moves with the encoder: FroSSL prefers 2e-4 on a ViT and 2e-3 on a ResNet. A new trainer needs its own sweep, not a borrowed default. The library defaults are tuned for batch 4096 and are far too large here; MoCo's default of 9.6 diverges to NaN under AdamW at this batch size.
* **Differences of a few thousandths are not meaningful.** Every number comes from a single seed, so treat gaps below roughly 0.005 as noise.

.. rubric:: Footnotes

.. [#lr] Learning rates were selected on validation accuracy under a non-default augmentation pipeline and reused for the default pipeline reported here, so these numbers may understate what each trainer reaches with a dedicated sweep.

.. [#floor] Per-band mean, standard deviation, minimum, and maximum of each image, concatenated into a 52-dimensional vector and fed to the same kNN probe. No network and no training.

.. [#frossl] FroSSL is not part of TorchGeo. It is included as a worked example of evaluating a new trainer, ported from a pending `lightly <https://github.com/lightly-ai/lightly/pull/1962>`__ pull request.

.. [#byolfix] Measured with a corrected BYOL in which the target network is a real exponential moving average of the online network. The shipped :class:`~torchgeo.tasks.BYOL` passes one encoder to both wrappers, so the momentum update is a no-op.

Reproducing a row
-----------------

Rows with a name in the last column are reproducible from a configuration file using the shipped trainers and :class:`~torchgeo.datamodules.EuroSATDataModule`, whose default normalization is exactly the per-band standardization this protocol requires. For example, ``moco_resnet50``:

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

The other rows differ only in ``model.init_args``:

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

All configurations use ``in_channels: 13`` and the same ``trainer`` and ``data`` blocks as above.

Scoring is not part of ``torchgeo fit``: the trainer writes a checkpoint, and the kNN probe is applied afterwards. Given a frozen encoder, the whole probe is:

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

Adding a new trainer
--------------------

A new SSL trainer should arrive with the same pieces as any other task:

#. ``torchgeo/tasks/foo.py``, subclassing :class:`~torchgeo.tasks.BaseTask`.
#. An entry in ``torchgeo/tasks/__init__.py``.
#. ``tests/conf/<dataset>_foo.yaml`` plus a case in ``tests/tasks/test_foo.py``.
#. ``docs/api/tasks.rst``.

Before claiming it works, sweep at least three or four learning rates, confirm the result clears the image-statistics floor, and check that the representation has not collapsed: the standard deviation of the embeddings should stay well away from zero, and the mean pairwise cosine similarity well away from one. Report the number alongside the table above, using the same protocol.
