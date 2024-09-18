# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# flake8: noqa: E501
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# # Custom Trainers
#
# In this tutorial, we demonstrate how to extend a TorchGeo ["trainer class"](https://torchgeo.readthedocs.io/en/latest/api/trainers.html). In TorchGeo there exist several trainer classes that are pre-made PyTorch Lightning Modules designed to allow for the easy training of models on semantic segmentation, classification, change detection, etc. tasks using TorchGeo's [prebuild DataModules](https://torchgeo.readthedocs.io/en/latest/api/datamodules.html). While the trainers aim to provide sensible defaults and customization options for common tasks, they will not be able to cover all situations (e.g. researchers will likely want to implement and use their own architectures, loss functions, optimizers, etc. in the training routine). If you run into such a situation, then you can simply extend the trainer class you are interested in, and write custom logic to override the default functionality.
#
# This tutorial shows how to do exactly this to customize a learning rate schedule, logging, and model checkpointing for a semantic segmentation task using the [LandCoverAI](https://landcover.ai.linuxpolska.com/) dataset.
#
# It's recommended to run this notebook on Google Colab if you don't have your own GPU. Click the "Open in Colab" button above to get started.

# ## Setup
#
# As always, we install TorchGeo.

# %pip install torchgeo[datasets]

# ## Imports
#
# Next, we import TorchGeo and any other libraries we need.

# Get rid of the pesky warnings raised by kornia
# UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
import warnings
from collections.abc import Sequence
from typing import Any

import lightning
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    FBetaScore,
    JaccardIndex,
    Precision,
    Recall,
)

from torchgeo.datamodules import LandCoverAIDataModule

# +
from torchgeo.trainers import SemanticSegmentationTask

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')


# -

# ## Custom SemanticSegmentationTask
#
# Now, we create a `CustomSemanticSegmentationTask` class that inhierits from `SemanticSegmentationTask` and that overrides a few methods:
# - `__init__`: We add two new parameters `tmax` and `eta_min` to control the learning rate scheduler
# - `configure_optimizers`: We use the `CosineAnnealingLR` learning rate scheduler instead of the default `ReduceLROnPlateau`
# - `configure_metrics`: We add a "MeanIou" metric (what we will use to evaluate the model's performance) and a variety of other classification metrics
# - `configure_callbacks`: We demonstrate how to stack `ModelCheckpoint` callbacks to save the best checkpoint as well as periodic checkpoints
# - `on_train_epoch_start`: We log the learning rate at the start of each epoch so we can easily see how it decays over a training run
#
# Overall these demonstrate how to customize the training routine to investigate specific research questions (e.g. of the scheduler on test performance).


class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    # any keywords we add here between *args and **kwargs will be found in self.hparams
    def __init__(
        self, *args: Any, tmax: int = 50, eta_min: float = 1e-6, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)  # pass args and kwargs to the parent class

    def configure_optimizers(
        self,
    ) -> 'lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig':
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        tmax: int = self.hparams['tmax']
        eta_min: float = self.hparams['eta_min']

        optimizer = AdamW(self.parameters(), lr=self.hparams['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor},
        }

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams['num_classes']

        self.train_metrics = MetricCollection(
            {
                'OverallAccuracy': Accuracy(
                    task='multiclass', num_classes=num_classes, average='micro'
                ),
                'OverallPrecision': Precision(
                    task='multiclass', num_classes=num_classes, average='micro'
                ),
                'OverallRecall': Recall(
                    task='multiclass', num_classes=num_classes, average='micro'
                ),
                'OverallF1Score': FBetaScore(
                    task='multiclass',
                    num_classes=num_classes,
                    beta=1.0,
                    average='micro',
                ),
                'MeanIoU': JaccardIndex(
                    num_classes=num_classes, task='multiclass', average='macro'
                ),
            },
            prefix='train_',
        )
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        """Initialize callbacks for saving the best and latest models.

        Returns:
            List of callbacks to apply.
        """
        return [
            ModelCheckpoint(every_n_epochs=50, save_top_k=-1, save_last=True),
            ModelCheckpoint(monitor=self.monitor, mode=self.mode, save_top_k=5),
        ]

    def on_train_epoch_start(self) -> None:
        """Log the learning rate at the start of each training epoch."""
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            lr = optimizers[0].param_groups[0]['lr']
        else:
            lr = optimizers.param_groups[0]['lr']
        self.logger.experiment.add_scalar('lr', lr, self.current_epoch)  # type: ignore


# ## Train model
#
# The remainder of the turial is straightforward and follows the typical [PyTorch Lightning](https://lightning.ai/) training routine. We instantiate a `DataModule` for the LandCover.AI dataset, instantiate a `CustomSemanticSegmentationTask` with a U-Net and ResNet-50 backbone, then train the model using a Lightning trainer.

dm = LandCoverAIDataModule(root='data/', batch_size=64, num_workers=8, download=True)

task = CustomSemanticSegmentationTask(
    model='unet',
    backbone='resnet50',
    weights=True,
    in_channels=3,
    num_classes=6,
    loss='ce',
    lr=1e-3,
    tmax=50,
)

# validate that the task's hyperparameters are as expected
task.hparams

# The following Trainer config is useful just for testing the code in this notebook.
trainer = pl.Trainer(
    limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0, max_epochs=1
)
# You can use the following for actual training runs.
# trainer = pl.Trainer(min_epochs=150, max_epochs=250, log_every_n_steps=50)

trainer.fit(task, dm)

# ## Test model
#
# Finally, we test the model (optionally loading from a previously saved checkpoint).

# You can load directly from a saved checkpoint with `.load_from_checkpoint(...)`
task = CustomSemanticSegmentationTask.load_from_checkpoint(
    'lightning_logs/version_0/checkpoints/epoch=0-step=1.ckpt'
)

trainer.test(task, dm)
