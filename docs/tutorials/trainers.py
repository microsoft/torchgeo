# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="b13c2251"
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the MIT License.

# + [markdown] id="e563313d"
# # Lightning Trainers
#
# In this tutorial, we demonstrate TorchGeo trainers to train and test a model. We will use the [EuroSAT](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#eurosat) dataset throughout this tutorial. Specifically, a subset containing only 100 images. We will train models to predict land cover classes.
#
# It's recommended to run this notebook on Google Colab if you don't have your own GPU. Click the "Open in Colab" button above to get started.

# + [markdown] id="8c1f4156"
# ## Setup
#
# First, we install TorchGeo and TensorBoard.

# + id="3f0d31a8"
# %pip install torchgeo tensorboard

# + [markdown] id="c90c94c7"
# ## Imports
#
# Next, we import TorchGeo and any other libraries we need.

# + id="bd39f485"
# %matplotlib inline
# %load_ext tensorboard

import os
import tempfile

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask

# + [markdown] id="e6e1d9b6"
# ## Lightning modules
#
# Our trainers use [Lightning](https://lightning.ai/docs/pytorch/stable/) to organize both the training code, and the dataloader setup code. This makes it easy to create and share reproducible experiments and results.
#
# First we'll create a `EuroSAT100DataModule` object which is simply a wrapper around the [EuroSAT100](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#eurosat) dataset. This object 1.) ensures that the data is downloaded, 2.) sets up PyTorch `DataLoader` objects for the train, validation, and test splits, and 3.) ensures that data from the same region **is not** shared between the training and validation sets so that you can properly evaluate the generalization performance of your model.

# + [markdown] id="9f2daa0d"
# The following variables can be modified to control training.

# + id="8e100f8b" nbmake={"mock": {"batch_size": 1, "fast_dev_run": true, "max_epochs": 1, "num_workers": 0}}
batch_size = 10
num_workers = 2
max_epochs = 50
fast_dev_run = False

# + id="0f2a04c7"
root = os.path.join(tempfile.gettempdir(), 'eurosat100')
datamodule = EuroSAT100DataModule(
    root=root, batch_size=batch_size, num_workers=num_workers, download=True
)

# + [markdown] id="056b7b4c"
# Next, we create a `ClassificationTask` object that holds the model object, optimizer object, and training logic. We will use a ResNet-18 model that has been pre-trained on Sentinel-2 imagery.

# + id="ba5c5442"
task = ClassificationTask(
    loss='ce',
    model='resnet18',
    weights=ResNet18_Weights.SENTINEL2_ALL_MOCO,
    in_channels=13,
    num_classes=10,
    lr=0.1,
    patience=5,
)

# + [markdown] id="d4b67f3e"
# ## Training
#
# Now that we have the Lightning modules set up, we can use a Lightning [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html) to run the training and evaluation loops. There are many useful pieces of configuration that can be set in the `Trainer` -- below we set up model checkpointing based on the validation loss, early stopping based on the validation loss, and a TensorBoard based logger. We encourage you to see the [Lightning docs](https://lightning.ai/docs/pytorch/stable/) for other options that can be set here, e.g. CSV logging, automatically selecting your optimizer's learning rate, and easy multi-GPU training.

# + id="ffe26e5c"
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
default_root_dir = os.path.join(tempfile.gettempdir(), 'experiments')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=default_root_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10)
logger = TensorBoardLogger(save_dir=default_root_dir, name='tutorial_logs')

# + [markdown] id="06afd8c7"
# For tutorial purposes we deliberately lower the maximum number of training epochs.

# + id="225a6d36"
trainer = Trainer(
    accelerator=accelerator,
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=logger,
    min_epochs=1,
    max_epochs=max_epochs,
)

# + [markdown] id="44d71e8f"
# When we first call `.fit(...)` the dataset will be downloaded and checksummed (if it hasn't already). After this, the training process will kick off, and results will be saved so that TensorBoard can read them.

# + id="00e08790"
trainer.fit(model=task, datamodule=datamodule)

# + [markdown] id="73700fb5"
# We launch TensorBoard to visualize various performance metrics across training and validation epochs. We can see that our model is just starting to converge, and would probably benefit from additional training time and a lower initial learning rate.
# -

# %tensorboard --logdir "$default_root_dir"

# + [markdown] id="04cfc7a8"
# Finally, after the model has been trained, we can easily evaluate it on the test set.

# + id="604a3b2f"
trainer.test(model=task, datamodule=datamodule)
