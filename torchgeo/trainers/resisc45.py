# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 trainer."""

from .tasks import ClassificationTask


class RESISC45ClassificationTask(ClassificationTask):
    """LightningModule for training models on the RESISC45 Dataset."""

    num_classes = 45
