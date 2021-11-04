# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced trainer."""

from .tasks import ClassificationTask


class UCMercedClassificationTask(ClassificationTask):
    """LightningModule for training models on the UC Merced Dataset."""

    num_classes = 21
