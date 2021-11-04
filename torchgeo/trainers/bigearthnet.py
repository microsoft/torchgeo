# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet trainer."""

from .tasks import MultiLabelClassificationTask


class BigEarthNetClassificationTask(MultiLabelClassificationTask):
    """LightningModule for training models on the BigEarthNet Dataset."""

    num_classes = 43
