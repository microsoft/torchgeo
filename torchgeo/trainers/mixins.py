# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mix-ins for trainers."""

import segmentation_models_pytorch as smp
import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    JaccardIndex,
    Precision,
    Recall,
)
from torchmetrics.wrappers import ClasswiseWrapper


class ClassificationMixin(LightningModule):
    """Mix-in for classification-based tasks.

    .. versionadded:: 0.9
    """

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        ignore_index: int | None = self.hparams['ignore_index']
        class_weights = self.hparams['class_weights']
        if class_weights is not None and not isinstance(class_weights, Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        match self.hparams['loss']:
            case 'ce':
                ignore_value = -1000 if ignore_index is None else ignore_index
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    ignore_index=ignore_value, weight=class_weights
                )
            case 'bce':
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self.hparams['pos_weight']
                )
            case 'jaccard':
                # JaccardLoss requires a list of classes to use instead of a class
                # index to ignore.
                if self.hparams['task'] == 'multiclass' and ignore_index is not None:
                    classes = [
                        i
                        for i in range(self.hparams['num_classes'])
                        if i != ignore_index
                    ]
                    self.criterion = smp.losses.JaccardLoss(
                        mode=self.hparams['task'], classes=classes
                    )
                else:
                    self.criterion = smp.losses.JaccardLoss(mode=self.hparams['task'])
            case 'focal':
                self.criterion = smp.losses.FocalLoss(
                    mode=self.hparams['task'],
                    ignore_index=ignore_index,
                    normalized=True,
                )
            case 'dice':
                self.criterion = smp.losses.DiceLoss(
                    mode=self.hparams['task'], ignore_index=ignore_index
                )

    def configure_metrics(self) -> None:
        r"""Initialize the performance metrics.

        Includes the following metrics:

        * :class:`~torchmetrics.Accuracy`: :math:`\frac{TP + TN}{P + N}`
        * :class:`~torchmetrics.Precision`: :math:`\frac{TP}{TP + FP}`
        * :class:`~torchmetrics.Recall`: :math:`\frac{TP}{P}`
        * :class:`~torchmetrics.F1Score`: :math:`\frac{2 TP}{2 TP + FP + FN}`
        * :class:`~torchmetrics.JaccardIndex`: :math:`\frac{TP}{TP + FN + FP}`

        See https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
        for more details.

        Higher values are better for all metrics. All metrics report multiple versions:

        * Overall (micro): Sum statistics over all labels
        * Average (macro): Calculate statistics for each label and average them
        * Classwise (none): Calculates statistic for each label and applies no reduction
        """
        kwargs = {
            'task': self.hparams['task'],
            'num_classes': self.hparams['num_classes'],
            'num_labels': self.hparams['num_labels'],
            'ignore_index': self.hparams['ignore_index'],
        }
        metrics_dict: dict[str, Metric | MetricCollection] = {}
        for metric in [Accuracy, Precision, Recall, F1Score, JaccardIndex]:
            metrics_dict |= {
                f'Overall{metric.__name__}': metric(average='micro', **kwargs),
                f'Average{metric.__name__}': metric(average='macro', **kwargs),
            }
            if self.hparams['task'] != 'binary':
                metrics_dict[metric.__name__] = ClasswiseWrapper(
                    metric(average='none', **kwargs),
                    labels=self.hparams['labels'],
                    prefix=f'Classwise{metric.__name__}_',
                )

        metrics = MetricCollection(metrics_dict)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def on_train_epoch_end(self) -> None:
        """Log train metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
