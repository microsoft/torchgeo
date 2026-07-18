# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Deprecated alias of torchgeo.trainers."""

from typing import Any

from typing_extensions import deprecated

from ..tasks import (
    BYOL,
    BaseTask,
    ChangeDetection,
    Classification,
    ClassificationMixin,
    InstanceSegmentation,
    IOBench,
    MoCo,
    ObjectDetection,
    PixelwiseRegression,
    Regression,
    SemanticSegmentation,
    SimCLR,
)


@deprecated('Use torchgeo.tasks.Classification instead')
class Classification(Classification):
    """Deprecated alias of torchgeo.tasks.Classification."""


@deprecated('Use torchgeo.tasks.ClassificationMixin instead')
class ClassificationMixin(ClassificationMixin):
    """Deprecated alias of torchgeo.tasks.ClassificationMixin."""


@deprecated('Use torchgeo.tasks.BaseTask instead')
class BaseTask(BaseTask):
    """Deprecated alias of torchgeo.tasks.BaseTask."""


@deprecated('Use torchgeo.tasks.BYOL instead')
class BYOLTask(BYOL):
    """Deprecated alias of torchgeo.tasks.BYOL."""


@deprecated('Use torchgeo.tasks.ChangeDetection instead')
class ChangeDetectionTask(ChangeDetection):
    """Deprecated alias of torchgeo.tasks.ChangeDetection."""


@deprecated('Use torchgeo.tasks.ObjectDetection instead')
class ObjectDetectionTask(ObjectDetection):
    """Deprecated alias of torchgeo.tasks.ObjectDetection."""


@deprecated('Use torchgeo.tasks.InstanceSegmentation instead')
class InstanceSegmentationTask(InstanceSegmentation):
    """Deprecated alias of torchgeo.tasks.InstanceSegmentation."""


@deprecated('Use torchgeo.tasks.IOBench instead')
class IOBenchTask(IOBench):
    """Deprecated alias of torchgeo.tasks.IOBench."""


@deprecated('Use torchgeo.tasks.MoCo instead')
class MoCoTask(MoCo):
    """Deprecated alias of torchgeo.tasks.MoCo."""


@deprecated('Use torchgeo.tasks.PixelwiseRegression instead')
class PixelwiseRegressionTask(PixelwiseRegression):
    """Deprecated alias of torchgeo.tasks.PixelwiseRegression."""


@deprecated('Use torchgeo.tasks.Regression instead')
class RegressionTask(Regression):
    """Deprecated alias of torchgeo.tasks.Regression."""


@deprecated('Use torchgeo.tasks.SemanticSegmentation instead')
class SemanticSegmentation(SemanticSegmentation):
    """Deprecated alias of torchgeo.tasks.SemanticSegmentation."""


@deprecated('Use torchgeo.tasks.SimCLR instead')
class SimCLRTask(SimCLR):
    """Deprecated alias of torchgeo.tasks.SimCLR."""


@deprecated('Use torchgeo.tasks.Classification instead')
class MultiLabelClassificationTask(Classification):
    """Deprecated alias of torchgeo.tasks.Classification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Wrapper around torchgeo.tasks.Classification to massage kwargs."""
        kwargs['task'] = 'multilabel'
        kwargs['num_labels'] = kwargs['num_classes']
        super().__init__(*args, **kwargs)
