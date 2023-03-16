# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Detection tasks."""

from functools import partial
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import torch
import torchvision.models.detection
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import resnet as R
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

from ..datasets.utils import unbind_samples

BACKBONE_LAT_DIM_MAP = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "resnext50_32x4d": 2048,
    "resnext101_32x8d": 2048,
    "wide_resnet50_2": 2048,
    "wide_resnet101_2": 2048,
}

BACKBONE_WEIGHT_MAP = {
    "resnet18": R.ResNet18_Weights.DEFAULT,
    "resnet34": R.ResNet34_Weights.DEFAULT,
    "resnet50": R.ResNet50_Weights.DEFAULT,
    "resnet101": R.ResNet101_Weights.DEFAULT,
    "resnet152": R.ResNet152_Weights.DEFAULT,
    "resnext50_32x4d": R.ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101_32x8d": R.ResNeXt101_32X8D_Weights.DEFAULT,
    "wide_resnet50_2": R.Wide_ResNet50_2_Weights.DEFAULT,
    "wide_resnet101_2": R.Wide_ResNet101_2_Weights.DEFAULT,
}


class ObjectDetectionTask(LightningModule):  # type: ignore[misc]
    """LightningModule for object detection of images.

    Currently, supports Faster R-CNN, FCOS, and RetinaNet models from
    `torchvision
    <https://pytorch.org/vision/stable/models.html
    #object-detection-instance-segmentation-and-person-keypoint-detection>`_ with
    one of the following *backbone* arguments:

    .. code-block:: python

        ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d','resnext101_32x8d', 'wide_resnet50_2',
        'wide_resnet101_2']

    .. versionadded:: 0.4
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        backbone_pretrained = self.hyperparams.get("pretrained", True)

        if self.hyperparams["backbone"] in BACKBONE_LAT_DIM_MAP:
            kwargs = {
                "backbone_name": self.hyperparams["backbone"],
                "trainable_layers": self.hyperparams.get("trainable_layers", 3),
            }
            if backbone_pretrained:
                kwargs["weights"] = BACKBONE_WEIGHT_MAP[self.hyperparams["backbone"]]
            else:
                kwargs["weights"] = None

            latent_dim = BACKBONE_LAT_DIM_MAP[self.hyperparams["backbone"]]
        else:
            raise ValueError(
                f"Backbone type '{self.hyperparams['backbone']}' is not valid."
            )

        num_classes = self.hyperparams["num_classes"]

        if self.hyperparams["model"] == "faster-rcnn":
            backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
            )

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )
            self.model = torchvision.models.detection.FasterRCNN(
                backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )
        elif self.hyperparams["model"] == "fcos":
            kwargs["extra_blocks"] = feature_pyramid_network.LastLevelP6P7(256, 256)
            kwargs["norm_layer"] = (
                misc.FrozenBatchNorm2d if kwargs["weights"] else torch.nn.BatchNorm2d
            )

            backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)),
            )

            self.model = torchvision.models.detection.FCOS(
                backbone, num_classes, anchor_generator=anchor_generator
            )
        elif self.hyperparams["model"] == "retinanet":
            kwargs["extra_blocks"] = feature_pyramid_network.LastLevelP6P7(
                latent_dim, 256
            )
            backbone = resnet_fpn_backbone(**kwargs)

            anchor_sizes = (
                (16, 20, 25),
                (32, 40, 50),
                (64, 80, 101),
                (128, 161, 203),
                (256, 322, 406),
                (512, 645, 812),
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            head = RetinaNetHead(
                backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )

            self.model = torchvision.models.detection.RetinaNet(
                backbone, num_classes, anchor_generator=anchor_generator, head=head
            )
        else:
            raise ValueError(f"Model type '{self.hyperparams['model']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            model: Name of the detection model type to use
            backbone: Name of the model backbone to use
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        Raises:
            ValueError: if kwargs arguments are invalid

        .. versionchanged:: 0.4
           The *detection_model* parameter was renamed to *model*.
        """
        super().__init__()
        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

        self.val_metrics = MeanAveragePrecision()
        self.test_metrics = MeanAveragePrecision()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        batch_size = x.shape[0]
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        loss_dict = self(x, y)
        train_loss = sum(loss_dict.values())

        self.log_dict(loss_dict)

        return cast(Tensor, train_loss)

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        batch_size = x.shape[0]
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x)

        self.val_metrics.update(y_hat, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction_boxes"] = [b["boxes"].cpu() for b in y_hat]
                batch["prediction_labels"] = [b["labels"].cpu() for b in y_hat]
                batch["prediction_scores"] = [b["scores"].cpu() for b in y_hat]
                batch["image"] = batch["image"].cpu()
                sample = unbind_samples(batch)[0]
                # Convert image to uint8 for plotting
                if torch.is_floating_point(sample["image"]):
                    sample["image"] *= 255
                    sample["image"] = sample["image"].to(torch.uint8)
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except ValueError:
                pass

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        metrics = self.val_metrics.compute()
        renamed_metrics = {f"val_{i}": metrics[i] for i in metrics.keys()}
        self.log_dict(renamed_metrics)
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test MAP.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        batch_size = x.shape[0]
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x)

        self.test_metrics.update(y_hat, y)

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        metrics = self.test_metrics.compute()
        renamed_metrics = {f"test_{i}": metrics[i] for i in metrics.keys()}
        self.log_dict(renamed_metrics)
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> List[Dict[str, Tensor]]:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader

        Returns:
            list of predicted boxes, labels and scores
        """
        batch = args[0]
        x = batch["image"]
        y_hat: List[Dict[str, Tensor]] = self(x)
        return y_hat

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_map",
            },
        }
