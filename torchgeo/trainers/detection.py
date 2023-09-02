# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for object detection."""

from functools import partial
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torchvision.models.detection
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
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


class ObjectDetectionTask(LightningModule):
    """Object detection.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        model: str = "faster-rcnn",
        backbone: str = "resnet50",
        weights: Optional[bool] = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        trainable_layers: int = 3,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new ObjectDetectionTask instance.

        Args:
            model: Name of the `torchvision
                <https://pytorch.org/vision/stable/models.html#object-detection>`__
                model to use. One of 'faster-rcnn', 'fcos', or 'retinanet'.
            backbone: Name of the `torchvision
                <https://pytorch.org/vision/stable/models.html#classification>`__
                backbone to use. One of 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', or 'wide_resnet101_2'.
            weights: Initial model weights. True for ImageNet weights, False or None
                for random weights.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes.
            trainable_layers: Number of trainable layers.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the detection
                head.

        Raises:
            ValueError: If any arguments are invalid.

        .. versionchanged:: 0.4
           *detection_model* was renamed to *model*.

        .. versionadded:: 0.5
           The *freeze_backbone* parameter.

        .. versionchanged:: 0.5
           *pretrained*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *weights*, *lr*, and *patience*.
        """
        super().__init__()

        self.save_hyperparameters()

        if backbone in BACKBONE_LAT_DIM_MAP:
            kwargs = {"backbone_name": backbone, "trainable_layers": trainable_layers}
            if weights:
                kwargs["weights"] = BACKBONE_WEIGHT_MAP[backbone]
            else:
                kwargs["weights"] = None

            latent_dim = BACKBONE_LAT_DIM_MAP[backbone]
        else:
            raise ValueError(f"Backbone type '{backbone}' is not valid.")

        if model == "faster-rcnn":
            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
            )

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

            if freeze_backbone:
                for param in model_backbone.parameters():
                    param.requires_grad = False

            self.model = torchvision.models.detection.FasterRCNN(
                model_backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )
        elif model == "fcos":
            kwargs["extra_blocks"] = feature_pyramid_network.LastLevelP6P7(256, 256)
            kwargs["norm_layer"] = (
                misc.FrozenBatchNorm2d if weights else torch.nn.BatchNorm2d
            )

            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)),
            )

            if freeze_backbone:
                for param in model_backbone.parameters():
                    param.requires_grad = False

            self.model = torchvision.models.detection.FCOS(
                model_backbone, num_classes, anchor_generator=anchor_generator
            )
        elif model == "retinanet":
            kwargs["extra_blocks"] = feature_pyramid_network.LastLevelP6P7(
                latent_dim, 256
            )
            model_backbone = resnet_fpn_backbone(**kwargs)

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
                model_backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )

            if freeze_backbone:
                for param in model_backbone.parameters():
                    param.requires_grad = False

            self.model = torchvision.models.detection.RetinaNet(
                model_backbone,
                num_classes,
                anchor_generator=anchor_generator,
                head=head,
            )
        else:
            raise ValueError(f"Model type '{model}' is not valid.")

        metrics = MetricCollection([MeanAveragePrecision()])
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self, x: list[Tensor], y: Optional[list[dict[str, Tensor]]] = None
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        """Forward pass of the model.

        Args:
            x: Mini-batch of images.
            y: Optional ground truth boxes present in the image.

        Returns:
            Output from the model.
        """
        out: tuple[dict[str, Tensor], list[dict[str, Tensor]]] = self.model(x, y)
        return out

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        batch_size = x.shape[0]
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        loss_dict = self(x, y)
        train_loss: Tensor = sum(loss_dict.values())

        self.log_dict(loss_dict)

        return train_loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        batch_size = x.shape[0]
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x)
        self.val_metrics(y_hat, y)

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
        """Log epoch level validation metrics."""
        # TODO: why is this method necessary?
        metrics = self.val_metrics.compute()

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop("val_classes", None)

        self.log_dict(metrics)
        self.val_metrics.reset()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        batch_size = x.shape[0]
        y = [
            {"boxes": batch["boxes"][i], "labels": batch["labels"][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x)

        self.test_metrics.update(y_hat, y)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Tensor]]:
        """Compute the predicted bounding boxes.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        y_hat: list[dict[str, Tensor]] = self(x)
        return y_hat

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=self.hparams["patience"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_map"},
        }
