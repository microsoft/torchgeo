# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Detection tasks."""

from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
from packaging.version import parse
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

if parse(torchvision.__version__) >= parse("0.13"):
    from torchvision.models import resnet as R

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

from ..datasets.utils import unbind_samples

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class ObjectDetectionTask(pl.LightningModule):
    """LightningModule for object detection of images.

    .. versionadded:: 0.4
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        backbone_pretrained = self.hyperparams.get("pretrained", True)
        if self.hyperparams["detection_model"] == "faster-rcnn":
            if "resnet" in self.hyperparams["backbone"]:
                kwargs = {
                    "backbone_name": self.hyperparams["backbone"],
                    "trainable_layers": self.hyperparams.get("trainable_layers", 3),
                }
                if parse(torchvision.__version__) >= parse("0.13"):
                    if backbone_pretrained:
                        kwargs["weights"] = BACKBONE_WEIGHT_MAP[
                            self.hyperparams["backbone"]
                        ]
                    else:
                        kwargs["weights"] = None
                else:
                    kwargs["pretrained"] = backbone_pretrained

                backbone = resnet_fpn_backbone(**kwargs)
            else:
                raise ValueError(
                    f"Backbone type '{self.hyperparams['backbone']}' is not valid."
                )

            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
            )

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )
            num_classes = self.hyperparams["num_classes"]
            self.model = FasterRCNN(
                backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )

        else:
            raise ValueError(
                f"Model type '{self.hyperparams['detection_model']}' is not valid."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            detection_model: Name of the detection model type to use
            backbone: Name of the model backbone to use
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
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

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
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
                summary_writer = self.logger.experiment  # type: ignore[union-attr]
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except AttributeError:
                pass

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
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

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
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
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
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
