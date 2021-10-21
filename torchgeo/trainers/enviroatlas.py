# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake datasets."""

from typing import Any, Callable, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU
from torchvision.transforms import Compose

from ..datasets import Enviroatlas
from ..models import FCN_modified
from ..samplers import GridGeoSampler, RandomBatchGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"

ENVIROATLAS_CLASS_COLORS_DICT = {
    0: (255, 255, 255, 255),  #
    1: (0, 197, 255, 255),  # from CC Water
    2: (156, 156, 156, 255),  # from CC Impervious
    3: (255, 170, 0, 255),  # from CC Barren
    4: (38, 115, 0, 255),  # from CC Tree Canopy
    5: (204, 184, 121, 255),  # from NLCD shrub
    6: (163, 255, 115, 255),  # from CC Low Vegetation
    7: (220, 217, 57, 255),  # from NLCD Pasture/Hay color
    8: (171, 108, 40, 255),  # from NLCD Cultivated Crops
    9: (184, 217, 235, 255),  # from NLCD Woody Wetlands
    10: (108, 159, 184, 255),  # from NLCD Emergent Herbaceous Wetlands
    11: (0, 0, 0, 0),  # extra for black
    12: (70, 100, 159, 255),  # extra for dark blue
}


def get_colors(class_colors):
    """Map colors dict to colors array."""
    return np.array([class_colors[c] for c in class_colors.keys()]) / 255.0


ENVIORATLAS_CLASS_COLORS = get_colors(ENVIROATLAS_CLASS_COLORS_DICT)


def vis_lc_from_colors(r, colors, renorm=True, reindexed=True):
    """Function for visualizing color scheme with potentially soft class assigments."""
    sparse = r.shape[0] != len(colors)
    colors_cycle = range(0, len(colors))

    if sparse:
        z = np.zeros((3,) + r.shape)
        s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * (s == c).astype(float)

    else:
        z = np.zeros((3,) + r.shape[1:])
        if renorm:
            s = r / r.sum(0)
        else:
            s = r
        for c in colors_cycle:
            for ch in range(3):
                z[ch] += colors[c][ch] * s[c]
    return z


class EnviroatlasSegmentationTask(LightningModule):
    """LightningModule for training models on the Enviroatlas Land Cover dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        """Configures the task based on kwargs parameters."""
        self.classes_keep = kwargs["classes_keep"]
        self.colors = [ENVIORATLAS_CLASS_COLORS[c] for c in self.classes_keep]
        self.n_classes = len(self.classes_keep)
        self.n_classes_with_nodata = len(self.classes_keep) + 1
        self.ignore_index = len(self.classes_keep)

        if (
            "include_prior_as_datalayer" in kwargs.keys()
            and kwargs["include_prior_as_datalayer"]
        ):
            self.include_prior_as_datayer = True
            self.in_channels = 9  # 5 for prior, 4 for naip
        else:
            self.include_prior_as_datayer = False
            self.in_channels = 4

        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                activation=kwargs["activation_layer"],
                in_channels=self.in_channels,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=self.in_channels,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.model = FCN_modified(
                in_channels=self.in_channels,
                classes=self.n_classes,
                num_filters=kwargs["num_filters"],
                output_smooth=kwargs["output_smooth"],
            )
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=7
            )
        elif kwargs["loss"] == "nll":
            self.loss = nn.NLLLoss()

        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

        self.train_accuracy = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_accuracy = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_accuracy = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_iou = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_iou = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_per_class = IoU(
            num_classes=self.n_classes_with_nodata,
            ignore_index=self.ignore_index,
            reduction="none",
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_accuracy(y_hat_hard, y)
        self.train_iou(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc_q", self.train_accuracy.compute())
        self.log("train_iou_q", self.train_iou.compute())
        self.train_accuracy.reset()
        self.train_iou.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)
        self.val_accuracy(y_hat_hard, y)
        self.val_iou(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0][:4].cpu().numpy(), 0, 3
            )
            if self.include_prior_as_datayer:
                prior = batch["image"][0][4:].cpu().numpy()
                prior_vis = vis_lc_from_colors(prior, self.colors).T.swapaxes(0, 1)
                img[:, :, :3] = (prior_vis + img[:, :, :3]) / 2.0

            # This is specific to the 5 class definition
            mask = batch["mask"][0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            if self.include_prior_as_datayer:
                fig, axs = plt.subplots(1, 4, figsize=(12, 4))
                axs[0].imshow(img[:, :, :3])
                axs[0].axis("off")
                axs[1].imshow(prior_vis)
                axs[1].axis("off")
                axs[2].imshow(
                    vis_lc_from_colors(mask, self.colors).T.swapaxes(0, 1),
                    interpolation="none",
                )
                axs[2].axis("off")
                axs[2].set_title("labels")
                axs[3].imshow(
                    vis_lc_from_colors(pred, self.colors).T.swapaxes(0, 1),
                    interpolation="none",
                )
                axs[3].axis("off")
                axs[3].set_title("predictions")
                plt.tight_layout()
            else:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img[:, :, :3])
                axs[0].axis("off")
                axs[1].imshow(
                    vis_lc_from_colors(mask, self.colors).T.swapaxes(0, 1),
                    interpolation="none",
                )
                axs[1].axis("off")
                axs[1].set_title("labels")
                axs[2].imshow(
                    vis_lc_from_colors(pred, self.colors).T.swapaxes(0, 1),
                    interpolation="none",
                )
                axs[2].axis("off")
                axs[2].set_title("predictions")
                plt.tight_layout()

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc_q", self.val_accuracy.compute())
        self.log("val_iou_q", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        #  deal with nodata in the test batches
        loss = 0

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy(y_hat_hard, y)
        self.test_iou(y_hat_hard, y)
        self.test_iou_per_class(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_q", self.test_accuracy.compute())
        self.log("test_iou_q", self.test_iou.compute())
        # print(self.test_iou_per_class.compute())
        self.log_dict(
            dict(
                zip(
                    [f"iou_{x}" for x in np.arange(self.n_classes)],
                    self.test_iou_per_class.compute(),
                )
            )
        )
        self.test_accuracy.reset()
        self.test_iou.reset()
        self.test_iou_per_class.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
                "verbose": True,
            },
        }


class EnviroatlasDataModule(LightningDataModule):
    """LightningDataModule implementation for the Enviroatlas dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        states_str: str,
        classes_keep: list,
        patches_per_tile: int = 200,
        patch_size: int = 128,
        batch_size: int = 64,
        num_workers: int = 4,
        train_set: str = "train",
        val_set: str = "val",
        test_set: str = "test",
        include_prior_as_datalayer=False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Enviroatlas based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the Enviroatlas Dataset
            states_str: The states to use to train the model, concatenated with '+'
            patches_per_tile: The number of patches per tile to sample
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of each instance in the batch, in pixels
            prior_version: string describing which prior to use
            classes_keep: list of valid classes for the prediction problem
            prior_smoothing_constant: additive smoothing constant
            train_set: Set to train on
            val_set:  Set to validate on
            test_set: Set to test on
            include_prior_as_datalayer: whether prior is input data or supervision data

        """
        super().__init__()  # type: ignore[no-untyped-call]

        states = states_str.split("+")
        for state in states:
            assert state in [
                "pittsburgh_pa-2010_1m",
                "durham_nc-2012_1m",
                "austin_tx-2012_1m",
                "phoenix_az-2010_1m",
            ]

        print(patches_per_tile)
        self.root_dir = root_dir
        self.include_prior_as_datalayer = include_prior_as_datalayer
        if self.include_prior_as_datalayer:
            self.layers = [
                "a_naip",
                "prior_from_cooccurrences_101_31",
                "h_highres_labels",
            ]
        else:
            self.layers = ["a_naip", "h_highres_labels"]
        self.patches_per_tile = patches_per_tile
        self.patch_size = patch_size
        self.original_patch_size = 300
        self.batch_size = batch_size
        self.num_workers = num_workers

        # if the prior is to be used, use it as input layer, not output supervision
        self.prior_as_input = True

        self.classes_keep = classes_keep
        self.ignore_index = len(classes_keep)

        self.train_sets = [f"{state}-{train_set}" for state in states]
        self.val_sets = [f"{state}-{val_set}" for state in states]
        self.test_sets = [f"{state}-{test_set}" for state in states]
        print(f"train sets are: {self.train_sets}")
        print(f"val sets are: {self.val_sets}")
        print(f"test sets are: {self.test_sets}")

    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample."""

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape

            assert height <= size and width <= size, print(height, width, size)

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample

        return pad_inner

    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample."""

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height >= size and width >= size, f"{height} or {width} < {size}"

            y1 = (height - size) // 2
            x1 = (width - size) // 2
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

            return sample

        return center_crop_inner

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample."""
        # reindex to just the EnviroAtlas classes to be kept
        reindex_map = dict(zip(self.classes_keep, np.arange(len(self.classes_keep))))
        reindexed_mask = -1 * torch.ones(sample["mask"].shape)
        for old_idx, new_idx in reindex_map.items():
            reindexed_mask[sample["mask"] == old_idx] = new_idx

        reindexed_mask[reindexed_mask == -1] = self.ignore_index
        assert (reindexed_mask >= 0).all()

        sample["mask"] = reindexed_mask

        # note prior and images both should be divided by 255 as below
        sample["image"] = sample["image"] / 255.0
        # don't subtract 1 because we already reindexed
        sample["mask"] = sample["mask"].squeeze()

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        Enviroatlas(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=None,
            download=False,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """
        train_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.preprocess,
            ]
        )
        val_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.preprocess,
            ]
        )
        test_transforms = Compose(
            [
                self.pad_to(self.original_patch_size, image_value=0, mask_value=11),
                self.preprocess,
            ]
        )

        self.train_dataset = Enviroatlas(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = Enviroatlas(
            self.root_dir,
            splits=self.val_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = Enviroatlas(
            self.root_dir,
            splits=self.test_sets,
            layers=self.layers,
            prior_as_input=self.prior_as_input,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        print("train set of size ", self.train_dataset.index.get_size())
        print("original patch size", self.original_patch_size)
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * self.train_dataset.index.get_size(),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        print("original patch size", self.original_patch_size)
        sampler = RandomBatchGeoSampler(
            self.val_dataset,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * self.val_dataset.index.get_size() // 2,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        sampler = GridGeoSampler(
            self.test_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=16,  # self.batch_size // 2,
            sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
        )
