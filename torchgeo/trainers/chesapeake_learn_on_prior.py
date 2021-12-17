# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake with prior datasets."""

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

from ..datasets import ChesapeakeCVPRPrior
from ..losses import loss_on_prior_reversed_kl_simple, loss_on_prior_simple
from ..models import FCN_modified
from ..samplers import GridGeoSampler, RandomBatchGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"


# define colors for chesapake prior
CHESAPEAKE_4_NO_ZEROS_CLASS_COLORS = {
    0: (0, 197, 255, 255),
    1: (38, 115, 0, 255),
    2: (163, 255, 115, 255),
    3: (156, 156, 156, 255),
}


def get_colors(class_colors):
    """Converts color dict to array."""
    return np.array([class_colors[c] for c in class_colors.keys()]) / 255.0


lc_colors = {"chesapeake_4_no_zeros": get_colors(CHESAPEAKE_4_NO_ZEROS_CLASS_COLORS)}


# visualize predicitons and priors
def vis_lc(r, lc_type, renorm=True, reindexed=True):
    """Visualize colors for certain landcover type."""
    colors = lc_colors[lc_type]

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


class ChesapeakeCVPRPriorSegmentationTask(LightningModule):
    """LightningModule for training models on the Chesapeake CVPR Land Cover dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        """Configures the task based on kwargs parameters."""
        if kwargs["condense_barren"]:
            n_classes = 4
            # need the n_classes_with_nodata for the padding in test set
            self.n_classes_with_nodata = n_classes + 1
            self.ignore_index = 4

            self.lc_type = "chesapeake_4_no_zeros"

        else:
            print("condense barren must be true to use this trainer")
            return

        print("lc type is ", self.lc_type)
        self.n_classes = n_classes

        qr_losses = ["qr_forward", "qr_reverse"]
        self.need_to_add_smoothing = (kwargs["segmentation_model"] != "fcn") and (
            kwargs["loss"] in qr_losses
        )
        if self.need_to_add_smoothing:
            print("will add smoothing after softmax")
            self.output_smooth = kwargs["output_smooth"]

        print(n_classes)
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=4,
                classes=n_classes,
                activation="softmax",
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=4,
                classes=n_classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.model = FCN_modified(
                in_channels=4,
                classes=n_classes,
                num_filters=kwargs["num_filters"],
                output_smooth=kwargs["output_smooth"],
            )
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "qr_forward":
            self.loss = loss_on_prior_simple
        elif kwargs["loss"] == "qr_reverse":
            self.loss = loss_on_prior_reversed_kl_simple
        elif kwargs["loss"] == "nll":
            self.loss = nn.NLLLoss()
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
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

        self.train_accuracy_q = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_accuracy_q = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_accuracy_q = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_iou_q = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_iou_q = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_q = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_accuracy_r = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_accuracy_r = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_accuracy_r = Accuracy(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

        self.train_iou_r = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.val_iou_r = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )
        self.test_iou_r = IoU(
            num_classes=self.n_classes_with_nodata, ignore_index=self.ignore_index
        )

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        preds = self.model(x)

        if self.need_to_add_smoothing:
            preds = nn.functional.normalize(
                preds + self.output_smooth, p=1, dim=1
            ).log()
        return preds

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hr = batch["highres_labels"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        with torch.no_grad():
            z = nn.functional.normalize(torch.exp(y_hat), p=1, dim=(0, 2, 3))
            # y is the prior
            r_hat_hard = (z * y).argmax(dim=1)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        self.train_accuracy_q(y_hat_hard, y_hr)
        self.train_iou_q(y_hat_hard, y_hr)
        self.train_accuracy_r(r_hat_hard, y_hr)
        self.train_iou_r(r_hat_hard, y_hr)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc_q", self.train_accuracy_q.compute())
        self.log("train_acc_r", self.train_accuracy_r.compute())
        self.log("train_iou_q", self.train_iou_q.compute())
        self.log("train_iou_r", self.train_iou_r.compute())
        self.train_accuracy_q.reset()
        self.train_accuracy_r.reset()
        self.train_iou_q.reset()
        self.train_iou_r.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hr = batch["highres_labels"]

        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        with torch.no_grad():
            z = nn.functional.normalize(torch.exp(y_hat), p=1, dim=(0, 2, 3))
            r_hat_hard = (z * y).argmax(dim=1)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)

        self.val_accuracy_q(y_hat_hard, y_hr)
        self.val_iou_q(y_hat_hard, y_hr)

        self.val_accuracy_r(r_hat_hard, y_hr)
        self.val_iou_r(r_hat_hard, y_hr)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0].cpu().numpy(), 0, 3
            )

            prior = batch["mask"][0]
            prior_vis = vis_lc(prior.cpu().numpy(), self.lc_type).T.swapaxes(0, 1)
            highres_labels_vis = vis_lc(
                batch["highres_labels"][0].cpu().numpy(), self.lc_type
            ).T.swapaxes(0, 1)

            q = torch.exp(y_hat[0])
            pred_vis = vis_lc(q.cpu().numpy(), self.lc_type).T.swapaxes(0, 1)
            # calculated r (one one image, so classes are on dim 0)
            r = nn.functional.normalize(z[0] * prior, p=1, dim=0)
            r_vis = vis_lc(r.cpu().numpy(), self.lc_type).T.swapaxes(0, 1)

            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            axs[0].imshow(img[:, :, :3])
            axs[0].set_title("NAIP")
            axs[0].axis("off")
            axs[1].imshow(prior_vis, interpolation="none")
            axs[1].set_title("prior")
            axs[1].axis("off")
            axs[2].imshow(pred_vis, interpolation="none")
            axs[2].set_title("q()")
            axs[2].axis("off")
            plt.tight_layout()
            axs[3].imshow(r_vis, interpolation="none")
            axs[3].set_title("r = z(q)*prior")
            axs[3].axis("off")
            axs[4].set_title("highres labels (CS)")
            axs[4].imshow(highres_labels_vis, interpolation="none")
            axs[4].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )
            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc_q", self.val_accuracy_q.compute())
        self.log("val_iou_q", self.val_iou_q.compute())
        self.log("val_acc_r", self.val_accuracy_r.compute())
        self.log("val_iou_r", self.val_iou_r.compute())
        self.val_accuracy_q.reset()
        self.val_accuracy_r.reset()
        self.val_iou_q.reset()
        self.val_iou_r.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"]
        y_hr = batch["highres_labels"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        with torch.no_grad():
            z = nn.functional.normalize(torch.exp(y_hat), p=1, dim=(0, 2, 3))
            r_hat_hard = (z * y).argmax(dim=1)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy_q(y_hat_hard, y_hr)
        self.test_iou_q(y_hat_hard, y_hr)
        self.test_accuracy_r(r_hat_hard, y_hr)
        self.test_iou_r(r_hat_hard, y_hr)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_q", self.test_accuracy_q.compute())
        self.log("test_acc_r", self.test_accuracy_r.compute())
        self.log("test_iou_q", self.test_iou_q.compute())
        self.log("test_iou_r", self.test_iou_r.compute())
        self.test_accuracy_q.reset()
        self.test_accuracy_r.reset()
        self.test_iou_q.reset()
        self.test_iou_r.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, patience=self.hparams["learning_rate_schedule_patience"]
                ),
                "monitor": "val_loss",
                "verbose": True,
            },
        }


class ChesapeakeCVPRPriorDataModule(LightningDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        states_str: str,
        prior_version: str,
        patches_per_tile: int = 200,
        patch_size: int = 128,
        batch_size: int = 64,
        num_workers: int = 4,
        prior_smoothing_constant: float = 1e-2,
        condense_barren: bool = True,
        condense_road_and_impervious: bool = True,
        train_set: str = "train",
        val_set: str = "val",
        test_set: str = "test",
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Chesapeake CVPR based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            patches_per_tile: The number of patches per tile to sample
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            condense_barren: whether to condense the barren class with impervious
            condense_road_and_impervious: whether to condense roads with impervious, if
                condense_barren is false
            patch_size: size of patches to use
            prior_smoothing_constant: additive smoothing to prior
            prior_version: string describing what prior to use
            states_str: states to use for training, concatenated with '+'
            test_set: string describing what set to test on
            val_set: string describing what set to validate on
            train_set: string describing what set to train on
        """
        super().__init__()  # type: ignore[no-untyped-call]

        states = states_str.split("+")

        self.root_dir = root_dir
        self.states = states
        self.prior_version = prior_version
        self.layers = ["naip-new", f"prior_{prior_version}", "lc"]
        self.patches_per_tile = np.int(patches_per_tile / len(states))
        print(self.patches_per_tile, " patches_per_tile")
        self.patch_size = patch_size
        self.original_patch_size = 512
        self.batch_size = batch_size
        print(self.batch_size, " batch size")
        self.num_workers = num_workers
        self.prior_smoothing_constant = prior_smoothing_constant
        self.condense_road_and_impervious = condense_road_and_impervious
        self.condense_barren = condense_barren
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
        # separate out the highres labels
        sample["highres_labels"] = sample["mask"][-1].int()  # labels
        sample["mask"] = sample["mask"][:-1]  # prior

        impervious_idxs_highres_orig = [4, 5, 6]
        impervious_idx_condesed = [4]
        if self.condense_barren:
            # prior mask should be good to go -- we'll normalize it below

            # Now remap all the highres labels
            impervious_idxs_highres_orig = [4, 5, 6]
            impervious_idx_condesed = 4
            for c_idx in impervious_idxs_highres_orig:
                sample["highres_labels"][
                    sample["highres_labels"] == c_idx
                ] = impervious_idx_condesed

            nodata_idx_pad_orig = 7
            nodata_idx_pad_condensed = 5  # note that we're about to subtract one

            sample["highres_labels"][
                sample["highres_labels"] == nodata_idx_pad_orig
            ] = nodata_idx_pad_condensed
            # subtract 1 so it starts with 0
            sample["highres_labels"] = sample["highres_labels"] - 1

        # note: we never use this option in the experiments
        elif self.condense_road_and_impervious:
            impervious_idx = 5
            road_idx = 6
            # recast road idx to impervious
            sample["highres_labels"][
                sample["highres_labels"] == road_idx
            ] = impervious_idx
            # subtract 1
            sample["highres_labels"] = sample["highres_labels"] - 1

            # add roads to impervious then remove road index
            sample["mask"][impervious_idx] += sample["mask"][road_idx]
            # remove road index and nodata
            sample["mask"] = sample["mask"][1:-1]

        # make sure prior is normalized, then smooth
        sample["mask"] = nn.functional.normalize(sample["mask"].float(), p=1, dim=0)
        sample["mask"] = nn.functional.normalize(
            sample["mask"] + self.prior_smoothing_constant, p=1, dim=0
        )
        sample["image"] = sample["image"].float() / 255.0

        sample["mask"] = sample["mask"]  # .squeeze()
        sample["highres_labels"] = sample["highres_labels"]  # .squeeze()
        return sample

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        ChesapeakeCVPRPrior(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            transforms=None,
            download=False,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """
        train_transforms = Compose([self.center_crop(self.patch_size), self.preprocess])
        val_transforms = Compose([self.center_crop(self.patch_size), self.preprocess])
        test_transforms = Compose(
            [
                self.pad_to(self.original_patch_size, image_value=0, mask_value=7),
                self.preprocess,
            ]
        )

        self.train_dataset = ChesapeakeCVPRPrior(
            self.root_dir,
            splits=self.train_sets,
            layers=self.layers,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = ChesapeakeCVPRPrior(
            self.root_dir,
            splits=self.val_sets,
            layers=self.layers,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = ChesapeakeCVPRPrior(
            self.root_dir,
            splits=self.test_sets,
            layers=self.layers,
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
