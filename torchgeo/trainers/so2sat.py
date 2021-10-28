# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat trainer."""

import os
from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models
from torch.nn.modules import Conv2d, Linear
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import So2Sat
from . import utils
from .tasks import ClassificationTask

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class So2SatClassificationTask(ClassificationTask):
    """LightningModule for training models on the So2Sat Dataset."""

    num_classes = 17

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters passed to the constructor."""
        in_channels = self.hparams["in_channels"]

        pretrained = False
        if not os.path.exists(self.hparams["weights"]):
            if self.hparams["weights"] == "imagenet":
                pretrained = True
            elif self.hparams["weights"] == "random":
                pretrained = False
            else:
                raise ValueError(
                    f"Weight type '{self.hparams['weights']}' is not valid."
                )

        # Create the model
        if "resnet" in self.hparams["classification_model"]:
            self.model = getattr(
                torchvision.models.resnet, self.hparams["classification_model"]
            )(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = Linear(in_features, out_features=self.num_classes)

            # Update first layer
            if in_channels != 3:
                w_old = None
                if pretrained:
                    w_old = torch.clone(  # type: ignore[attr-defined]
                        self.model.conv1.weight
                    ).detach()
                # Create the new layer
                self.model.conv1 = Conv2d(
                    in_channels, 64, kernel_size=7, stride=1, padding=2, bias=False
                )
                nn.init.kaiming_normal_(  # type: ignore[no-untyped-call]
                    self.model.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

                # We copy over the pretrained RGB weights
                if pretrained:
                    w_new = torch.clone(  # type: ignore[attr-defined]
                        self.model.conv1.weight
                    ).detach()
                    w_new[:, :3, :, :] = w_old
                    self.model.conv1.weight = nn.Parameter(  # type: ignore[attr-defined] # noqa: E501
                        w_new
                    )
        else:
            raise ValueError(
                f"Model type '{self.hparams['classification_model']}' is not valid."
            )

        # Load pretrained weights checkpoint weights
        if "resnet" in self.hparams["classification_model"]:
            if os.path.exists(self.hparams["weights"]):
                name, state_dict = utils.extract_encoder(self.hparams["weights"])

                if self.hparams["classification_model"] != name:
                    raise ValueError(
                        f"Trying to load {name} weights into a "
                        f"{self.hparams['classification_model']}"
                    )

                self.model = utils.load_state_dict(self.model, state_dict)


class So2SatDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [
            -3.591224256609313e-05,
            -7.658561276843396e-06,
            5.9373857475971184e-05,
            2.5166231537121083e-05,
            0.04420110659759328,
            0.25761027084996196,
            0.0007556743372573258,
            0.0013503466830024448,
            0.12375696117681859,
            0.1092774636368323,
            0.1010855203267882,
            0.1142398616114001,
            0.1592656692023089,
            0.18147236008771792,
            0.1745740312291377,
            0.19501607349635292,
            0.15428468872076637,
            0.10905050699570007,
        ]
    ).reshape(18, 1, 1)

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [
            0.17555201137417686,
            0.17556463274968204,
            0.45998793417834255,
            0.455988755730148,
            2.8559909213125763,
            8.324800606439833,
            2.4498757382563103,
            1.4647352984509094,
            0.03958795985905458,
            0.047778262752410296,
            0.06636616706371974,
            0.06358874912497474,
            0.07744387147984592,
            0.09101635085921553,
            0.09218466562387101,
            0.10164581233948201,
            0.09991773043519253,
            0.08780632509122865,
        ]
    ).reshape(18, 1, 1)

    # this reorders the bands to put S2 RGB first, then remainder of S2, then S1
    reindex_to_rgb_first = [
        10,
        9,
        8,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        # 0,
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
    ]

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        bands: str = "rgb",
        unsupervised_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for So2Sat based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the So2Sat Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            bands: Either "rgb" or "s2"
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bands = bands
        self.unsupervised_mode = unsupervised_mode

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        # sample["image"] = (sample["image"] - self.band_means) / self.band_stds
        sample["image"] = sample["image"].float()
        sample["image"] = sample["image"][self.reindex_to_rgb_first, :, :]

        if self.bands == "rgb":
            sample["image"] = sample["image"][:3, :, :]

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        So2Sat(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_transforms = Compose([self.preprocess])
        val_test_transforms = self.preprocess

        if not self.unsupervised_mode:

            self.train_dataset = So2Sat(
                self.root_dir, split="train", transforms=train_transforms
            )

            self.val_dataset = So2Sat(
                self.root_dir, split="validation", transforms=val_test_transforms
            )

            self.test_dataset = So2Sat(
                self.root_dir, split="test", transforms=val_test_transforms
            )

        else:

            temp_train = So2Sat(
                self.root_dir, split="train", transforms=train_transforms
            )

            self.val_dataset = So2Sat(
                self.root_dir, split="validation", transforms=train_transforms
            )

            self.test_dataset = So2Sat(
                self.root_dir, split="test", transforms=train_transforms
            )

            self.train_dataset = cast(
                So2Sat, temp_train + self.val_dataset + self.test_dataset
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
