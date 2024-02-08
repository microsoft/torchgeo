# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Digital Typhoon Data Module."""

from typing import Any

from ..datasets import DigitalTyphoonAnalysis
from ..datasets.digital_typhoon import _SampleSequenceDict
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class DigitalTyphoonAnalysisDataModule(NonGeoDataModule):
    """Digital Typhoon Analysis Data Module."""

    valid_split_types = ["time", "typhoon_id"]

    def __init__(
        self,
        split_by: str = "time",
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DigitalTyphoonAnalysisDataModule instance.

        Args:
            split_by: Either 'time' or 'typhoon_id', which decides how to split
                the dataset for train, val, test
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DigitalTyphoonAnalysis`.

        """
        super().__init__(DigitalTyphoonAnalysis, batch_size, num_workers, **kwargs)

        assert (
            split_by in self.valid_split_types
        ), f"Please choose from {self.valid_split_types}"
        self.split_by = split_by

    def _split_dataset(
        self, dataset: DigitalTyphoonAnalysis
    ) -> tuple[list[_SampleSequenceDict], list[_SampleSequenceDict]]:
        """Split dataset into two parts.

        Args:
            dataset: Dataset to be split into train/test or train/val subsets

        Returns:
            a tuple of the subset datasets
        """
        if self.split_by == "time":
            sequences = list(enumerate(dataset.sample_sequences))

            sorted_sequences = sorted(sequences, key=lambda x: x[1]["seq_id"])
            selected_indices = [x[0] for x in sorted_sequences]

            split_idx = int(len(sorted_sequences) * 0.8)
            train_indices = selected_indices[:split_idx]
            val_indices = selected_indices[split_idx:]

        else:
            sequences = list(enumerate(dataset.sample_sequences))
            train_indices, val_indices = group_shuffle_split(
                [x[1]["id"] for x in sequences], train_size=0.8, random_state=0
            )

        # select train and val sequences and remove enumeration
        train_sequences = [sequences[i][1] for i in train_indices]
        val_sequences = [sequences[i][1] for i in val_indices]

        return train_sequences, val_sequences

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = DigitalTyphoonAnalysis(**self.kwargs)

        train_sequences, test_sequences = self._split_dataset(self.dataset)

        if stage in ["fit", "validate"]:
            # resplit the train indices into train and val
            self.dataset.sample_sequences = train_sequences
            train_sequences, val_sequences = self._split_dataset(self.dataset)

            # create training dataset
            self.train_dataset = DigitalTyphoonAnalysis(**self.kwargs)
            self.train_dataset.sample_sequences = train_sequences

            # create validation dataseqt
            self.val_dataset = DigitalTyphoonAnalysis(**self.kwargs)
            self.val_dataset.sample_sequences = val_sequences

        if stage in ["test"]:
            self.test_dataset = DigitalTyphoonAnalysis(**self.kwargs)
            self.test_dataset.sample_sequences = test_sequences
