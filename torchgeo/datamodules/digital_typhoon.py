# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Digital Typhoon Data Module."""

from typing import Any

from torch.utils.data import Dataset, Subset

from ..datasets import DigitalTyphoonAnalysis
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
            split_by: Either 'time' or 'typhoon_id', which decides how to split the dataset for train, val, test
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

    def split_dataset(self, dataset: Dataset) -> tuple[Subset]:
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

        # select train and val sequences
        train_sequences = [sequences[i] for i in train_indices]
        val_sequences = [sequences[i] for i in val_indices]

        # remove the enumeration
        train_sequences = [seq[1] for seq in train_sequences]
        val_sequences = [seq[1] for seq in val_sequences]

        if self.split_by == "time":

            def find_max_time_per_id(split_sequences):
                # Find the maximum value of each id in train_sequences
                max_values = {}
                for seq in split_sequences:
                    id = seq["id"]
                    value = max(seq["seq_id"])
                    if id not in max_values or value > max_values[id]:
                        max_values[id] = value
                return max_values

            train_max_values = find_max_time_per_id(train_sequences)
            val_max_values = find_max_time_per_id(val_sequences)
            # Assert that each max value in train_max_values is lower than in val_max_values for each key id
            for id, max_value in train_max_values.items():
                assert (
                    id not in val_max_values or max_value < val_max_values[id]
                ), f"Max value for id {id} in train is not lower than in validation."
        else:
            train_ids = {seq["id"] for seq in train_sequences}
            val_ids = {seq["id"] for seq in val_sequences}

            # Assert that the intersection between train_ids and val_ids is empty
            assert (
                len(train_ids & val_ids) == 0
            ), "Train and validation datasets have overlapping ids."

        return train_sequences, val_sequences

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = DigitalTyphoonAnalysis(**self.kwargs)

        train_sequences, test_sequences = self.split_dataset(self.dataset)

        if stage in ["fit", "validate"]:
            # resplit the train indices into train and val
            self.dataset.sample_sequences = train_sequences
            train_sequences, val_sequences = self.split_dataset(self.dataset)

            # create training dataset
            self.train_dataset = DigitalTyphoonAnalysis(**self.kwargs)
            self.train_dataset.sample_sequences = train_sequences

            # create validation dataseqt
            self.val_dataset = DigitalTyphoonAnalysis(**self.kwargs)
            self.val_dataset.sample_sequences = val_sequences

        if stage in ["test"]:
            self.test_dataset = DigitalTyphoonAnalysis(**self.kwargs)
            self.test_dataset.sample_sequences = test_sequences