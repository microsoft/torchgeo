# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable

"""Test Digital Typhoon Datamodule."""


import pytest

from torchgeo.datamodules import DigitalTyphoonDataModule
from torchgeo.datasets.digital_typhoon import DigitalTyphoon, _SampleSequenceDict

pytest.importorskip('h5py', minversion='3.10')


class TestDigitalTyphoonDataModule:
    def test_invalid_param_config(self, test_data: Callable[[str], str]) -> None:
        with pytest.raises(AssertionError, match='Please choose from'):
            DigitalTyphoonDataModule(
                root=test_data('digital_typhoon'),
                split_by='invalid',
                batch_size=2,
                num_workers=0,
            )

    @pytest.mark.parametrize('split_by', ['time', 'typhoon_id'])
    def test_split_dataset(
        self, split_by: str, test_data: Callable[[str], str]
    ) -> None:
        dm = DigitalTyphoonDataModule(
            root=test_data('digital_typhoon'),
            split_by=split_by,
            batch_size=2,
            num_workers=0,
        )
        dataset = DigitalTyphoon(root=test_data('digital_typhoon'))
        train_indices, val_indices = dm._split_dataset(dataset.sample_sequences)
        train_sequences, val_sequences = (
            [dataset.sample_sequences[i] for i in train_indices],
            [dataset.sample_sequences[i] for i in val_indices],
        )

        if split_by == 'time':

            def find_max_time_per_id(
                split_sequences: list[_SampleSequenceDict],
            ) -> dict[str, int]:
                # Find the maximum value of each id in train_sequences
                max_values: dict[str, int] = {}
                for seq in split_sequences:
                    id: str = str(seq['id'])
                    value: int = max(seq['seq_id'])
                    if id not in max_values or value > max_values[id]:
                        max_values[id] = value
                return max_values

            train_max_values = find_max_time_per_id(train_sequences)
            val_max_values = find_max_time_per_id(val_sequences)
            # Assert that each max value in train_max_values is lower
            # than in val_max_values for each key id
            for id, max_value in train_max_values.items():
                assert id not in val_max_values or max_value < val_max_values[id], (
                    f'Max value for id {id} in train is not lower than in validation.'
                )
        else:
            train_ids = {seq['id'] for seq in train_sequences}
            val_ids = {seq['id'] for seq in val_sequences}

            # Assert that the intersection between train_ids and val_ids is empty
            assert len(train_ids & val_ids) == 0, (
                'Train and validation datasets have overlapping ids.'
            )
