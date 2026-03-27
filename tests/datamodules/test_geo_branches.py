# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for branch coverage of _valid_attribute in datamodules/geo.py.

Relates to: https://github.com/microsoft/torchgeo/pull/3549
"""


import pytest
from torch.utils.data import Dataset

from torchgeo.datamodules.geo import BaseDataModule
from torchgeo.datamodules.utils import MisconfigurationException
from torchgeo.datasets.utils import Sample


class DummyDataset(Dataset[Sample]):
    """Minimal dataset for testing."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Sample:
        return {}


class ConcreteDataModule(BaseDataModule):
    """Minimal concrete subclass for testing _valid_attribute."""

    def setup(self, stage: str) -> None:
        """No-op setup."""


def test_valid_attribute_returns_first_valid() -> None:
    """Test that _valid_attribute returns the first non-None, non-empty attribute."""
    dm = ConcreteDataModule(dataset_class=DummyDataset)
    dm.__dict__['attr1'] = None
    dm.__dict__['attr2'] = [1, 2, 3]
    result = dm._valid_attribute('attr1', 'attr2')
    assert result == [1, 2, 3]


def test_valid_attribute_raises_when_empty() -> None:
    """Test that _valid_attribute raises when attribute is empty."""
    dm = ConcreteDataModule(dataset_class=DummyDataset)
    dm.__dict__['attr1'] = []
    with pytest.raises(MisconfigurationException, match='has length 0'):
        dm._valid_attribute('attr1')


def test_valid_attribute_raises_when_all_none() -> None:
    """Test that _valid_attribute raises when all attributes are None."""
    dm = ConcreteDataModule(dataset_class=DummyDataset)
    dm.__dict__['attr1'] = None
    dm.__dict__['attr2'] = None
    with pytest.raises(MisconfigurationException, match='setup must define'):
        dm._valid_attribute('attr1', 'attr2')
