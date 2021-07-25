import os

import pytest

from torchgeo.trainers import SEN12MSDataModule


@pytest.mark.parametrize("band_set", ["all", "s1", "s2-all", "s2-reduced"])
def test_band_set(band_set: str) -> None:
    dm = SEN12MSDataModule(os.path.join("tests", "data"), 0, band_set)
    dm.prepare_data()
    dm.all_train_dataset[0]
