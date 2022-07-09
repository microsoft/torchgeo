import pytest
from matplotlib import pyplot as plt

from torchgeo.datasets import neonspecies


@pytest.fixture()
def dataset(tmpdir):
    dataset = neonspecies.NEONTreeSpecies(download=True, root=tmpdir)
    return dataset


def test_getitem(dataset):
    items = dataset.__getitem__(0)
    assert len(items) == 5
    assert list(items.keys()) == ["image", "hsi", "chm", "metadata", "label"]


def test_plot(dataset):
    items = dataset.__getitem__(10)
    dataset.plot(items)
    plt.close()
