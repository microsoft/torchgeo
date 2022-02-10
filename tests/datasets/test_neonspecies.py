from torchgeo.datasets import neonspecies
import pytest

@pytest.fixture()
def dataset(tmpdir):
    dataset = neonspecies.NEONTreeSpecies(download=True, root=tmpdir)
    
    return dataset

def test_getitem(dataset):
    items = dataset.__getitem__(0)
    assert len(items) == 5
    assert items.keys() == ["image","hsi","chm","points","label"]
    
    
    