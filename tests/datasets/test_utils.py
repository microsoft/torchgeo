from pathlib import Path

import torch

from torchgeo.datasets import BoundingBox, collate_dict
from torchgeo.datasets.utils import working_dir


def test_bounding_box() -> None:
    bbox = BoundingBox(0, 1, 2, 3, 4, 5)
    assert bbox.minx == 0
    assert bbox.maxx == 1
    assert bbox.miny == 2
    assert bbox.maxy == 3
    assert bbox.mint == 4
    assert bbox.maxt == 5


def test_collate_dict() -> None:
    samples = [
        {
            "foo": torch.tensor(1),  # type: ignore[attr-defined]
            "bar": torch.tensor(2),  # type: ignore[attr-defined]
        },
        {
            "foo": torch.tensor(3),  # type: ignore[attr-defined]
            "bar": torch.tensor(4),  # type: ignore[attr-defined]
        },
    ]
    sample = collate_dict(samples)
    assert torch.allclose(  # type: ignore[attr-defined]
        sample["foo"], torch.tensor([1, 3])  # type: ignore[attr-defined]
    )
    assert torch.allclose(  # type: ignore[attr-defined]
        sample["bar"], torch.tensor([2, 4])  # type: ignore[attr-defined]
    )


def test_existing_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "foo" / "bar"
    subdir.mkdir(parents=True)

    assert subdir.exists()

    with working_dir(str(subdir)):
        assert subdir.cwd() == subdir


def test_nonexisting_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "foo" / "bar"

    assert not subdir.exists()

    with working_dir(str(subdir), create=True):
        assert subdir.cwd() == subdir
