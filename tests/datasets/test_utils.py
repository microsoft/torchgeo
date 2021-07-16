from pathlib import Path
from typing import Tuple

import pytest
import torch

from rasterio.crs import CRS
from torchgeo.datasets import BoundingBox, collate_dict
from torchgeo.datasets.utils import working_dir


class TestBoundingBox:
    def test_new_init(self) -> None:
        bbox = BoundingBox(0, 1, 2, 3, 4, 5)

        assert bbox.minx == 0
        assert bbox.maxx == 1
        assert bbox.miny == 2
        assert bbox.maxy == 3
        assert bbox.mint == 4
        assert bbox.maxt == 5

        assert bbox[0] == 0
        assert bbox[-1] == 5
        assert bbox[1:3] == (1, 2)

    def test_repr_str(self) -> None:
        bbox = BoundingBox(0, 1, 2.0, 3.0, -5, -4)
        expected = "BoundingBox(minx=0, maxx=1, miny=2.0, maxy=3.0, mint=-5, maxt=-4)"
        assert repr(bbox) == expected
        assert str(bbox) == expected

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # bbox1 strictly within bbox2
            ((-1, 2, -1, 2, -1, 2), True),
            # bbox2 strictly within bbox1
            ((0.25, 0.75, 0.25, 0.75, 0.25, 0.75), True),
            # One corner of bbox1 within bbox2
            ((0.5, 1.5, 0.5, 1.5, 0.5, 1.5), True),
            ((0.5, 1.5, -0.5, 0.5, 0.5, 1.5), True),
            ((0.5, 1.5, 0.5, 1.5, -0.5, 0.5), True),
            ((0.5, 1.5, -0.5, 0.5, -0.5, 0.5), True),
            ((-0.5, 0.5, 0.5, 1.5, 0.5, 1.5), True),
            ((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5), True),
            ((-0.5, 0.5, 0.5, 1.5, -0.5, 0.5), True),
            ((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), True),
            # No overlap
            ((0.5, 1.5, 0.5, 1.5, 2, 3), False),
            ((0.5, 1.5, 2, 3, 0.5, 1.5), False),
            ((2, 3, 0.5, 1.5, 0.5, 1.5), False),
            ((2, 3, 2, 3, 2, 3), False),
        ],
    )
    def test_intersects(
        self,
        test_input: Tuple[float, float, float, float, float, float],
        expected: bool,
    ) -> None:
        bbox1 = BoundingBox(0, 1, 0, 1, 0, 1)
        bbox2 = BoundingBox(*test_input)
        assert bbox1.intersects(bbox2) == bbox2.intersects(bbox1) == expected

    def test_invalid_x(self) -> None:
        with pytest.raises(
            ValueError, match="Bounding box is invalid: 'minx=1' > 'maxx=0'"
        ):
            BoundingBox(1, 0, 2, 3, 4, 5)

    def test_invalid_y(self) -> None:
        with pytest.raises(
            ValueError, match="Bounding box is invalid: 'miny=3' > 'maxy=2'"
        ):
            BoundingBox(0, 1, 3, 2, 4, 5)

    def test_invalid_t(self) -> None:
        with pytest.raises(
            ValueError, match="Bounding box is invalid: 'mint=5' > 'maxt=4'"
        ):
            BoundingBox(0, 1, 2, 3, 5, 4)


def test_collate_dict() -> None:
    samples = [
        {
            "foo": torch.tensor(1),  # type: ignore[attr-defined]
            "bar": torch.tensor(2),  # type: ignore[attr-defined]
            "crs": CRS.from_epsg(3005),
        },
        {
            "foo": torch.tensor(3),  # type: ignore[attr-defined]
            "bar": torch.tensor(4),  # type: ignore[attr-defined]
            "crs": CRS.from_epsg(3005),
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
