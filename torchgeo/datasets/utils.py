import contextlib
import os
from typing import Dict, Iterator, List, Tuple, Union

import torch
from rasterio.crs import CRS
from torch import Tensor


class BoundingBox(Tuple[float, float, float, float, float, float]):
    """Data class for indexing spatiotemporal data.

    Attributes:
        minx (float): western boundary
        maxx (float): eastern boundary
        miny (float): southern boundary
        maxy (float): northern boundary
        mint (float): earliest boundary
        maxt (float): latest boundary
    """

    def __new__(
        cls,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        mint: float,
        maxt: float,
    ) -> "BoundingBox":
        """Create a new instance of BoundingBox.

        Parameters:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary

        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)
        """
        if minx > maxx:
            raise ValueError(f"Bounding box is invalid: 'minx={minx}' > 'maxx={maxx}'")
        if miny > maxy:
            raise ValueError(f"Bounding box is invalid: 'miny={miny}' > 'maxy={maxy}'")
        if mint > maxt:
            raise ValueError(f"Bounding box is invalid: 'mint={mint}' > 'maxt={maxt}'")

        # Using super() doesn't work with mypy, see:
        # https://stackoverflow.com/q/60611012/5828163
        return tuple.__new__(cls, [minx, maxx, miny, maxy, mint, maxt])

    def __init__(
        self,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        mint: float,
        maxt: float,
    ) -> None:
        """Initialize a new instance of BoundingBox.

        Parameters:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary
        """
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.mint = mint
        self.maxt = maxt

    def __repr__(self) -> str:
        """Return the formal string representation of the object.

        Returns:
            formal string representation
        """
        return (
            f"{self.__class__.__name__}(minx={self.minx}, maxx={self.maxx}, "
            f"miny={self.miny}, maxy={self.maxy}, mint={self.mint}, maxt={self.maxt})"
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether or not two bounding boxes intersect.

        Parameters:
            other: another bounding box

        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.minx < other.maxx
            and self.maxx > other.minx
            and self.miny < other.maxy
            and self.maxy > other.miny
            and self.mint < other.maxt
            and self.maxt > other.mint
        )


@contextlib.contextmanager
def working_dir(dirname: str, create: bool = False) -> Iterator[None]:
    """Context manager for changing directories.

    Parameters:
        dirname: directory to temporarily change to
        create: if True, create the destination directory
    """
    if create:
        os.makedirs(dirname, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(dirname)

    try:
        yield
    finally:
        os.chdir(cwd)


def collate_dict(
    samples: List[Dict[str, Union[Tensor, CRS]]]
) -> Dict[str, Union[Tensor, CRS]]:
    """Merge a list of samples for form a mini-batch of Tensors.

    Parameters:
        samples: list of samples

    Returns:
        a single sample
    """
    collated = {}
    for key, value in samples[0].items():
        if isinstance(value, CRS):
            collated[key] = value
        elif isinstance(value, Tensor):
            collated[key] = torch.stack([sample[key] for sample in samples])
    return collated
