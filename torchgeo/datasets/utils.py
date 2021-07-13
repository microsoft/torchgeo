import contextlib
import os
from typing import Dict, Iterator, List, NamedTuple, Union

import torch
from torch import Tensor


class BoundingBox(NamedTuple):
    """Named tuple for indexing spatiotemporal data."""

    minx: Union[int, float]
    maxx: Union[int, float]
    miny: Union[int, float]
    maxy: Union[int, float]
    mint: Union[int, float]
    maxt: Union[int, float]


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


def collate_dict(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Merge a list of samples for form a mini-batch of Tensors.

    Parameters:
        samples: list of samples

    Returns:
        a single sample
    """
    collated = {}
    for key in samples[0]:
        collated[key] = torch.stack([sample[key] for sample in samples])
    return collated
