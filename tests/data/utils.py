# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

_rng = np.random.default_rng(0)


def write_image(path: str, profile: dict[str, Any]) -> None:
    """Write a synthetic image with the required dimensions and georeferencing."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    shape = (profile['count'], profile['height'], profile['width'])
    data = _rng.integers(0, 2, size=shape).astype(profile['dtype'])
    if destination.suffix.lower() in {'.jpg', '.png'}:
        data = data.transpose(1, 2, 0)
        if data.shape[-1] == 1:
            data = data[..., 0]
        Image.fromarray(data).save(destination)
    else:
        profile['transform'] = Affine(*profile['transform'])
        with rasterio.open(destination, 'w', **profile) as dst:
            dst.write(data)
