# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile
from pathlib import Path

from tests.data.utils import write_image

for path in ['vision/airport/00063.jpg', 'vision/beach/00093.jpg']:
    write_image(
        path,
        {
            'driver': 'JPEG',
            'dtype': 'uint8',
            'count': 3,
            'height': 1,
            'width': 1,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

import numpy as np
from scipy.io import wavfile

path = Path('sound/airport/00063.wav')
path.parent.mkdir(parents=True, exist_ok=True)
wavfile.write(path, 22050, np.zeros((1,), dtype='int32'))

import numpy as np
from scipy.io import wavfile

path = Path('sound/beach/00093.wav')
path.parent.mkdir(parents=True, exist_ok=True)
wavfile.write(path, 22050, np.zeros((1,), dtype='int32'))

with zipfile.ZipFile(
    'ADVANCE_sound.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['sound/airport/00063.wav', 'sound/beach/00093.wav']:
        archive.write(member, member)

with zipfile.ZipFile(
    'ADVANCE_vision.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in ['vision/airport/00063.jpg', 'vision/beach/00093.jpg']:
        archive.write(member, member)
