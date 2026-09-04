# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    'NWPU-RESISC45/airplane/airplane_001.jpg',
    'NWPU-RESISC45/airplane/airplane_002.jpg',
    'NWPU-RESISC45/airplane/airplane_003.jpg',
    'NWPU-RESISC45/airport/airport_001.jpg',
    'NWPU-RESISC45/airport/airport_002.jpg',
    'NWPU-RESISC45/airport/airport_003.jpg',
    'NWPU-RESISC45/baseball_diamond/baseball_diamond_001.jpg',
    'NWPU-RESISC45/baseball_diamond/baseball_diamond_002.jpg',
    'NWPU-RESISC45/baseball_diamond/baseball_diamond_003.jpg',
]:
    write_image(
        path,
        {
            'driver': 'JPEG',
            'dtype': 'uint8',
            'count': 3,
            'height': 256,
            'width': 256,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

with zipfile.ZipFile(
    'NWPU-RESISC45.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'NWPU-RESISC45/baseball_diamond/baseball_diamond_003.jpg',
        'NWPU-RESISC45/baseball_diamond/baseball_diamond_002.jpg',
        'NWPU-RESISC45/baseball_diamond/baseball_diamond_001.jpg',
        'NWPU-RESISC45/airplane/airplane_001.jpg',
        'NWPU-RESISC45/airplane/airplane_002.jpg',
        'NWPU-RESISC45/airplane/airplane_003.jpg',
        'NWPU-RESISC45/airport/airport_001.jpg',
        'NWPU-RESISC45/airport/airport_002.jpg',
        'NWPU-RESISC45/airport/airport_003.jpg',
    ]:
        archive.write(member, member)
