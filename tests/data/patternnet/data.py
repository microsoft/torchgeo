# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    'PatternNet/images/airplane/airplane001.jpg',
    'PatternNet/images/airplane/airplane002.jpg',
    'PatternNet/images/airplane/airplane003.jpg',
    'PatternNet/images/bridge/bridge001.jpg',
    'PatternNet/images/bridge/bridge002.jpg',
    'PatternNet/images/bridge/bridge003.jpg',
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
    'PatternNet.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'PatternNet/images/airplane/airplane001.jpg',
        'PatternNet/images/airplane/airplane002.jpg',
        'PatternNet/images/airplane/airplane003.jpg',
        'PatternNet/images/bridge/bridge001.jpg',
        'PatternNet/images/bridge/bridge002.jpg',
        'PatternNet/images/bridge/bridge003.jpg',
    ]:
        archive.write(member, member)
