# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    'UCMerced_LandUse/Images/agricultural/agricultural00.tif',
    'UCMerced_LandUse/Images/airplane/airplane00.tif',
]:
    write_image(
        path,
        {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 3,
            'height': 256,
            'width': 256,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            'compress': 'lzw',
        },
    )

write_image(
    'UCMerced_LandUse/Images/agricultural/agricultural01.tif',
    {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 3,
        'height': 249,
        'width': 256,
        'crs': None,
        'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        'compress': 'lzw',
    },
)

write_image(
    'UCMerced_LandUse/Images/agricultural/agricultural02.tif',
    {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 3,
        'height': 247,
        'width': 247,
        'crs': None,
        'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        'compress': 'lzw',
    },
)

with zipfile.ZipFile(
    'UCMerced_LandUse.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'UCMerced_LandUse/Images/agricultural/agricultural00.tif',
        'UCMerced_LandUse/Images/agricultural/agricultural01.tif',
        'UCMerced_LandUse/Images/agricultural/agricultural02.tif',
        'UCMerced_LandUse/Images/airplane/airplane00.tif',
    ]:
        archive.write(member, member)
