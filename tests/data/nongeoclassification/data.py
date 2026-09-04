# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


from tests.data.utils import write_image

for path in ['class0/001.jpg', 'class1/001.jpg']:
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
