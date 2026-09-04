# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    'GID/ann_dir/train/GF2_PMS1__L1A0000564539-MSS1_15label.png',
    'GID/ann_dir/train/GF2_PMS1__L1A0000575925-MSS1_15label.png',
    'GID/ann_dir/val/GF2_PMS1__L1A0001064454-MSS1_15label.png',
    'GID/ann_dir/val/GF2_PMS1__L1A0001118839-MSS1_15label.png',
]:
    write_image(
        path,
        {
            'driver': 'PNG',
            'dtype': 'uint8',
            'count': 1,
            'height': 1,
            'width': 1,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

for path in [
    'GID/img_dir/test/GF2_PMS1__L1A0000708367-MSS1.tif',
    'GID/img_dir/test/GF2_PMS1__L1A0001344822-MSS1.tif',
    'GID/img_dir/train/GF2_PMS1__L1A0000564539-MSS1.tif',
    'GID/img_dir/train/GF2_PMS1__L1A0000575925-MSS1.tif',
    'GID/img_dir/val/GF2_PMS1__L1A0001064454-MSS1.tif',
    'GID/img_dir/val/GF2_PMS1__L1A0001118839-MSS1.tif',
]:
    write_image(
        path,
        {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 3,
            'height': 1,
            'width': 1,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            'compress': 'lzw',
        },
    )

with zipfile.ZipFile('gid-15.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for member in [
        'GID/ann_dir/val/GF2_PMS1__L1A0001118839-MSS1_15label.png',
        'GID/ann_dir/val/GF2_PMS1__L1A0001064454-MSS1_15label.png',
        'GID/ann_dir/train/GF2_PMS1__L1A0000564539-MSS1_15label.png',
        'GID/ann_dir/train/GF2_PMS1__L1A0000575925-MSS1_15label.png',
        'GID/img_dir/val/GF2_PMS1__L1A0001118839-MSS1.tif',
        'GID/img_dir/val/GF2_PMS1__L1A0001064454-MSS1.tif',
        'GID/img_dir/test/GF2_PMS1__L1A0000708367-MSS1.tif',
        'GID/img_dir/test/GF2_PMS1__L1A0001344822-MSS1.tif',
        'GID/img_dir/train/GF2_PMS1__L1A0000575925-MSS1.tif',
        'GID/img_dir/train/GF2_PMS1__L1A0000564539-MSS1.tif',
    ]:
        archive.write(member, member)
