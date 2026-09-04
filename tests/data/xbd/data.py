# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import tarfile

from tests.data.utils import write_image

for path in [
    'test/images/hurricane-michael_00000105_post_disaster.png',
    'test/images/hurricane-michael_00000105_pre_disaster.png',
    'test/images/hurricane-michael_00000450_post_disaster.png',
    'test/images/hurricane-michael_00000450_pre_disaster.png',
    'train/images/hurricane-harvey_00000072_post_disaster.png',
    'train/images/hurricane-harvey_00000072_pre_disaster.png',
    'train/images/hurricane-harvey_00000471_post_disaster.png',
    'train/images/hurricane-harvey_00000471_pre_disaster.png',
]:
    write_image(
        path,
        {
            'driver': 'PNG',
            'dtype': 'uint8',
            'count': 3,
            'height': 128,
            'width': 128,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

for path in [
    'test/targets/hurricane-michael_00000105_post_disaster_target.png',
    'test/targets/hurricane-michael_00000105_pre_disaster_target.png',
    'test/targets/hurricane-michael_00000450_post_disaster_target.png',
    'test/targets/hurricane-michael_00000450_pre_disaster_target.png',
    'train/targets/hurricane-harvey_00000072_post_disaster_target.png',
    'train/targets/hurricane-harvey_00000072_pre_disaster_target.png',
    'train/targets/hurricane-harvey_00000471_post_disaster_target.png',
    'train/targets/hurricane-harvey_00000471_pre_disaster_target.png',
]:
    write_image(
        path,
        {
            'driver': 'PNG',
            'dtype': 'uint8',
            'count': 1,
            'height': 128,
            'width': 128,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        },
    )

with tarfile.open('test_images_labels_targets.tar.gz', 'w:gz') as archive:
    for member in [
        'test/images/hurricane-michael_00000105_pre_disaster.png',
        'test/images/hurricane-michael_00000105_post_disaster.png',
        'test/images/hurricane-michael_00000450_pre_disaster.png',
        'test/images/hurricane-michael_00000450_post_disaster.png',
        'test/targets/hurricane-michael_00000105_pre_disaster_target.png',
        'test/targets/hurricane-michael_00000105_post_disaster_target.png',
        'test/targets/hurricane-michael_00000450_pre_disaster_target.png',
        'test/targets/hurricane-michael_00000450_post_disaster_target.png',
    ]:
        archive.add(member, member)

with tarfile.open('train_images_labels_targets.tar.gz', 'w:gz') as archive:
    for member in [
        'train/images/hurricane-harvey_00000072_pre_disaster.png',
        'train/images/hurricane-harvey_00000072_post_disaster.png',
        'train/images/hurricane-harvey_00000471_pre_disaster.png',
        'train/images/hurricane-harvey_00000471_post_disaster.png',
        'train/targets/hurricane-harvey_00000072_pre_disaster_target.png',
        'train/targets/hurricane-harvey_00000072_post_disaster_target.png',
        'train/targets/hurricane-harvey_00000471_pre_disaster_target.png',
        'train/targets/hurricane-harvey_00000471_post_disaster_target.png',
    ]:
        archive.add(member, member)
