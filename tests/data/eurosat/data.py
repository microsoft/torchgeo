# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

write_image(
    'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/AnnualCrop/AnnualCrop_1.tif',
    {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'count': 13,
        'height': 64,
        'width': 64,
        'crs': 'EPSG:32635',
        'transform': (
            10.00570688714736,
            0.0,
            624602.2348443292,
            0.0,
            -9.994088099999352,
            4877286.033637,
        ),
        'compress': 'lzw',
    },
)

write_image(
    'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/Forest/Forest_1.tif',
    {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'count': 13,
        'height': 64,
        'width': 64,
        'crs': 'EPSG:32632',
        'transform': (
            10.005196165071764,
            0.0,
            533043.6787387297,
            0.0,
            -10.008080018868775,
            5567792.427645094,
        ),
        'compress': 'lzw',
    },
)

with zipfile.ZipFile(
    'EuroSAT100.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/AnnualCrop/AnnualCrop_1.tif',
        'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/Forest/Forest_1.tif',
    ]:
        archive.write(member, member)

with zipfile.ZipFile(
    'EuroSATallBands.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/AnnualCrop/AnnualCrop_1.tif',
        'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/Forest/Forest_1.tif',
    ]:
        archive.write(member, member)
