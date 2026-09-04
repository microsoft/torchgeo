# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    '4_Ortho_RGBIR/top_potsdam_2_10_RGBIR.tif',
    '4_Ortho_RGBIR/top_potsdam_2_11_RGBIR.tif',
    '4_Ortho_RGBIR/top_potsdam_5_15_RGBIR.tif',
    '4_Ortho_RGBIR/top_potsdam_6_15_RGBIR.tif',
]:
    write_image(
        path,
        {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 4,
            'height': 2,
            'width': 2,
            'crs': 'EPSG:32633',
            'transform': (0.05, 0.0, 366976.5, 0.0, -0.05, 5808562.6),
            'compress': 'lzw',
        },
    )

for path in [
    'top_potsdam_2_10_label.tif',
    'top_potsdam_2_11_label.tif',
    'top_potsdam_5_15_label.tif',
    'top_potsdam_6_15_label.tif',
]:
    write_image(
        path,
        {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 3,
            'height': 2,
            'width': 2,
            'crs': None,
            'transform': (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            'compress': 'lzw',
        },
    )

with zipfile.ZipFile(
    '4_Ortho_RGBIR.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        '4_Ortho_RGBIR/top_potsdam_5_15_RGBIR.tif',
        '4_Ortho_RGBIR/top_potsdam_2_10_RGBIR.tif',
        '4_Ortho_RGBIR/top_potsdam_2_11_RGBIR.tif',
        '4_Ortho_RGBIR/top_potsdam_6_15_RGBIR.tif',
    ]:
        archive.write(member, member)

with zipfile.ZipFile(
    '5_Labels_all.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'top_potsdam_2_10_label.tif',
        'top_potsdam_2_11_label.tif',
        'top_potsdam_5_15_label.tif',
        'top_potsdam_6_15_label.tif',
    ]:
        archive.write(member, member)
