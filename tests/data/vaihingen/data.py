# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile

from tests.data.utils import write_image

for path in [
    'top/top_mosaic_09cm_area1.tif',
    'top/top_mosaic_09cm_area11.tif',
    'top/top_mosaic_09cm_area24.tif',
    'top/top_mosaic_09cm_area6.tif',
    'top_mosaic_09cm_area1.tif',
    'top_mosaic_09cm_area11.tif',
    'top_mosaic_09cm_area24.tif',
    'top_mosaic_09cm_area6.tif',
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
    'ISPRS_semantic_labeling_Vaihingen.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'top/top_mosaic_09cm_area11.tif',
        'top/top_mosaic_09cm_area6.tif',
        'top/top_mosaic_09cm_area1.tif',
        'top/top_mosaic_09cm_area24.tif',
    ]:
        archive.write(member, member)

with zipfile.ZipFile(
    'ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip',
    'w',
    compression=zipfile.ZIP_DEFLATED,
) as archive:
    for member in [
        'top_mosaic_09cm_area1.tif',
        'top_mosaic_09cm_area11.tif',
        'top_mosaic_09cm_area24.tif',
        'top_mosaic_09cm_area6.tif',
    ]:
        archive.write(member, member)
