#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

index = [
    'TorchGeo',
    'eo-learn',
    'Raster Vision',
    'DeepForest',
    'samgeo',
    'TerraTorch',
    'SITS',
    'srai',
    'scikit-eo',
    'geo-bench',
    'GeoAI',
    'OTBTF',
    'GeoDeep',
]

name_to_pypi = {
    'TorchGeo': 'torchgeo',
    'eo-learn': 'eo-learn',
    'Raster Vision': 'rastervision',
    'DeepForest': 'deepforest',
    'samgeo': 'segment-geospatial',
    'TerraTorch': 'terratorch',
    'SITS': 'pysits',
    'srai': 'srai',
    'scikit-eo': 'scikeo',
    'geo-bench': 'geobench',
    'GeoAI': 'geoai-py',
    'GeoDeep': 'geodeep',
}
name_to_cran = {'SITS': 'sits'}
name_to_conda = {
    'TorchGeo': 'torchgeo',
    'eo-learn': 'eo-learn',
    'Raster Vision': 'rastervision-core',
    'DeepForest': 'deepforest',
    'samgeo': 'segment-geospatial',
    'SITS': 'r-sits',
    'GeoAI': 'geoai',
}
name_to_github = {
    'TorchGeo': ('torchgeo', 'torchgeo'),
    'eo-learn': ('sentinel-hub', 'eo-learn'),
    'Raster Vision': ('azavea', 'raster-vision'),
    'DeepForest': ('weecology', 'DeepForest'),
    'samgeo': ('opengeos', 'segment-geospatial'),
    'TerraTorch': ('IBM', 'terratorch'),
    'SITS': ('e-sensing', 'sits'),
    'srai': ('kraina-ai', 'srai'),
    'scikit-eo': ('yotarazona', 'scikit-eo'),
    'geo-bench': ('ServiceNow', 'geo-bench'),
    'GeoAI': ('opengeos', 'geoai'),
    'OTBTF': ('remicres', 'otbtf'),
    'GeoDeep': ('uav4geo', 'GeoDeep'),
}
name_to_codecov = {
    'TorchGeo': ('gh', 'torchgeo', 'torchgeo'),
    'eo-learn': ('gh', 'sentinel-hub', 'eo-learn'),
    'SITS': ('gh', 'e-sensing', 'sits'),
    'srai': ('gh', 'kraina-ai', 'srai'),
}
