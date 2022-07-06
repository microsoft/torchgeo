#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

filename = "observations-012345.csv"

# User can select which columns to export. The following are the default columns.
# Not all columns may exist in the actual dataset.
size = 4
data = {
    "id": [""] * size,
    "observed_on_string": [""] * size,
    "observed_on": ["", "", "2022-05-07", "2022-05-07"],
    "time_observed_at": ["", "", "", "2022-05-07 11:02:53 +0100"],
    "time_zone": ["Central Time (US & Canada)"] * size,
    "user_id": [123] * size,
    "user_login": ["darwin"] * size,
    "created_at": ["2022-05-07 11:02:53 +0100"] * size,
    "updated_at": ["2022-05-07 11:02:53 +0100"] * size,
    "quality_grade": ["research"] * size,
    "license": ["CCO"] * size,
    "url": ["https://inaturalist.org/observations/123"] * size,
    "image_url": [
        "https://inaturalist-open-data.s3.amazonaws.com/photos/123/medium.jpg"
    ]
    * size,
    "sound_url": ["https://static.inaturalist.org/sounds/123.m4a?123"] * size,
    "tag_list": ["Chicago"] * size,
    "description": [""] * size,
    "num_identification_agreements": [1] * size,
    "num_identification_disagreements": [0] * size,
    "captive_cultivated": ["false"] * size,
    "oauth_application_id": [""] * size,
    "place_guess": ["Chicago"] * size,
    "latitude": [41.881832] * size,
    "longitude": [""] + [-87.623177] * (size - 1),
    "positional_accuracy": [5] * size,
    "private_place_guess": [""] * size,
    "private_latitude": [""] * size,
    "private_longitude": [""] * size,
    "public_positional_accuracy": [5] * size,
    "geoprivacy": [""] * size,
    "taxon_geoprivacy": [""] * size,
    "coordinates_obscured": ["false"] * size,
    "positioning_method": ["gps"] * size,
    "positioning_device": ["gps"] * size,
    "species_guess": ["Homo sapiens"] * size,
    "scientific_name": ["Homo sapiens"] * size,
    "common_name": ["human"] * size,
    "iconic_taxon_name": ["Animalia"] * size,
    "taxon_id": [123] * size,
}

df = pd.DataFrame(data)
df.to_csv(filename, index=False)
