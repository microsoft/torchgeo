#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import hashlib
import json
import zipfile

SIZE = 100


def create_data_file(zipfilename):
    meta_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [0.0, 0.0],
                        [0.0, SIZE],
                        [SIZE, SIZE],
                        [SIZE, 0.0],
                        [0.0, 0.0],
                    ]],
                },
                "properties": {
                    "EC_hcat_c": "1000000010",
                },
            },
        ],
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::31287",
            },
        },
    }
    return meta_data


def create_csv(fname):
    with open(fname, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["HCAT2_code"])
        writer.writeheader()
        writer.writerow({
            "HCAT2_code": "1000000000",
        })
        writer.writerow({
            "HCAT2_code": "1000000010",
        })


if __name__ == "__main__":
    csvname = "HCAT2.csv"
    dataname = "AA.geojson"
    zipfilename = "AA.zip"

    # create crop type data
    geojson_data = create_data_file(zipfilename)
    with open(dataname, "w") as fp:
        json.dump(geojson_data, fp)

    # archive the geojson to zip
    with zipfile.ZipFile(zipfilename, 'w') as zipf:
        zipf.write(dataname)

    # create csv metadata file
    create_csv(csvname)

    # Compute checksums
    with open(zipfilename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{zipfilename}: {md5}")
    with open(csvname, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{csvname}: {md5}")
