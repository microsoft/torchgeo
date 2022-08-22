#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

from PIL import Image

SIZE = 64  # image width/height

metadatas = [
    {
        "filename": "train.zip",
        "directory": "train",
        "subdirs": [
            "nebraska_20170108t002112",
            "bangladesh_20170314t115609",
            "northal_20190302t234651",
        ],
    },
    {
        "filename": "val_with_ref_labels.zip",
        "directory": "test",
        "subdirs": [
            "florence_20180510t231343",
            "florence_20180522t231344",
            "florence_20190302t234651",
        ],
    },
    {
        "filename": "test_without_ref_labels.zip",
        "directory": "test_internal",
        "subdirs": [
            "redrivernorth_20190104t002247",
            "redrivernorth_20190116t002247",
            "redrivernorth_20190302t234651",
        ],
    },
]

tiles = ["vh", "vv", "water_body_label", "flood_label"]

for metadata in metadatas:
    filename = metadata["filename"]
    directory = metadata["directory"]

    # Remove old data
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # Create images
    for subdir in metadata["subdirs"]:
        for tile in tiles:
            if directory == "test_internal" and tile == "flood_label":
                continue

            fn = f"{subdir}_x-0_y-0"
            if tile in ["vh", "vv"]:
                fn += f"_{tile}"
            fn += ".png"
            fd = os.path.join(directory, subdir, "tiles", tile)
            os.makedirs(fd)

            img = Image.new("RGB", (SIZE, SIZE))
            img.save(os.path.join(fd, fn))

    # Compress data
    shutil.make_archive(filename.replace(".zip", ""), "zip", ".", directory)

    # Compute checksums
    with open(filename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(repr(filename) + ":", repr(md5) + ",")
