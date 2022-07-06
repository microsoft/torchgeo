#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SIZE = 32

np.random.seed(0)

PATHS = {
    "images": [
        "Bebehojd_20190527/Images/B01_0004.JPG",
        "Bebehojd_20190527/Images/B01_0005.JPG",
    ],
    "annotations": [
        "Bebehojd_20190527/Annotations/B01_0004.xml",
        "Bebehojd_20190527/Annotations/B01_0005.xml",
    ],
    "labels": [True, False],
}


def create_annotation(path: str) -> None:
    root = ET.Element("annotation")

    ET.SubElement(root, "filename").text = os.path.basename(path)

    size = ET.SubElement(root, "size")

    ET.SubElement(size, "width").text = str(SIZE)
    ET.SubElement(size, "height").text = str(SIZE)
    ET.SubElement(size, "depth").text = str(3)

    for label in PATHS["labels"]:
        annotation = ET.SubElement(root, "object")

        if label:
            ET.SubElement(annotation, "damage").text = "other"

        bbox = ET.SubElement(annotation, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(0 + int(SIZE / 4))
        ET.SubElement(bbox, "ymin").text = str(0 + int(SIZE / 4))
        ET.SubElement(bbox, "xmax").text = str(SIZE - int(SIZE / 4))
        ET.SubElement(bbox, "ymax").text = str(SIZE - int(SIZE / 4))

    tree = ET.ElementTree(root)
    tree.write(path)


def create_file(path: str) -> None:
    Z = np.random.rand(SIZE, SIZE, 3) * 255
    img = Image.fromarray(Z.astype("uint8")).convert("RGB")
    img.save(path)


if __name__ == "__main__":
    data_root = "Data_Set_Larch_Casebearer"
    # remove old data
    if os.path.isdir(data_root):
        shutil.rmtree(data_root)
    else:
        os.makedirs(data_root)

    for path in PATHS["images"]:
        os.makedirs(os.path.join(data_root, os.path.dirname(path)), exist_ok=True)
        create_file(os.path.join(data_root, path))

    for path in PATHS["annotations"]:
        os.makedirs(os.path.join(data_root, os.path.dirname(path)), exist_ok=True)
        create_annotation(os.path.join(data_root, path))

    # compress data
    shutil.make_archive(data_root, "zip", ".", data_root)

    # Compute checksums
    with open(data_root + ".zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{data_root}: {md5}")
