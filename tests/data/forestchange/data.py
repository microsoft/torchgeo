#!/usr/bin/env python3

import json
import os
import shutil
import zipfile

import numpy as np
from PIL import Image

np.random.seed(0)

root = "Forest-Change-dataset"
splits = ["train", "val", "test"]
directories = ["A", "B", "label"]
N = 2


def create_image(path: str) -> None:
    arr = np.random.randint(255, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).convert("RGB").save(path)


def create_mask(path: str) -> None:
    arr = np.random.randint(2, size=(32, 32), dtype=np.uint8) * 255
    Image.fromarray(arr).convert("L").save(path)


def create_captions() -> dict:
    return {
        "images": [
            {
                "filename": f"{split}_{i:06d}.png",
                "filepath": split,
                "split": split,
                "sentences": [
                    {"raw": "minor forest loss is visible"},
                    {"raw": "small areas of deforestation"},
                    {"raw": "limited tree cover reduction"},
                    {"raw": "minor deforestation observed"},
                    {"raw": "sparse forest loss detected"},
                ],
            }
            for split in splits
            for i in range(N)
        ]
    }


if __name__ == "__main__":
    if os.path.exists(root):
        shutil.rmtree(root)

    # create directory structure
    for split in splits:
        for d in directories:
            os.makedirs(os.path.join(root, "images", split, d))

    # write images + masks
    for split in splits:
        for i in range(N):
            name = f"{split}_{i:06d}.png"

            create_image(os.path.join(root, "images", split, "A", name))
            create_image(os.path.join(root, "images", split, "B", name))
            create_mask(os.path.join(root, "images", split, "label", name))

    # captions
    with open(os.path.join(root, "ForestChatcaptions.json"), "w") as f:
        json.dump(create_captions(), f)

    # zip (keep directory)
    zip_path = root + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(root):
            for file in files:
                path = os.path.join(folder, file)
                zf.write(path, os.path.relpath(path, "."))

    print(zip_path)
