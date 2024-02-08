# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import h5py
import numpy as np
import pandas as pd
from torchvision.datasets.utils import calculate_md5

# Define the root directory
root = "WP"
IMAGE_SIZE = 32
NUM_TYHOON_IDS = 5
NUM_IMAGES_PER_ID = 4
CHUNK_SIZE = 2**12

# If the root directory exists, remove it
if os.path.exists(root):
    shutil.rmtree(root)

# Create the 'image' and 'metadata' directories
os.makedirs(os.path.join(root, "image"))
os.makedirs(os.path.join(root, "metadata"))

# For each typhoon_id
all_dfs = []
for typhoon_id in range(NUM_TYHOON_IDS):
    # Create a directory under 'root/image/typhoon_id/'
    os.makedirs(os.path.join(root, "image", str(typhoon_id)), exist_ok=True)

    # Create dummy .h5 files
    image_paths_per_typhoon = []
    for image_id in range(NUM_IMAGES_PER_ID):
        image_file_name = f"{image_id}.h5"
        with h5py.File(
            os.path.join(root, "image", str(typhoon_id), image_file_name), "w"
        ) as hf:
            hf.create_dataset("Infrared", data=np.random.rand(IMAGE_SIZE, IMAGE_SIZE))
        image_paths_per_typhoon.append(image_file_name)

    start_time = pd.Timestamp(
        year=np.random.randint(1978, 2022),
        month=np.random.randint(1, 13),
        day=np.random.randint(1, 29),
        hour=np.random.randint(0, 24),
    )
    times = pd.date_range(start=start_time, periods=NUM_IMAGES_PER_ID, freq="H")
    df = pd.DataFrame(
        {
            "id": np.repeat(typhoon_id, NUM_IMAGES_PER_ID),
            "image_path": image_paths_per_typhoon,
            "year": times.year,
            "month": times.month,
            "day": times.day,
            "hour": times.hour,
            "grade": np.random.randint(1, 5, NUM_IMAGES_PER_ID),
            "lat": np.random.uniform(-90, 90, NUM_IMAGES_PER_ID),
            "lng": np.random.uniform(-180, 180, NUM_IMAGES_PER_ID),
            "pressure": np.random.uniform(900, 1000, NUM_IMAGES_PER_ID),
            "wind": np.random.uniform(0, 100, NUM_IMAGES_PER_ID),
            "dir50": np.random.randint(0, 360, NUM_IMAGES_PER_ID),
            "long50": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "short50": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "dir30": np.random.randint(0, 360, NUM_IMAGES_PER_ID),
            "long30": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "short30": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "landfall": np.random.randint(0, 2, NUM_IMAGES_PER_ID),
            "intp": np.random.randint(0, 2, NUM_IMAGES_PER_ID),
            "file_1": [f"{idx}.h5" for idx in range(1, NUM_IMAGES_PER_ID + 1)],
            "mask_1": [
                "mask_" + str(i) for i in np.random.randint(1, 100, NUM_IMAGES_PER_ID)
            ],
            "mask_1_pct": np.random.uniform(0, 100, NUM_IMAGES_PER_ID),
        }
    )

    # Save the DataFrame to correspoding typhoon id as metadata
    df.to_csv(os.path.join(root, "metadata", f"{typhoon_id}.csv"), index=False)

    all_dfs.append(df)

# Save the aux_data.csv
aux_data = pd.concat(all_dfs)
aux_data.to_csv(os.path.join(root, "aux_data.csv"), index=False)


# Create tarball
shutil.make_archive(root, "gztar", ".", root)

# simulate multiple tar files
path = f"{root}.tar.gz"
paths = []
with open(path, "rb") as f:
    # Write the entire tarball to gzaa
    split = f"{path}aa"
    with open(split, "wb") as g:
        g.write(f.read())
    paths.append(split)

# Create gzab as a copy of gzaa
shutil.copy2(f"{path}aa", f"{path}ab")
paths.append(f"{path}ab")


# Calculate the md5sum of the tar file
for path in paths:
    print(f"{path}: {calculate_md5(path)}")
