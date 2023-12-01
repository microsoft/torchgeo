# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import tarfile

import h5py
import numpy as np
import pandas as pd
from torchvision.datasets.utils import calculate_md5

# Define the root directory
root = "WP/"
IMAGE_SIZE = 32
NUM_TYHOON_IDS = 5
NUM_IMAGES_PER_ID = 4

# Define the root directory
root = "./WP"

# If the root directory exists, remove it
if os.path.exists(root):
    shutil.rmtree(root)

# Create the root directory if it doesn't exist
os.makedirs(root)

# Create the 'image' and 'metadata' directories
os.makedirs(os.path.join(root, "image"))
os.makedirs(os.path.join(root, "metadata"))

# For each typhoon_id
all_dfs = []
for typhoon_id in range(1, NUM_TYHOON_IDS):
    # Create a directory under 'root/image/typhoon_id/'
    os.makedirs(os.path.join(root, "image", str(typhoon_id)), exist_ok=True)

    # Create dummy .hf files

    for image_id in range(1, NUM_IMAGES_PER_ID):
        image_file_name = f"{image_id}.hf"
        with h5py.File(
            os.path.join(root, "image", str(typhoon_id), image_file_name), "w"
        ) as hf:
            hf.create_dataset("Infrared", data=np.random.rand(IMAGE_SIZE, IMAGE_SIZE))

    # Create a dummy .csv file with metadata for each typhoon_id
    df = pd.DataFrame(
        {
            "year": np.random.randint(1978, 2022, NUM_IMAGES_PER_ID),
            "month": np.random.randint(1, 13, NUM_IMAGES_PER_ID),
            "day": np.random.randint(1, 32, NUM_IMAGES_PER_ID),
            "hour": np.random.randint(0, 24, NUM_IMAGES_PER_ID),
            "grade": np.random.randint(1, 5, NUM_IMAGES_PER_ID),
            "lat": np.random.uniform(-90, 90, NUM_IMAGES_PER_ID),
            "lng": np.random.uniform(-180, 180, NUM_IMAGES_PER_ID),
            "pressure": np.random.uniform(950, 1050, NUM_IMAGES_PER_ID),
            "wind": np.random.uniform(0, 100, NUM_IMAGES_PER_ID),
            "dir50": np.random.randint(0, 360, NUM_IMAGES_PER_ID),
            "long50": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "short50": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "dir30": np.random.randint(0, 360, NUM_IMAGES_PER_ID),
            "long30": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "short30": np.random.randint(0, 100, NUM_IMAGES_PER_ID),
            "landfall": np.random.randint(0, 2, NUM_IMAGES_PER_ID),
            "intp": np.random.randint(0, 2, NUM_IMAGES_PER_ID),
            "file_1": [f"{idx}.hf" for idx in range(1, NUM_IMAGES_PER_ID + 1)],
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

# Create a tar file
tar_path = "WP.tar.gz"
with tarfile.open(tar_path, "w") as tar:
    tar.add(root, arcname=os.path.basename(root))

# Calculate the md5sum of the tar file
print(f"{tar_path}: {calculate_md5(tar_path)}")
