import os
import shutil

import numpy as np
from PIL import Image

# Define the root directory and subdirectories
root_dir = 'my_new_dataset'
sub_dirs = ['sub_dir_1', 'sub_dir_2', 'sub_dir_3']
splits = ['train', 'val', 'test']

image_file_names = ['/content/torchgeo/images/geodataset.png', '/content/torchgeo/images/inria.png', '/content/torchgeo/images/vhr10.png']

IMG_SIZE = 32


# Function to create dummy input images
def create_input_image(path: str, shape: tuple[int], pixel_values: list[int]) -> None:
    data = np.random.choice(pixel_values, size=shape, replace=True).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(path)


# Function to create dummy targets
def create_target_images(split: str, filename: str) -> None:
    target_pixel_values = range(10)
    path = os.path.join(root_dir, 'target', split, filename)
    create_input_image(path, (IMG_SIZE, IMG_SIZE), target_pixel_values)


# Create a new clean version when re-running the script
if os.path.exists(root_dir):
    shutil.rmtree(root_dir)

# Create the directory structure
for sub_dir in sub_dirs:
    for split in splits:
        os.makedirs(os.path.join(root_dir, sub_dir, split), exist_ok=True)

# Create dummy data for all splits and filenames
for split in splits:
    for filename in image_file_names:
        create_input_image(
            os.path.join(root_dir, 'image', split, filename),
            (IMG_SIZE, IMG_SIZE),
            range(2**16),
        )
        create_target_images(split, filename.replace('_', '_target_'))

# Zip directory
shutil.make_archive(root_dir, 'zip', '.', root_dir)
