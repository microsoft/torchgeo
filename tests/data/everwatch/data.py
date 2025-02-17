#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os
import csv
import shutil
import numpy as np
from PIL import Image
import hashlib

DATASET_DIR = 'everwatch-benchmark'
TRAIN_CSV_FILENAME = 'train.csv'
TEST_CSV_FILENAME = 'test.csv'
IMG_SIZE = 64  # width and height in pixels

# EverWatch classes as defined in the dataset
EVERWATCH_CLASSES = (
    'White Ibis',
    'Great Egret',
    'Great Blue Heron',
    'Snowy Egret',
    'Wood Stork',
    'Roseate Spoonbill',
    'Anhinga',
)

def create_dummy_image(path: str, size: int = IMG_SIZE) -> None:
    """Create a dummy RGB image."""
    np.random.seed(0)
    array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    image = Image.fromarray(array, 'RGB')
    image.save(path)

def create_dummy_annotations(csv_path: str, image_names: list[str]) -> None:
    """Create a CSV annotation file with two annotations for each image."""
    headers = ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    data = []
    
    # Define two dummy bounding boxes
    box1 = (5, 5, IMG_SIZE // 2, IMG_SIZE // 2)
    box2 = (IMG_SIZE // 2 + 1, IMG_SIZE // 2 + 1, IMG_SIZE - 5, IMG_SIZE - 5)
    
    class_assignment = {
        'train1.png': [EVERWATCH_CLASSES[0], EVERWATCH_CLASSES[1]],
        'train2.png': [EVERWATCH_CLASSES[2], EVERWATCH_CLASSES[3]],
        'train3.png': [EVERWATCH_CLASSES[4], EVERWATCH_CLASSES[5]],
        'test1.png':  [EVERWATCH_CLASSES[0], EVERWATCH_CLASSES[2]],
        'test2.png':  [EVERWATCH_CLASSES[3], EVERWATCH_CLASSES[6]],
    }
    
    for img_name in image_names:
        cls1, cls2 = class_assignment.get(img_name, [EVERWATCH_CLASSES[0], EVERWATCH_CLASSES[0]])
        data.append([img_name, *box1, cls1])
        data.append([img_name, *box2, cls2])
    
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

def main():
    dataset_path = os.path.join(".", DATASET_DIR)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Define image names for each split
    train_image_names = ['train1.png', 'train2.png', 'train3.png']
    test_image_names = ['test1.png', 'test2.png']
    
    for name in train_image_names + test_image_names:
        img_path = os.path.join(dataset_path, name)
        create_dummy_image(img_path, size=IMG_SIZE)
    
    train_csv_path = os.path.join(dataset_path, TRAIN_CSV_FILENAME)
    test_csv_path = os.path.join(dataset_path, TEST_CSV_FILENAME)
    create_dummy_annotations(train_csv_path, train_image_names)
    create_dummy_annotations(test_csv_path, test_image_names)
    
    # create zip archive and compute md5sum
    shutil.make_archive(dataset_path, 'zip', ".", dataset_path)

    # Compute checksums
    with open(dataset_path + '.zip', 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{dataset_path}: {md5}')

if __name__ == '__main__':
    main()