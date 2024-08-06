# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
import subprocess

import numpy as np
from PIL import Image
from torchvision.datasets.utils import calculate_md5

ANNOTATION_FILE = {'images': [], 'annotations': []}


def write_data(path: str, img: np.ndarray) -> None:
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = Image.fromarray(img)
    img.save(path)


def generate_test_data(root: str, n_imgs: int = 3) -> str:
    folder_path = os.path.join(root, 'NWPU VHR-10 dataset')
    pos_img_dir = os.path.join(folder_path, 'positive image set')
    neg_img_dir = os.path.join(folder_path, 'negative image set')
    ann_file = os.path.join(folder_path, 'annotations.json')
    ann_file2 = os.path.join(root, 'annotations.json')

    if not os.path.exists(pos_img_dir):
        os.makedirs(pos_img_dir)
    if not os.path.exists(neg_img_dir):
        os.makedirs(neg_img_dir)

    for img_id in range(1, n_imgs + 1):
        pos_img_name = os.path.join(pos_img_dir, f'00{img_id}.jpg')
        neg_img_name = os.path.join(neg_img_dir, f'00{img_id}.jpg')

        img = np.random.randint(255, size=(8, 8), dtype=np.dtype('uint8'))
        write_data(pos_img_name, img)
        write_data(neg_img_name, img)

        img_name = os.path.basename(pos_img_name)

        ANNOTATION_FILE['images'].append(
            {'file_name': img_name, 'height': 8, 'width': 8, 'id': img_id - 1}
        )

    ann = 0
    for _, img in enumerate(ANNOTATION_FILE['images']):
        annot = {
            'id': ann,
            'image_id': img['id'],
            'category_id': 1,
            'area': 4.0,
            'bbox': [4, 4, 2, 2],
            'segmentation': [[1, 1, 2, 2, 3, 3, 4, 5, 5]],
            'iscrowd': 0,
        }
        ANNOTATION_FILE['annotations'].append(annot)
        ann += 1

    with open(ann_file, 'w') as j:
        json.dump(ANNOTATION_FILE, j)

    with open(ann_file2, 'w') as j:
        json.dump(ANNOTATION_FILE, j)

    # Create rar file
    subprocess.run(
        ['rar', 'a', 'NWPU VHR-10 dataset.rar', '-m5', 'NWPU VHR-10 dataset'],
        capture_output=True,
        check=True,
    )

    annotations_md5 = calculate_md5(ann_file)
    archive_md5 = calculate_md5('NWPU VHR-10 dataset.rar')
    shutil.rmtree(folder_path)

    return f'archive md5: {archive_md5}, annotation md5: {annotations_md5}'


if __name__ == '__main__':
    md5 = generate_test_data(os.getcwd(), 5)
    print(md5)
