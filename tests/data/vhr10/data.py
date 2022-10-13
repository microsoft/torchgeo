import json
import os
from pathlib import Path

import numpy as np
import rasterio as rio
from torchvision.datasets.utils import calculate_md5

ANNOTATION_FILE = {"images": [], "annotations": []}


def write_data(path: str, img: np.ndarray) -> None:
    with rio.open(
        path,
        "w",
        driver="JP2OpenJPEG",
        height=img.shape[0],
        width=img.shape[1],
        count=3,
        dtype=img.dtype,
    ) as dst:
        for i in range(1, dst.count + 1):
            dst.write(img, i)


def generate_test_data(root: str, n_imgs: int = 3) -> str:
    folder_path = os.path.join(root, "NWPU VHR-10 dataset")
    pos_img_dir = os.path.join(folder_path, "positive image set")
    neg_img_dir = os.path.join(folder_path, "negative image set")
    ann_file = os.path.join(folder_path, "annotations.json")

    if not os.path.exists(pos_img_dir):
        os.makedirs(pos_img_dir)
    if not os.path.exists(neg_img_dir):
        os.makedirs(neg_img_dir)

    for img_id in range(1, n_imgs + 1):
        pos_img_name = os.path.join(pos_img_dir, f"00{img_id}.jpg")
        neg_img_name = os.path.join(neg_img_dir, f"00{img_id}.jpg")

        img = np.random.randint(255, size=(8, 8), dtype=np.dtype("uint8"))
        write_data(pos_img_name, img)
        write_data(neg_img_name, img)

        img_name = Path(pos_img_name).name
        ANNOTATION_FILE["images"].append(
            {"file_name": img_name, "height": 8, "width": 8, "id": img_id - 1}
        )

    ann = 0
    for img in ANNOTATION_FILE["images"]:
        for _ in range(2):
            annot = {
                "id": ann,
                "image_id": img["id"],
                "category_id": 1,
                "area": 4.0,
                "bbox": [4, 4, 2, 2],
                "segmentation": [[1, 1, 2, 2, 3, 3, 4, 5, 5]],
                "iscrowd": 0,
            }
            ann += 1
            ANNOTATION_FILE["annotations"].append(annot)

    with open(ann_file, "w") as j:
        json.dump(ANNOTATION_FILE, j)

    annotations_md5 = calculate_md5(ann_file)
    # TODO: Create rar and return md5 hash
    return annotations_md5


if __name__ == "__main__":
    md5 = generate_test_data(os.getcwd())
    print(md5)
