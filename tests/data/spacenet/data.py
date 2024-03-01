#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections import OrderedDict
from typing import cast

import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from torchvision.datasets.utils import calculate_md5

from torchgeo.datasets import (
    SpaceNet,
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet6,
    SpaceNet7,
)

transform = Affine(0.3, 0.0, 616500.0, 0.0, -0.3, 3345000.0)
crs = CRS.from_epsg(4326)

img_count = {
    "MS.tif": 8,
    "PAN.tif": 1,
    "PS-MS.tif": 8,
    "PS-RGB.tif": 3,
    "PS-RGBNIR.tif": 4,
    "RGB.tif": 3,
    "RGBNIR.tif": 4,
    "SAR-Intensity.tif": 1,
    "mosaic.tif": 3,
    "8Band.tif": 8,
}


sn4_catalog = [
    "10300100023BC100",
    "10300100036D5200",
    "1030010003BDDC00",
    "1030010003CD4300",
]
sn4_angles = [8, 30, 52, 53]

sn4_imgdirname = "sn4_SN4_buildings_train_AOI_6_Atlanta_732701_3730989-nadir{}_catid_{}"
sn4_lbldirname = "sn4_SN4_buildings_train_AOI_6_Atlanta_732701_3730989-labels"
sn4_emptyimgdirname = (
    "sn4_SN4_buildings_train_AOI_6_Atlanta_732701_3720639-nadir53_"
    + "catid_1030010003CD4300"
)
sn4_emptylbldirname = "sn4_SN4_buildings_train_AOI_6_Atlanta_732701_3720639-labels"


datasets = [SpaceNet1, SpaceNet2, SpaceNet3, SpaceNet4, SpaceNet5, SpaceNet6, SpaceNet7]


def create_test_image(img_dir: str, imgs: list[str]) -> list[list[float]]:
    """Create test image

    Args:
        img_dir (str): Name of image directory
        imgs (List[str]): List of images to be created

    Returns:
        List[List[float]]: Boundary coordinates
    """
    for img in imgs:
        imgpath = os.path.join(img_dir, img)
        Z = np.arange(4, dtype="uint16").reshape(2, 2)
        count = img_count[img]
        with rasterio.open(
            imgpath,
            "w",
            driver="GTiff",
            height=Z.shape[0],
            width=Z.shape[1],
            count=count,
            dtype=Z.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(1, dst.count + 1):
                dst.write(Z, i)

    tim = rasterio.open(imgpath)
    slice_index = [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]]
    return [list(tim.transform * p) for p in slice_index]


def create_test_label(
    lbldir: str,
    lblname: str,
    coords: list[list[float]],
    det_type: str,
    empty: bool = False,
    diff_crs: bool = False,
) -> None:
    """Create test label

    Args:
        lbldir (str): Name of label directory
        lblname (str): Name of label file
        coords (List[Tuple[float, float]]): Boundary coordinates
        det_type (str): Type of dataset. Must be either buildings or roads.
        empty (bool, optional): Creates empty label file if True. Defaults to False.
        diff_crs (bool, optional): Assigns EPSG:3857 as CRS instead of
                                   default EPSG:4326. Defaults to False.
    """
    if empty:
        # Creates a new file
        with open(os.path.join(lbldir, lblname), "w"):
            pass
        return

    if det_type == "buildings":
        meta_properties = OrderedDict()
        geom = "Polygon"
        rec = {
            "type": "Feature",
            "id": "0",
            "properties": OrderedDict(),
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }
    else:
        meta_properties = OrderedDict(
            [
                ("heading", "str"),
                ("lane_number", "str"),
                ("one_way_ty", "str"),
                ("paved", "str"),
                ("road_id", "int"),
                ("road_type", "str"),
                ("origarea", "int"),
                ("origlen", "float"),
                ("partialDec", "int"),
                ("truncated", "int"),
                ("bridge_type", "str"),
                ("inferred_speed_mph", "float"),
                ("inferred_speed_mps", "float"),
            ]
        )
        geom = "LineString"

        dummy_vals = {"str": "a", "float": 45.0, "int": 0}
        ROAD_DICT = [(k, dummy_vals[v]) for k, v in meta_properties.items()]
        rec = {
            "type": "Feature",
            "id": "0",
            "properties": OrderedDict(ROAD_DICT),
            "geometry": {"type": "LineString", "coordinates": [coords[0], coords[2]]},
        }

    meta = {
        "driver": "GeoJSON",
        "schema": {"properties": meta_properties, "geometry": geom},
        "crs": {"init": "epsg:4326"},
    }
    if diff_crs:
        meta["crs"] = {"init": "epsg:3857"}
    out_file = os.path.join(lbldir, lblname)
    with fiona.open(out_file, "w", **meta) as dst:
        dst.write(rec)


def main() -> None:
    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    for dataset in datasets:
        collections = list(dataset.collection_md5_dict.keys())
        for collection in collections:
            dataset = cast(SpaceNet, dataset)
            if dataset.dataset_id == "spacenet4":
                num_samples = 4
            elif collection == "sn5_AOI_7_Moscow" or collection not in [
                "sn5_AOI_8_Mumbai",
                "sn7_test_source",
            ]:
                num_samples = 3
            elif collection == "sn5_AOI_8_Mumbai":
                num_samples = 3
            else:
                num_samples = 1

            for sample in range(num_samples):
                out_dir = os.path.join(ROOT_DIR, collection)
                if collection == "sn6_AOI_11_Rotterdam":
                    out_dir = os.path.join(ROOT_DIR, "spacenet6", collection)

                # Create img dir
                if dataset.dataset_id == "spacenet4":
                    assert num_samples == 4
                    if sample != 3:
                        imgdirname = sn4_imgdirname.format(
                            sn4_angles[sample], sn4_catalog[sample]
                        )
                        lbldirname = sn4_lbldirname
                    else:
                        imgdirname = sn4_emptyimgdirname.format(
                            sn4_angles[sample], sn4_catalog[sample]
                        )
                        lbldirname = sn4_emptylbldirname
                else:
                    imgdirname = f"{collection}_img{sample + 1}"
                    lbldirname = f"{collection}_img{sample + 1}-labels"

                imgdir = os.path.join(out_dir, imgdirname)
                os.makedirs(imgdir, exist_ok=True)
                bounds = create_test_image(imgdir, list(dataset.imagery.values()))

                # Create lbl dir
                lbldir = os.path.join(out_dir, lbldirname)
                os.makedirs(lbldir, exist_ok=True)
                det_type = "roads" if dataset in [SpaceNet3, SpaceNet5] else "buildings"
                if dataset.dataset_id == "spacenet4" and sample == 3:
                    # Creates an empty file
                    create_test_label(
                        lbldir, dataset.label_glob, bounds, det_type, empty=True
                    )
                else:
                    create_test_label(lbldir, dataset.label_glob, bounds, det_type)

                if collection == "sn5_AOI_8_Mumbai":
                    if sample == 1:
                        create_test_label(
                            lbldir, dataset.label_glob, bounds, det_type, empty=True
                        )
                    if sample == 2:
                        create_test_label(
                            lbldir, dataset.label_glob, bounds, det_type, diff_crs=True
                        )

                if collection == "sn1_AOI_1_RIO" and sample == 1:
                    create_test_label(
                        lbldir, dataset.label_glob, bounds, det_type, diff_crs=True
                    )

                if collection not in [
                    "sn2_AOI_2_Vegas",
                    "sn3_AOI_5_Khartoum",
                    "sn4_AOI_6_Atlanta",
                    "sn5_AOI_8_Mumbai",
                    "sn6_AOI_11_Rotterdam",
                    "sn7_train_source",
                ]:
                    # Create collection.json
                    with open(
                        os.path.join(ROOT_DIR, collection, "collection.json"), "w"
                    ):
                        pass
                if collection == "sn6_AOI_11_Rotterdam":
                    # Create collection.json
                    with open(
                        os.path.join(
                            ROOT_DIR, "spacenet6", collection, "collection.json"
                        ),
                        "w",
                    ):
                        pass

            # Create archive
            if collection == "sn6_AOI_11_Rotterdam":
                break
            archive_path = os.path.join(ROOT_DIR, collection)
            shutil.make_archive(
                archive_path, "gztar", root_dir=ROOT_DIR, base_dir=collection
            )
            shutil.rmtree(out_dir)
            print(f'{collection}: {calculate_md5(f"{archive_path}.tar.gz")}')


if __name__ == "__main__":
    main()
