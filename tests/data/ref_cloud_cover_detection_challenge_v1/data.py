# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
from pystac import (
    Asset,
    CatalogType,
    Collection,
    Extent,
    Item,
    Link,
    MediaType,
    SpatialExtent,
    TemporalExtent,
)
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.label import (
    LabelClasses,
    LabelCount,
    LabelExtension,
    LabelOverview,
    LabelType,
)
from rasterio.crs import CRS
from rasterio.transform import Affine

np.random.seed(0)

SIZE = 512
BANDS = ["B02", "B03", "B04", "B08"]

SOURCE_COLLECTION_ID = "ref_cloud_cover_detection_challenge_v1_test_source"
SOURCE_ITEM_ID = "ref_cloud_cover_detection_challenge_v1_test_source_aaaa"
LABEL_COLLECTION_ID = "ref_cloud_cover_detection_challenge_v1_test_labels"
LABEL_ITEM_ID = "ref_cloud_cover_detection_challenge_v1_test_labels_aaaa"

# geometry used by both source and label items
TEST_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [137.86580132892396, -29.52744848758255],
            [137.86450090473795, -29.481297003404038],
            [137.91724642199793, -29.48015007212528],
            [137.9185707094313, -29.526299409555623],
            [137.86580132892396, -29.52744848758255],
        ]
    ],
}

# bbox used by both source and label items
TEST_BBOX = [
    137.86450090473795,
    -29.52744848758255,
    137.9185707094313,
    -29.48015007212528,
]

# sentinel-2 bands for EO extension
S2_BANDS = [
    Band.create(name="B02", common_name="blue", description="Blue"),
    Band.create(name="B03", common_name="green", description="Green"),
    Band.create(name="B04", common_name="red", description="Red"),
    Band.create(name="B08", common_name="nir", description="NIR"),
]

# class map for overviews
CLASS_COUNT_MAP = {"0": "no cloud", "1": "cloud"}

# define the spatial and temporal extent of collections
TEST_EXTENT = Extent(
    spatial=SpatialExtent(
        bboxes=[
            [
                -80.05464265420176,
                -53.31380701212582,
                151.75593282192196,
                35.199126843018696,
            ]
        ]
    ),
    temporal=TemporalExtent(
        intervals=[
            [
                dt.strptime("2018-02-18", "%Y-%m-%d"),
                dt.strptime("2020-09-13", "%Y-%m-%d"),
            ]
        ]
    ),
)


def create_raster(path: str, dtype: str, num_channels: int, collection: str) -> None:

    if not os.path.exists(os.path.split(path)[0]):
        Path(os.path.split(path)[0]).mkdir(parents=True)

    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = CRS.from_epsg(32753)
    profile["transform"] = Affine(1.0, 0.0, 777760.0, 0.0, -10.0, 6735270.0)
    profile["height"] = SIZE
    profile["width"] = SIZE
    profile["compress"] = "lzw"
    profile["predictor"] = 2

    if collection == "source":
        if "float" in profile["dtype"]:
            Z = np.random.randn(SIZE, SIZE).astype(profile["dtype"])
        else:
            Z = np.random.randint(
                np.iinfo(profile["dtype"]).max,
                size=(SIZE, SIZE),
                dtype=profile["dtype"],
            )
    elif collection == "labels":
        Z = np.random.randint(0, 2, (SIZE, SIZE)).astype(profile["dtype"])

    with rasterio.open(path, "w", **profile) as src:
        for i in range(1, profile["count"] + 1):
            src.write(Z, i)


def create_source_item() -> Item:
    # instantiate source Item
    test_source_item = Item(
        id=SOURCE_ITEM_ID,
        geometry=TEST_GEOMETRY,
        bbox=TEST_BBOX,
        datetime=dt.strptime("2020-06-03", "%Y-%m-%d"),
        properties={},
    )

    # add Asset with EO Extension for each S2 band
    for band in BANDS:
        img_path = os.path.join(
            os.getcwd(), SOURCE_COLLECTION_ID, SOURCE_ITEM_ID, f"{band}.tif"
        )
        image_asset = Asset(href=img_path, media_type=MediaType.GEOTIFF)
        eo_asset_ext = EOExtension.ext(image_asset)

        for s2_band in S2_BANDS:
            if s2_band.name == band:
                eo_asset_ext.apply(bands=[s2_band])
                test_source_item.add_asset(key=band, asset=image_asset)

    eo_image_ext = EOExtension.ext(test_source_item, add_if_missing=True)
    eo_image_ext.apply(bands=S2_BANDS)

    return test_source_item


def get_class_label_list(overview: LabelOverview) -> LabelClasses:
    label_list = [d["name"] for d in overview.properties["counts"]]
    label_classes = LabelClasses.create(classes=label_list, name="labels")
    return label_classes


def get_item_class_overview(label_type: LabelType, asset_path: str) -> LabelOverview:

    """Takes a path to an asset based on type and returns the class label
    overview object

    Args:
    label_type: LabelType - the type of label, either RASTER or VECTOR
    asset_path: str - path to the asset to read in either a raster image or
    geojson vector

    Returns:
    overview: LabelOverview - the STAC LabelOverview object containing label classes

    """

    count_list = []

    img_arr = rasterio.open(asset_path).read()
    value_count = np.unique(img_arr.flatten(), return_counts=True)

    for ix, classy in enumerate(value_count[0]):
        if classy > 0:
            label_count = LabelCount.create(
                name=CLASS_COUNT_MAP[str(int(classy))], count=int(value_count[1][ix])
            )
            count_list.append(label_count)

    overview = LabelOverview(properties={})
    overview.apply(property_key="labels", counts=count_list)

    return overview


def create_label_item() -> Item:
    # instantiate label Item
    test_label_item = Item(
        id=LABEL_ITEM_ID,
        geometry=TEST_GEOMETRY,
        bbox=TEST_BBOX,
        datetime=dt.strptime("2020-06-03", "%Y-%m-%d"),
        properties={},
    )

    label_overview = get_item_class_overview(LabelType.RASTER, label_path)
    label_list = get_class_label_list(label_overview)

    label_ext = LabelExtension.ext(test_label_item, add_if_missing=True)
    label_ext.apply(
        label_description="Sentinel-2 Cloud Cover Segmentation Test Labels",
        label_type=LabelType.RASTER,
        label_classes=[label_list],
        label_overviews=[label_overview],
    )

    label_asset = Asset(href=label_path, media_type=MediaType.GEOTIFF)
    test_label_item.add_asset(key="labels", asset=label_asset)

    return test_label_item


if __name__ == "__main__":

    # create a geotiff for each s2 band
    for b in BANDS:
        tif_path = os.path.join(
            os.getcwd(), SOURCE_COLLECTION_ID, SOURCE_ITEM_ID, f"{b}.tif"
        )
        create_raster(tif_path, "uint8", 1, "source")

    # create a geotiff for label
    label_path = os.path.join(
        os.getcwd(), LABEL_COLLECTION_ID, LABEL_ITEM_ID, "labels.tif"
    )
    create_raster(label_path, "uint8", 1, "labels")

    # instantiate the source Collection
    test_source_collection = Collection(
        id=SOURCE_COLLECTION_ID,
        description="Test Source Collection for Torchgo Cloud Cover Detection Dataset",
        extent=TEST_EXTENT,
        catalog_type=CatalogType.RELATIVE_PUBLISHED,
        license="CC-BY-4.0",
    )

    source_item = create_source_item()
    test_source_collection.add_item(source_item)

    test_source_collection.normalize_hrefs(
        os.path.join(os.getcwd(), SOURCE_COLLECTION_ID)
    )
    test_source_collection.make_all_asset_hrefs_relative()
    test_source_collection.save(catalog_type=CatalogType.SELF_CONTAINED)

    # instantiate the label Collection
    test_label_collection = Collection(
        id=LABEL_COLLECTION_ID,
        description="Test Label Collection for Torchgo Cloud Cover Detection Dataset",
        extent=TEST_EXTENT,
        catalog_type=CatalogType.RELATIVE_PUBLISHED,
        license="CC-BY-4.0",
    )

    label_item = create_label_item()
    label_item.add_link(
        Link(rel="source", target=source_item, media_type=MediaType.GEOTIFF)
    )
    test_label_collection.add_item(label_item)

    test_label_collection.normalize_hrefs(
        os.path.join(os.getcwd(), LABEL_COLLECTION_ID)
    )
    test_label_collection.make_all_asset_hrefs_relative()
    test_label_collection.save(catalog_type=CatalogType.SELF_CONTAINED)
