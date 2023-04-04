""" Sample patch locations for downloading

### run the script:
## sample new locations with rtree overlap search
python sample_ssl4eo.py \
    --save_path ./data \
    --radius 1320 \
    --num_cities 10000 \
    --std 50 \
    --indices_range 0 250000

## resume from interruption
python sample_ssl4eo.py \
    --save_path ./data \
    --radius 1320 \
    --num_cities 10000 \
    --std 50 \
    --resume ./data/sampled_locations.csv \
    --indices_range 0 250000

### Notes
# The script will sample locations with rtree overlap search.
# The script will save the sampled locations to a csv file.
# By default, GaussianSampler is used to sample locations with a standard deviation
 of 50 km from top 10000 populated cities.
# By default, 25% overlap of adjacent patches is allowed.
# Radius (meter) is half the wanted patch size.

"""

import argparse
import csv
import os
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
from rtree import index
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)


""" samplers to get locations of interest points"""


def get_world_cities(download_root: str = "world_cities") -> List[Dict[str, Any]]:
    url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip"  # noqa: E501
    filename = "worldcities.csv"
    if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
        download_and_extract_archive(url, download_root)
    with open(os.path.join(download_root, filename), encoding="UTF-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
        cities = []
        for row in reader:
            row["population"] = (
                row["population"].replace(".", "") if row["population"] else "0"
            )
            cities.append(row)
    return cities


def get_interest_points(
    cities: List[Dict[str, str]], size: int = 10000
) -> List[List[float]]:
    cities = sorted(cities, key=lambda c: int(c["population"]), reverse=True)[:size]
    points = [[float(c["lng"]), float(c["lat"])] for c in cities]
    return points


def km2deg(kms: float, radius: float = 6371) -> float:
    return kms / (2.0 * radius * np.pi / 360.0)


def sample_point(interest_points: List[List[float]], std: float) -> Tuple[float, float]:
    rng = np.random.default_rng()
    point = rng.choice(interest_points)
    std = km2deg(std)
    lon, lat = np.random.normal(loc=point, scale=[std, std])
    return (lon, lat)


def create_bbox(
    coords: Tuple[float, float], bbox_size_degree: float
) -> Tuple[float, float, float, float]:
    lon, lat = coords
    bbox = (
        lon - bbox_size_degree,
        lat - bbox_size_degree,
        lon + bbox_size_degree,
        lat + bbox_size_degree,
    )
    return bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default="./data/", help="dir to save data"
    )
    parser.add_argument(
        "--radius", type=float, default=1320, help="patch radius in meters"
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.25,
        help="max overlap ratio between adjacent patches",
    )
    parser.add_argument(
        "--num-cities", type=int, default=10000, help="number of cities to sample"
    )
    parser.add_argument(
        "--std", type=int, default=50, help="std of gaussian distribution"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    parser.add_argument(
        "--indices-range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to download",
    )

    args = parser.parse_args()
    np.random.seed(42)

    # if resume
    ext_coords = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
    else:
        ext_path = os.path.join(args.save_path, "sampled_locations.csv")

    # initialize sampler
    cities = get_world_cities()
    interest_points = get_interest_points(cities, size=args.num_cities)
    bbox_size = (
        (1 - args.overlap_ratio) * args.radius / 1000
    )  # allow little overlap between adjacent patches
    bbox_size_degree = km2deg(bbox_size)

    # build rtree
    rtree_coords = index.Index()
    if args.resume:
        print("Load existing locations.")
        for i, key in enumerate(tqdm(ext_coords.keys())):
            c = ext_coords[key]
            bbox = create_bbox(c, bbox_size_degree)
            rtree_coords.insert(i, bbox)

    # sample locations
    start_time = time.time()
    indices = range(args.indices_range[0], args.indices_range[1])

    for idx in tqdm(indices):
        # skip if already sampled
        if str(idx) in ext_coords.keys():
            if args.resume:
                continue

        # sample new coord and check overlap
        count = 0
        while count < 1:
            new_coord = sample_point(
                interest_points, args.std
            )  # (lon,lat) of top-10000 cities
            bbox = create_bbox(new_coord, bbox_size_degree)
            if list(rtree_coords.intersection(bbox)):
                continue
            rtree_coords.insert(idx, bbox)
            count += 1

        # save to file
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            data = [idx, new_coord[0], new_coord[1]]
            writer.writerow(data)

    print(
        f"Sampled locations saved to {ext_path} in {time.time()-start_time:.2f} seconds."  # noqa: E501
    )
