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
    --resume ./data/checked_locations.csv \
    --indices_range 0 250000

### Notes
# The script will sample locations with rtree overlap search.
# The script will save the sampled locations to a csv file.
# By default, GaussianSampler is used to sample locations with a standard deviation of 50 km from top 10000 populated cities. # noqa: E501
# By default, 25% overlap of adjacent patches is allowed.
# Radius (meter) is half the wanted patch size.

"""

import argparse
import csv
import os
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import shapefile
from rtree import index
from shapely.geometry import Point, shape
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)


""" samplers to get locations of interest points"""


class UniformSampler:
    def sample_point(self) -> List[float]:
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-90, 90)
        return [lon, lat]


class GaussianSampler:
    def __init__(
        self,
        interest_points: Optional[List[List[float]]] = None,
        num_cities: int = 1000,
        std: float = 20,
    ) -> None:
        if interest_points is None:
            cities = self.get_world_cities()
            self.interest_points = self.get_interest_points(cities, size=num_cities)
        else:
            self.interest_points = interest_points
        self.std = std

    def sample_point(self) -> List[float]:
        rng = np.random.default_rng()
        point = rng.choice(self.interest_points)
        std = self.km2deg(self.std)
        lon, lat = np.random.normal(loc=point, scale=[std, std])
        return [lon, lat]

    @staticmethod
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

    @staticmethod
    def get_interest_points(
        cities: List[Dict[str, str]], size: int = 10000
    ) -> List[List[float]]:
        cities = sorted(cities, key=lambda c: int(c["population"]), reverse=True)[:size]
        points = [[float(c["lng"]), float(c["lat"])] for c in cities]
        return points

    @staticmethod
    def km2deg(kms: float, radius: float = 6371) -> float:
        return kms / (2.0 * radius * np.pi / 360.0)

    @staticmethod
    def deg2km(deg: float, radius: float = 6371) -> float:
        return deg * (2.0 * radius * np.pi / 360.0)


class BoundedUniformSampler:
    def __init__(self, boundaries: shape = None) -> None:
        if boundaries is None:
            self.boundaries = self.get_country_boundaries()
        else:
            self.boundaries = boundaries

    def sample_point(self) -> List[float]:
        minx, miny, maxx, maxy = self.boundaries.bounds
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        p = Point(lon, lat)
        if self.boundaries.contains(p):
            return [p.x, p.y]
        else:
            return self.sample_point()

    @staticmethod
    def get_country_boundaries(
        download_root: str = os.path.expanduser("~/.cache/naturalearth"),
    ) -> shape:
        url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"  # noqa: E501
        filename = "ne_110m_admin_0_countries.shp"
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        sf = shapefile.Reader(os.path.join(download_root, filename))
        return shape(sf.shapes().__geo_interface__)


class OverlapError(Exception):
    pass


def sample_random_locations_rtree(
    idx: int,
    sampler: Any,
    radius: float,
    overlap_ratio: float,
    rtree_obj: index.Index = index.Index(),
) -> List[float]:
    """sample new coord and check overlap --- rtree"""
    # use rtree to avoid strong overlap
    count = 0
    bbox_radius = (1 - overlap_ratio) * radius / 1000
    while count < 1:
        new_coord = sampler.sample_point()  # (lon,lat) of top-10000 cities
        bbox = (
            new_coord[0] - sampler.km2deg(bbox_radius),
            new_coord[1] - sampler.km2deg(bbox_radius),
            new_coord[0] + sampler.km2deg(bbox_radius),
            new_coord[1] + sampler.km2deg(bbox_radius),
        )
        if list(rtree_obj.intersection(bbox)):
            continue
        rtree_obj.insert(idx, bbox)
        count += 1
        center_coord = [new_coord[0], new_coord[1]]
    return center_coord


def fix_random_seeds(seed: int = 42) -> None:
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path", type=str, default="./data/", help="dir to save data"
    )
    parser.add_argument(
        "--radius", type=float, default=1320, help="patch radius in meters"
    )
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=0.25,
        help="max overlap ratio between adjacent patches",
    )
    parser.add_argument(
        "--num_cities", type=int, default=10000, help="number of cities to sample"
    )
    parser.add_argument(
        "--std", type=int, default=50, help="std of gaussian distribution"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    parser.add_argument(
        "--indices_range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to download",
    )

    args = parser.parse_args()

    fix_random_seeds(seed=42)

    # initialize sampler
    sampler = GaussianSampler(num_cities=args.num_cities, std=args.std)

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

    rtree_coords = index.Index()
    bbox_radius = (
        (1 - args.overlap_ratio) * args.radius / 1000
    )  # allow little overlap between adjacent patches
    if args.resume:
        print("Load existing locations.")
        for i, key in enumerate(tqdm(ext_coords.keys())):
            c = ext_coords[key]
            rtree_coords.insert(
                i,
                (
                    c[0] - sampler.km2deg(bbox_radius),
                    c[1] - sampler.km2deg(bbox_radius),
                    c[0] + sampler.km2deg(bbox_radius),
                    c[1] + sampler.km2deg(bbox_radius),
                ),
            )

    start_time = time.time()

    indices = range(args.indices_range[0], args.indices_range[1])

    for idx in tqdm(indices):
        if str(idx) in ext_coords.keys():
            if args.resume:
                continue

        center_coord = sample_random_locations_rtree(
            idx, sampler, args.radius, args.overlap_ratio, rtree_coords
        )

        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            data = [idx, center_coord[0], center_coord[1]]
            writer.writerow(data)

    print(
        f"Sampled locations saved to {ext_path} in {time.time()-start_time:.2f} seconds."  # noqa: E501
    )
