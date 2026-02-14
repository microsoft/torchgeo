#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import Point

np.random.seed(0)

DOMAIN_YEARS = {
    'D006-2020': {'TILE': 'FF-S1-14', 'COORDS': '5-5'},
    'D012-2019': {'TILE': 'AF-S1-27', 'COORDS': '5-10'},
    'D032-2019': {'TILE': 'AA-S1-V18', 'COORDS': '10-22'},
}

# Patch IDs matching FLAIRHUB _load_files format: D{region}-{year}_{tile}_{coords}
OFFICIAL_SPLITS_PATCH_IDS = [
    f'{domain_year}_{info["TILE"]}_{info["COORDS"]}'
    for domain_year, info in DOMAIN_YEARS.items()
]
# Toy data uses split_toy column (patch_id, split_toy, geometry)
OFFICIAL_SPLITS_SPLIT = ['train', 'valid', 'test']

# Modalities generated for all domain-years; others only for one domain-year (plotting).
REQUIRED_MODALITIES_PER_DOMAIN = [
    'AERIAL_LABEL-COSIA',
    'ALL_LABEL-LPIS',
    'SENTINEL2_TS',
    'AERIAL_RGBI',
]
PLOTTING_ONLY_DOMAIN_YEAR = 'D006-2020'


def create_official_splits_gdf() -> gpd.GeoDataFrame:
    """Creating the test data for the split file for the FLAIRHUB dataset."""
    gdf = gpd.GeoDataFrame(
        {
            'patch_id': OFFICIAL_SPLITS_PATCH_IDS,
            'split_1': OFFICIAL_SPLITS_SPLIT,
            'geometry': [Point(0, 0) for _ in range(len(OFFICIAL_SPLITS_PATCH_IDS))],
        }
    )
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def create_toy_splits_gdf() -> gpd.GeoDataFrame:
    """Creating the test data for the split file for the FLAIRHUB Toy dataset."""
    gdf = gpd.GeoDataFrame(
        {
            'patch_id': OFFICIAL_SPLITS_PATCH_IDS,
            'split_toy': OFFICIAL_SPLITS_SPLIT,
            'geometry': [Point(0, 0) for _ in range(len(OFFICIAL_SPLITS_PATCH_IDS))],
        }
    )
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf


MODALITIES = {
    'AERIAL_RGBI': {'channels': 4, 'dtype': np.uint8, 'range': (0, 200), 'size': 512},
    'AERIAL_LABEL-COSIA': {
        'channels': 1,
        'dtype': np.uint8,
        'range': (0, 18),
        'size': 512,
    },
    'ALL_LABEL-LPIS': {'channels': 3, 'dtype': np.uint8, 'range': (0, 22), 'size': 512},
    'DEM_ELEV': {'channels': 2, 'dtype': np.uint16, 'range': (300, 350), 'size': 512},
    'AERIAL-RLT_PAN': {
        'channels': 1,
        'dtype': np.uint8,
        'range': (0, 200),
        'size': 256,
    },
    'SPOT_RGBI': {'channels': 4, 'dtype': np.uint8, 'range': (0, 255), 'size': 64},
    'SENTINEL2_TS': {
        'channels': 20,
        'dtype': np.uint16,
        'range': (0, 10000),
        'size': 10,
    },
    'SENTINEL2_MSK-SC': {
        'channels': 20,
        'dtype': np.uint8,
        'range': (0, 2),
        'size': 10,
    },
    'SENTINEL1-ASC_TS': {
        'channels': 20,
        'dtype': np.uint16,
        'range': (0, 1000),
        'size': 10,
    },
    'SENTINEL1-DESC_TS': {
        'channels': 20,
        'dtype': np.uint16,
        'range': (0, 1000),
        'size': 10,
    },
}


def create_geotiff(
    path: Path, channels: int, dtype: np.dtype, value_range: tuple[int, int], size: int
) -> None:
    """Create a dummy GeoTIFF file with random data."""
    min_val, max_val = value_range
    data = np.random.randint(min_val, max_val, size=(channels, size, size), dtype=dtype)

    transform = Affine(0.4, 0.0, 1022361.6, 0.0, -0.4, 6313574.4)
    crs = CRS.from_epsg(2154)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size,
        width=size,
        count=channels,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress='lzw',
    ) as dst:
        for i in range(channels):
            dst.write(data[i], i + 1)


def create_modality_files(
    modality: str, domain_year: str, domain_info: dict[str, str], base_dir: Path
) -> Path:
    """Create all files for a specific modality and domain year."""
    props = MODALITIES[modality]
    tile = domain_info['TILE']
    coords = domain_info['COORDS']
    modality_dir = base_dir / f'{domain_year}_{modality}' / tile
    modality_dir.mkdir(parents=True, exist_ok=True)

    filename = f'{domain_year}_{modality}_{tile}_{coords}.tif'
    filepath = modality_dir / filename
    create_geotiff(
        filepath, props['channels'], props['dtype'], props['range'], props['size']
    )

    return modality_dir.parent


def create_zip_archive(modality: str, domain_year: str, base_dir: Path) -> Path:
    """Create a zip archive for a modality and domain year."""
    modality_dir = base_dir / f'{domain_year}_{modality}'
    zip_path = base_dir / f'{domain_year}_{modality}.zip'

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in modality_dir.rglob('*.tif'):
            zipf.write(file_path, file_path.relative_to(base_dir))

    return zip_path


def create_toy_dataset(base_dir: Path) -> Path:
    """Create the FLAIR-HUB_TOY dataset structure."""
    toy_dir = base_dir / 'FLAIR-HUB_TOY'
    toy_dir.mkdir(parents=True, exist_ok=True)

    for domain_year, domain_info in DOMAIN_YEARS.items():
        for modality in MODALITIES.keys():
            dir_year = (
                domain_year.replace(domain_year[-4:], '195X')
                if modality == 'AERIAL-RLT_PAN'
                else domain_year
            )
            source_dir = base_dir / f'{dir_year}_{modality}'
            if source_dir.exists():
                shutil.copytree(
                    source_dir, toy_dir / f'{dir_year}_{modality}', dirs_exist_ok=True
                )

    # Official splits gpkg for FLAIRHUBDataModule official_splits (e.g. test_segmentation)
    gpkg_dir = toy_dir / 'GLOBAL_ALL_MTD'
    gpkg_dir.mkdir(exist_ok=True)
    official_splits_gdf = create_toy_splits_gdf()
    official_splits_gdf.to_file(gpkg_dir / 'GLOBAL_ALL_MTD_SPLIT.gpkg', driver='GPKG')

    zip_path = base_dir / 'FLAIR-HUB_TOY_DATASET.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in toy_dir.rglob('*.tif'):
            zipf.write(
                file_path, Path('FLAIR-HUB_TOY') / file_path.relative_to(toy_dir)
            )
        for file_path in toy_dir.rglob('*.gpkg'):
            zipf.write(
                file_path, Path('FLAIR-HUB_TOY') / file_path.relative_to(toy_dir)
            )

    return zip_path


def main() -> None:
    """Generate test data for FLAIRHUB dataset and FLAIRHUB Toy dataset."""
    base_dir = Path(__file__).resolve().parent

    # Clean up
    for domain_year in DOMAIN_YEARS.keys():
        for modality in MODALITIES.keys():
            dir_path = base_dir / f'{domain_year}_{modality}'
            if dir_path.exists():
                shutil.rmtree(dir_path)
            zip_path = base_dir / f'{domain_year}_{modality}.zip'
            if zip_path.exists():
                zip_path.unlink()

    toy_dir = base_dir / 'FLAIR-HUB_TOY'
    if toy_dir.exists():
        shutil.rmtree(toy_dir)
    toy_zip = base_dir / 'FLAIR-HUB_TOY_DATASET.zip'
    if toy_zip.exists():
        toy_zip.unlink()

    # Create files: COSIA, LPIS, SENTINEL2_TS and AERIAL_RGBI for all domain-years; others for D006 only (plotting).
    for domain_year, domain_info in DOMAIN_YEARS.items():
        for modality in MODALITIES.keys():
            if (
                modality not in REQUIRED_MODALITIES_PER_DOMAIN
                and domain_year != PLOTTING_ONLY_DOMAIN_YEAR
            ):
                continue
            # AERIAL-RLT_PAN uses 195X suffix per FLAIRHUB spec
            dir_year = (
                domain_year.replace(domain_year[-4:], '195X')
                if modality == 'AERIAL-RLT_PAN'
                else domain_year
            )
            create_modality_files(modality, dir_year, domain_info, base_dir)
            zip_path = create_zip_archive(modality, dir_year, base_dir)
            print(f'Created: {zip_path}')

    # Create toy dataset
    toy_zip_path = create_toy_dataset(base_dir)
    print(f'Created: {toy_zip_path}')

    # GLOBAL_ALL_MTD for FLAIRHUB (non-toy) ensure_splits_available; real format uses split_1
    gpkg_dir = base_dir / 'GLOBAL_ALL_MTD'
    gpkg_dir.mkdir(exist_ok=True)
    create_official_splits_gdf().to_file(
        gpkg_dir / 'GLOBAL_ALL_MTD_SPLIT.gpkg', driver='GPKG'
    )
    mtd_zip = base_dir / 'GLOBAL_ALL_MTD.zip'
    with zipfile.ZipFile(mtd_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fp in gpkg_dir.rglob('*'):
            zipf.write(fp, fp.relative_to(base_dir))
    print(f'Created: {mtd_zip}')


if __name__ == '__main__':
    main()
