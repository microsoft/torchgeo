import os
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.transform import Affine


def generate_fake_dataset(root_dir='data', num_train=2, num_val=1, num_test=1):
    """Generates a fake dataset for testing the SenBenchCloudS3 dataset class.

    Args:
        root_dir (str): Root directory where the fake dataset will be created.
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        num_test (int): Number of test samples.
    """
    # Create directories
    s3_olci_dir = os.path.join(root_dir, 's3_olci')
    cloud_multi_dir = os.path.join(root_dir, 'cloud_multi')
    cloud_binary_dir = os.path.join(root_dir, 'cloud_binary')
    os.makedirs(s3_olci_dir, exist_ok=True)
    os.makedirs(cloud_multi_dir, exist_ok=True)
    os.makedirs(cloud_binary_dir, exist_ok=True)

    # Generate filename components
    start_date = datetime(2020, 1, 1)

    def generate_samples(num_samples, offset=0):
        samples = []
        for i in range(num_samples):
            current_date = start_date + timedelta(days=offset + i)
            date_str = current_date.strftime('%Y%m%d')
            fname = f'S3_{i + offset:04d}____{date_str}_000000.tif'
            samples.append(fname)
        return samples

    # Generate sample filenames for each split
    # Create sample lists with sequential dates
    train_samples = generate_samples(num_train, 0)
    val_samples = generate_samples(num_val, num_train)
    test_samples = generate_samples(num_test, num_train + num_val)

    # Write CSV files
    def write_csv(split, samples):
        csv_path = os.path.join(root_dir, f'{split}.csv')
        with open(csv_path, 'w') as f:
            f.write('\n'.join(samples))

    write_csv('train', train_samples)
    write_csv('val', val_samples)
    write_csv('test', test_samples)

    # Generate all samples (train + val + test)
    all_samples = train_samples + val_samples + test_samples

    # Rasterio parameters
    height = width = 256
    transform = Affine.identity()  # Identity transform for simplicity
    crs = 'EPSG:4326'  # WGS84 coordinate system

    for sample in all_samples:
        # Generate fake Sentinel-3 OLCI image (21 bands)
        img_path = os.path.join(s3_olci_dir, sample)
        with rasterio.open(
            img_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=21,
            dtype=np.float32,
            transform=transform,
            crs=crs,
        ) as dst:
            for band in range(1, 22):
                data = np.random.rand(height, width).astype(np.float32)
                dst.write(data, band)

        # Generate multi-class cloud mask (values 0-5)
        multi_path = os.path.join(cloud_multi_dir, sample)
        with rasterio.open(
            multi_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            transform=transform,
            crs=crs,
        ) as dst:
            data = np.random.randint(0, 6, (height, width), dtype=np.uint8)
            dst.write(data, 1)

        # Generate binary cloud mask (values 0-2)
        binary_path = os.path.join(cloud_binary_dir, sample)
        with rasterio.open(
            binary_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.uint8,
            transform=transform,
            crs=crs,
        ) as dst:
            data = np.random.randint(0, 3, (height, width), dtype=np.uint8)
            dst.write(data, 1)


if __name__ == '__main__':
    generate_fake_dataset(root_dir='./senbench_cloud_s3')
