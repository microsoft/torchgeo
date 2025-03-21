# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench LC100Seg-S3 dataset."""

import os
from collections.abc import Callable, Sequence
from datetime import date

import cv2
import numpy as np
import rasterio
import torch
from pyproj import Transformer
from torch import Tensor

from .base import CopernicusBenchBase
from .utils import Path


class CopernicusBenchLC100SegS3(CopernicusBenchBase):
    """Copernicus-Bench LC100Cls-S3 dataset.

    The SenBench-LC100Seg-S3 dataset is a level-2 dataset from the SentinelBench benchmark.
    It contains Sentinel-3 OLCI images and land cover maps for the land cover segmentation task.
    It supports both static (1 image / location) and time series (1-4 images / location) mode, the former is used in the original benchmark.

    Dataset features:

    * task: semantic segmentation
    * # samples: 5181/1727/1727 (train/val/test, static mode)
    * image resolution: 96x96 (GSD 300m)
    * label resolution: 282x282 (GSD 100m)
    * # classes: 23


    Dataset format:

    * images: 21-band Sentinel-3 OLCI images (GeoTIFF)
    * labels: land cover maps (GeoTIFF)

    If you use this dataset in your research, please cite the following paper:

    * To be released soon


    """

    url = 'https://huggingface.co/datasets/wangyi111/SentinelBench/resolve/main/l2_lc100_s3/lc100_s3olci.zip'
    splits = ('train', 'val', 'test')
    label_filenames = {
        'train': 'lc100_multilabel-train.csv',
        'val': 'lc100_multilabel-val.csv',
        'test': 'lc100_multilabel-test.csv',
    }
    static_filenames = {
        'train': 'static_fnames-train.csv',
        'val': 'static_fnames-val.csv',
        'test': 'static_fnames-test.csv',
    }
    all_band_names = (
        'Oa01_radiance',
        'Oa02_radiance',
        'Oa03_radiance',
        'Oa04_radiance',
        'Oa05_radiance',
        'Oa06_radiance',
        'Oa07_radiance',
        'Oa08_radiance',
        'Oa09_radiance',
        'Oa10_radiance',
        'Oa11_radiance',
        'Oa12_radiance',
        'Oa13_radiance',
        'Oa14_radiance',
        'Oa15_radiance',
        'Oa16_radiance',
        'Oa17_radiance',
        'Oa18_radiance',
        'Oa19_radiance',
        'Oa20_radiance',
        'Oa21_radiance',
    )
    all_band_scale = (
        0.0139465,
        0.0133873,
        0.0121481,
        0.0115198,
        0.0100953,
        0.0123538,
        0.00879161,
        0.00876539,
        0.0095103,
        0.00773378,
        0.00675523,
        0.0071996,
        0.00749684,
        0.0086512,
        0.00526779,
        0.00530267,
        0.00493004,
        0.00549962,
        0.00502847,
        0.00326378,
        0.00324118,
    )
    rgb_bands = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')

    LC100_CLSID = {
        0: 0,  # unknown
        20: 1,  # shrubs
        30: 2,  # herbaceous vegetation
        40: 3,  # cultivated and managed vegetation/agriculture
        50: 4,  # urban / built-up
        60: 5,  # bare / sparse vegetation
        70: 6,  # snow and ice
        80: 7,  # permanent water bodies
        90: 8,  # herbaceous wetland
        100: 9,  # moss and lichen
        111: 10,  # closed forest, evergreen needle leaf
        112: 11,  # closed forest, evergreen broad leaf
        113: 12,  # closed forest, deciduous needle leaf
        114: 13,  # closed forest, deciduous broad leaf
        115: 14,  # closed forest, mixed
        116: 15,  # closed forest, other
        121: 16,  # open forest, evergreen needle leaf
        122: 17,  # open forest, evergreen broad leaf
        123: 18,  # open forest, deciduous needle leaf
        124: 19,  # open forest, deciduous broad leaf
        125: 20,  # open forest, mixed
        126: 21,  # open forest, other
        200: 22,  # oceans, seas
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_band_names,
        mode='static',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.download = download

        assert split in ['train', 'val', 'test']

        self.bands = bands
        self.band_indices = [
            (self.all_band_names.index(b) + 1)
            for b in bands
            if b in self.all_band_names
        ]

        self.mode = mode
        self.img_dir = os.path.join(self.root, 's3_olci')
        self.label_dir = os.path.join(self.root, 'lc100')

        if self.mode == 'static':
            self.static_csv = os.path.join(self.root, self.static_filenames[split])
            with open(self.static_csv) as f:
                lines = f.readlines()
                self.static_img = {}
                for line in lines:
                    pid = line.strip().split(',')[0]
                    img_fname = line.strip().split(',')[1]
                    self.static_img[pid] = img_fname

        self.pids = list(self.static_img.keys())

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (8 * 300 / 1000) ** 2  # patchsize 8 pix, gsd 300m

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        images, meta_infos = self._load_image(index)
        label = self._load_target(index)
        if self.mode == 'static':
            sample = {'image': images[0], 'mask': label, 'meta': meta_infos[0]}
        elif self.mode == 'series':
            sample = {'image': images, 'mask': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index):
        pid = self.pids[index]
        s3_path = os.path.join(self.img_dir, pid)
        if self.mode == 'static':
            img_fname = self.static_img[pid]
            s3_paths = [os.path.join(s3_path, img_fname)]
        else:
            img_fnames = os.listdir(s3_path)
            s3_paths = []
            for img_fname in img_fnames:
                s3_paths.append(os.path.join(s3_path, img_fname))

        imgs = []
        img_paths = []
        meta_infos = []
        for img_path in s3_paths:
            with rasterio.open(img_path) as src:
                img = src.read()
                img[np.isnan(img)] = 0
                chs = []
                for b in range(21):
                    ch = img[b] * self.all_band_scale[b]
                    # ch = cv2.resize(ch, (96,96), interpolation=cv2.INTER_CUBIC)
                    ch = cv2.resize(
                        ch, (282, 282), interpolation=cv2.INTER_CUBIC
                    )  # to match label size
                    chs.append(ch)
                img = np.stack(chs)
                img = torch.from_numpy(img).float()

                # get lon, lat
                cx, cy = src.xy(src.height // 2, src.width // 2)
                if src.crs.to_string() != 'EPSG:4326':
                    # convert to lon, lat
                    crs_transformer = Transformer.from_crs(
                        src.crs, 'epsg:4326', always_xy=True
                    )
                    lon, lat = crs_transformer.transform(cx, cy)
                else:
                    lon, lat = cx, cy
                # get time
                img_fname = os.path.basename(img_path)
                date_str = img_fname.split('_')[1][:8]
                date_obj = date(
                    int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
                )
                delta = (date_obj - self.reference_date).days
                # this is what CopernicusFM requires
                # meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
                # meta_info = torch.from_numpy(meta_info)
                # this is more general
                meta_info = {
                    'lon': torch.tensor(lon),
                    'lat': torch.tensor(lat),
                    'delta-t': torch.tensor(delta),  # days since 1970-01-01
                    'area-p': torch.tensor(self.patch_area),  # ViT patch area in km^2
                }

            imgs.append(img)
            img_paths.append(img_path)
            meta_infos.append(meta_info)

        if self.mode == 'series':
            # pad to 4 images if less than 4
            while len(imgs) < 4:
                imgs.append(img)
                img_paths.append(img_path)
                meta_infos.append(meta_info)

        return imgs, meta_infos  # return list of images and meta_infos

    def _load_target(self, index):
        pid = self.pids[index]
        label_path = os.path.join(self.label_dir, pid + '.tif')

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label = cv2.resize(
                label, (282, 282), interpolation=cv2.INTER_NEAREST
            )  # 0-650
            # label = cv2.resize(label, (96,96), interpolation=cv2.INTER_NEAREST) # 0-650
            label_new = np.zeros_like(label)
            # remap classes
            for k, v in self.LC100_CLSID.items():
                label_new[label == k] = v
            labels = torch.from_numpy(label_new.astype('int64'))

        return labels
