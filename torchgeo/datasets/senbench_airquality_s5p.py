import kornia as K
import torch
from torchgeo.datasets.geo import NonGeoDataset
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
import cv2
from pyproj import Transformer
from datetime import date
from typing import TypeAlias, ClassVar
import pathlib

import logging

logging.getLogger("rasterio").setLevel(logging.ERROR)
Path: TypeAlias = str | os.PathLike[str]

class SenBenchAirQualityS5P(NonGeoDataset):
    """Parent class for SenBench-AQ-NO2-S5P and SenBench-AQ-O3-S5P datasets.

    The SenBench-AQ-NO2-S5P and SenBench-AQ-O3-S5P datasets are level-3 datasets from the SentinelBench benchmark.
    It contains Sentinel-5P NO2/O3 images and EEA NO2/O3 maps for the air pollutants regression task.
    It supports both static (1 image / location, annual mean) and time series (~4 images / location, seasonal mean) mode, the former is used in the original benchmark.

    Dataset features:
    * task: dense regression
    * # samples: 1480/493/494 (train/val/test, static mode)
    * image resolution: 56x56 (GSD 1km)
    * label resolution: 56x56 (GSD 1km)
    * mode: annual (static) or seasonal (time series)
    * modality: NO2 or O3

    Dataset format:
    * images: 1 band Sentinel-5P NO2/O3 images (GeoTIFF)
    * labels: EEA NO2/O3 maps (GeoTIFF)

    If you use this dataset in your research, please cite the following paper:

    * To be released soon


    """

    url = 'https://huggingface.co/datasets/wangyi111/SentinelBench/resolve/main/l3_airquality_s5p/airquality_s5p.zip'
    splits = ('train', 'val', 'test')
    split_fnames = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        modality = 'no2', # or 'o3'
        mode = 'annual', # or 'seasonal'
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download
        #self.checksum = checksum

        assert split in ['train', 'val', 'test']

        self.modality = modality
        self.mode = mode

        if self.mode == 'annual':
            mode_dir = 's5p_annual'
        elif self.mode == 'seasonal':
            mode_dir = 's5p_seasonal'

        self.img_dir = os.path.join(root, modality, mode_dir)
        self.label_dir = os.path.join(root, modality, 'label_annual')
        
        self.split_csv = os.path.join(self.root, modality, self.split_fnames[split])
        with open(self.split_csv, 'r') as f:
            lines = f.readlines()
            self.pids = []
            for line in lines:
                self.pids.append(line.strip())

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (4*1)**2 # patchsize 4 pix, gsd 1km

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        label = self._load_target(index)
        if self.mode == 'annual':
            sample = {'image': images[0], 'groundtruth': label, 'meta': meta_infos[0]}
        elif self.mode == 'seasonal':
            sample = {'image': images, 'groundtruth': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        pid = self.pids[index]
        s5p_path = os.path.join(self.img_dir, pid)

        img_fnames = os.listdir(s5p_path)
        s5p_paths = []
        for img_fname in img_fnames:
            s5p_paths.append(os.path.join(s5p_path, img_fname))
        
        imgs = []
        meta_infos = []
        for img_path in s5p_paths:
            with rasterio.open(img_path) as src:
                img = src.read(1)
                img[np.isnan(img)] = 0
                img = cv2.resize(img, (56,56), interpolation=cv2.INTER_CUBIC)
                img = torch.from_numpy(img).float()
                img = img.unsqueeze(0)

                # get lon, lat
                cx,cy = src.xy(src.height // 2, src.width // 2)
                if src.crs.to_string() != 'EPSG:4326':
                    # convert to lon, lat
                    crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                    lon, lat = crs_transformer.transform(cx,cy)
                else:
                    lon, lat = cx, cy
                # get time
                img_fname = os.path.basename(img_path)
                date_str = img_fname.split('_')[0][:10]
                date_obj = date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]))
                delta = (date_obj - self.reference_date).days
                #meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
                #meta_info = torch.from_numpy(meta_info)
                meta_info = {
                    'lon': torch.tensor(lon), 
                    'lat': torch.tensor(lat), 
                    'delta-t': torch.tensor(delta), # days since 1970-01-01
                    'area-p': torch.tensor(self.patch_area), # ViT patch area in km^2
                    }

            imgs.append(img)
            meta_infos.append(meta_info)

        if self.mode == 'seasonal':
            # pad to 4 images if less than 4
            while len(imgs) < 4:
                imgs.append(img)
                meta_infos.append(meta_info)

        return imgs, meta_infos # return list of images and meta_infos

    def _load_target(self, index):

        pid = self.pids[index]
        label_path = os.path.join(self.label_dir, pid+'.tif')

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label = cv2.resize(label, (56,56), interpolation=cv2.INTER_NEAREST) # 0-650
            # label contains -inf
            label[label<-1e10] = np.nan
            label[label>1e10] = np.nan
            label = torch.from_numpy(label.astype('float32'))

        return label
    

class SenBenchAQNO2S5P(SenBenchAirQualityS5P):
    """SenBench-AQ-NO2-S5P dataset."""
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        mode = 'annual', # or 'seasonal'
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, 'no2', mode, transforms, download)


class SenBenchAQO3S5P(SenBenchAirQualityS5P):
    """SenBench-AQ-O3-S5P dataset."""
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        mode = 'annual', # or 'seasonal'
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, 'o3', mode, transforms, download)