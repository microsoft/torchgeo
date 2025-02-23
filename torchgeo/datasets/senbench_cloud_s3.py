import torch
from torchgeo.datasets.geo import NonGeoDataset
import os
from collections.abc import Callable, Sequence
from torch import Tensor
import numpy as np
import rasterio
from pyproj import Transformer
from datetime import date
from typing import TypeAlias, ClassVar
from .utils import Path, download_and_extract_archive, extract_archive

import logging

logging.getLogger("rasterio").setLevel(logging.ERROR)

class SenBenchCloudS3(NonGeoDataset):
    """SenBench-Cloud-S3 dataset.

    The SenBench-Cloud-S3 dataset is a level-1 dataset from the SentinelBench benchmark.
    It contains Sentinel-3 OLCI images, multi-class cloud masks, and binary cloud masks for the cloud segmentation task.

    Dataset features:

    * task: semantic segmentation
    * # samples: 1197/399/399 (train/val/test)
    * image resolution: 256x256
    * # classes: 5 (multi-class) / 2 (binary)

    Dataset format:

    * images: 21-band Sentinel-3 OLCI images (GeoTIFF)
    * labels: multi-class cloud masks (GeoTIFF)
    * binary_labels: binary cloud masks (GeoTIFF)

    If you use this dataset in your research, please cite the following paper:

    * To be released soon


    """
    url = 'https://huggingface.co/datasets/wangyi111/SentinelBench/resolve/main/l3_biomass_s3/biomass_s3olci.zip'
    
    splits = ('train', 'val', 'test')

    split_filenames = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }

    all_band_names = (
        'Oa01_radiance', 'Oa02_radiance', 'Oa03_radiance', 'Oa04_radiance', 'Oa05_radiance', 'Oa06_radiance', 'Oa07_radiance',
        'Oa08_radiance', 'Oa09_radiance', 'Oa10_radiance', 'Oa11_radiance', 'Oa12_radiance', 'Oa13_radiance', 'Oa14_radiance',
        'Oa15_radiance', 'Oa16_radiance', 'Oa17_radiance', 'Oa18_radiance', 'Oa19_radiance', 'Oa20_radiance', 'Oa21_radiance',
    )

    all_band_scale = (
        0.0139465,0.0133873,0.0121481,0.0115198,0.0100953,0.0123538,0.00879161,
        0.00876539,0.0095103,0.00773378,0.00675523,0.0071996,0.00749684,0.0086512,
        0.00526779,0.00530267,0.00493004,0.00549962,0.00502847,0.00326378,0.00324118)
    
    rgb_bands = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')

    Cls_index_binary = {
        'invalid': 0, # --> 255 should be ignored during training
        'clear': 1, # --> 0
        'cloud': 2, # --> 1
    }

    Cls_index_multi = {
        'invalid': 0, # --> 255 should be ignored during training
        'clear': 1, # --> 0
        'cloud-sure': 2, # --> 1
        'cloud-ambiguous': 3, # --> 2
        'cloud shadow': 4, # --> 3
        'snow and ice': 5, # --> 4
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_band_names,
        mode = 'multi',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:

        self.root = root
        self.transforms = transforms
        self.download = download

        assert split in ['train', 'val', 'test']

        self.bands = bands
        self.band_indices = [(self.all_band_names.index(b)+1) for b in bands if b in self.all_band_names]

        self.mode = mode
        self.img_dir = os.path.join(self.root, 's3_olci')
        self.label_dir = os.path.join(self.root, 'cloud_'+mode)
        
        self.split_csv = os.path.join(self.root, self.split_filenames[split])
        self.fnames = []
        with open(self.split_csv, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fname = line.strip()
                self.fnames.append(fname)

        self.reference_date = date(1970, 1, 1)
        self.patch_area = (8*300/1000)**2 # patchsize 8 pix, gsd 300m

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):

        images, meta_infos = self._load_image(index)
        label = self._load_target(index)
        sample = {'image': images, 'mask': label, 'meta': meta_infos}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def _load_image(self, index):

        fname = self.fnames[index]
        s3_path = os.path.join(self.img_dir, fname)
        
        with rasterio.open(s3_path) as src:
            img = src.read(self.band_indices)
            img[np.isnan(img)] = 0
            chs = []
            for b in range(21):
                ch = img[b]*self.all_band_scale[b]
                chs.append(ch)
            img = np.stack(chs)
            img = torch.from_numpy(img).float()

            # get lon, lat
            cx,cy = src.xy(src.height // 2, src.width // 2)
            if src.crs.to_string() != 'EPSG:4326':
                # convert to lon, lat
                crs_transformer = Transformer.from_crs(src.crs, 'epsg:4326', always_xy=True)
                lon, lat = crs_transformer.transform(cx,cy)
            else:
                lon, lat = cx, cy
            # get time
            img_fname = os.path.basename(s3_path)
            date_str = img_fname.split('____')[1][:8]
            date_obj = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            delta = (date_obj - self.reference_date).days 
            # this is what CopernicusFM requires
            #meta_info = np.array([lon, lat, delta, self.patch_area]).astype(np.float32)
            #meta_info = torch.from_numpy(meta_info)
            # this is more general
            meta_info = {
                'lon': torch.tensor(lon), 
                'lat': torch.tensor(lat), 
                'delta-t': torch.tensor(delta), # days since 1970-01-01
                'area-p': torch.tensor(self.patch_area), # ViT patch area in km^2
                }

        return img, meta_info

    def _load_target(self, index):

        fname = self.fnames[index]
        label_path = os.path.join(self.label_dir, fname)

        with rasterio.open(label_path) as src:
            label = src.read(1)
            label[label==0] = 256
            label = label - 1
            labels = torch.from_numpy(label).long()

        return labels