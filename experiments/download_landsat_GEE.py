#!/usr/bin/env python3

''' Sample and download Landsat 8 tiles with Google Earth Engine
https://developers.google.com/earth-engine/datasets/catalog/landsat-8

#### run the script:
### option 1: match ssl4eo-s12, and fill unmatched locations with newly sampled locations
# match ssl4eo-s12 ids, unavailable ids skip
!python download_landsat.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --match_file ssl4eo-s12_coords_v1.csv --indices_range 0 250000

# fill unmatched ids with rtree overlap search
!python download_landsat.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --resume ./data/checked_locations.csv --overlap_check rtree --indices_range 0 250000

### option 2: resample new locations
# (op1) resample new ids with rtree overlap search
!python download_landsat.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --overlap_check rtree --indices_range 0 250000

# (op2) resample new ids with grid overlap search
!python download_landsat.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --overlap_check grid --indices_range 0 250000

### (optional) resume from interruption
!python download_landsat.py --save_path ./data --num_workers 8 --cloud_pct 20 --log_freq 100 --resume ./data/checked_locations.csv --overlap_check rtree --indices_range 0 250000


#### default settings:
product: Landsat 8 TOA
bands: B1-B11, QA_PIXEL
cloud_threshold: 20
patch_size: 176x176 for 15m resolution, 88x88 for 30m resolution (2640mx2640m)
dtype: float32 (value range 0-1)
period: 2 months buffer on 4 reference season dates in 2020/2021

'''


import argparse
import csv
import json
from multiprocessing.dummy import Pool, Lock
import os
from collections import OrderedDict
import time
from datetime import datetime, timedelta, date
import warnings
warnings.simplefilter('ignore', UserWarning)

import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine
from torchvision.datasets.utils import download_and_extract_archive
import shapefile
from shapely.geometry import shape, Point

import pickle
import pdb
import math

from rtree import index
import sys
from tqdm import tqdm


ALL_BANDS_SR = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'] # Q: which product to use
ALL_BANDS_TOA = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'QA_PIXEL'] # Q: which product to use


''' samplers to get locations of interest points'''
class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()


class UniformSampler(GeoSampler):

    def sample_point(self):
        #fix_random_seeds()
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-90, 90)
        return [lon, lat]


class GaussianSampler(GeoSampler):

    def __init__(self, interest_points=None, num_cities=1000, std=20):
        if interest_points is None:
            cities = self.get_world_cities()
            self.interest_points = self.get_interest_points(cities,size=num_cities)
        else:
            self.interest_points = interest_points
        self.std = std

    def sample_point(self,idx):
        #pdb.set_trace()
        
        #rng = np.random.default_rng(seed=idx)
        rng = np.random.default_rng()
        point = rng.choice(self.interest_points)
        std = self.km2deg(self.std)
        #fix_random_seeds(idx)
        lon, lat = np.random.normal(loc=point, scale=[std, std])
        return [lon, lat]

    @staticmethod
    def get_world_cities(download_root=os.path.expanduser('./world_cities/')):
        url = 'https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip'
        filename = 'worldcities.csv'
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        with open(os.path.join(download_root, filename),encoding='UTF-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            cities = []
            for row in reader:
                row['population'] = row['population'].replace('.', '') if row['population'] else '0'
                cities.append(row)
        return cities

    @staticmethod
    def get_interest_points(cities, size=10000):
        cities = sorted(cities, key=lambda c: int(c['population']), reverse=True)[:size]
        points = [[float(c['lng']), float(c['lat'])] for c in cities]
        return points

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)
    @staticmethod    
    def deg2km(deg, radius=6371):
        return deg * (2.0 * radius * np.pi / 360.0)


class BoundedUniformSampler(GeoSampler):

    def __init__(self, boundaries=None):
        if boundaries is None:
            self.boundaries = self.get_country_boundaries()
        else:
            self.boundaries = boundaries

    def sample_point(self):
        minx, miny, maxx, maxy = self.boundaries.bounds
        #fix_random_seeds()
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        p = Point(lon, lat)
        if self.boundaries.contains(p):
            return [p.x, p.y]
        else:
            return self.sample_point()

    @staticmethod
    def get_country_boundaries(download_root=os.path.expanduser('~/.cache/naturalearth')):
        url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip'
        filename = 'ne_110m_admin_0_countries.shp'
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        sf = shapefile.Reader(os.path.join(download_root, filename))
        return shape(sf.shapes().__geo_interface__)


class OverlapError(Exception):
    pass


def date2str(date):
    return date.strftime('%Y-%m-%d')

def get_period(date, days=5):
    date1 = date - timedelta(days=days / 2)
    date2 = date + timedelta(days=days / 2)    
    date3 = date1 - timedelta(days=365)
    date4 = date2 - timedelta(days=365)     
    return date2str(date1), date2str(date2), date2str(date3), date2str(date4) # two-years buffer


'''get collection and remove clouds from ee'''
def maskS2clouds(image):
    qa = image.select('QA_PIXEL')
    # Bits 3 and 4 are clouds and cloud shadow, respectively. Q: which bits to use
    cloudBitMask = 1 << 3
    #shadowBitMask = 1 << 4
    # flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    #mask = mask.bitwiseAnd(shadowBitMask).eq(0)
    return image.updateMask(mask)
    
def get_collection(cloud_pct=10):
    #collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
    collection = collection.filter(ee.Filter.lt('CLOUD_COVER', cloud_pct)) # Q: there's also cloud cover land
    #collection = collection.map(maskS2clouds) ## don't apply cloud mask, instead download it as extra band
    return collection


''' filter periods and region '''
def filter_collection(collection, coords, period=None):
    filtered = collection
    if period is not None:
        #filtered = filtered.filterDate(*period)  # filter time, Q: 1 year? 2 years?
        filtered = filtered.filter(ee.Filter.Or(
            ee.Filter.date(period[0],period[1]), 
            ee.Filter.date(period[2],period[3]),
            )
        )
    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region

    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f'ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}.')
    return filtered


''' fine adjust coordinates '''
def center_crop(img, out_size):
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = int((image_height - crop_height + 1) * 0.5)
    crop_left = int((image_width - crop_width + 1) * 0.5)
    return img[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]


def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [coords[0][0] + ((xoff + new_size[1]) * xres), coords[0][1] - ((yoff + new_size[0]) * yres)]
    ]

''' get metatdata '''
def get_properties(image):
    #### this is too slow, don't use it
    #properties = {}
    #for property in image.propertyNames().getInfo():
    #    properties[property] = image.get(property)
    #return ee.Dictionary(properties).getInfo()
    
    return image.getInfo()


''' download a patch '''
def get_patch(collection, coords, radius, bands=None, crop=None):
    image = collection.sort('system:time_start', False).first()  # get most recent
    region = ee.Geometry.Point(coords).buffer(radius).bounds() # sample region bound
    patch = image.select(*bands).sampleRectangle(region,defaultValue=0) # out-of-boundary values set to 0 --> later need to percentile out
    features = patch.getInfo()  # the actual download, this is most time-consuming
    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features['properties'][band])
        if crop is not None and band!='QA_PIXEL':
            img = center_crop(img, out_size=crop[band])
        raster[band] = img.astype('float32') # has to be checked if change product

    coords = np.array(features['geometry']['coordinates'][0])
    coords = [
        [coords[:, 0].min(), coords[:, 1].max()],
        [coords[:, 0].max(), coords[:, 1].min()]
    ]
    if crop is not None:
        band = bands[0]
        old_size = (len(features['properties'][band]), len(features['properties'][band][0]))
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict({
        'raster': raster,
        'coords': coords,
        'metadata': get_properties(image)
    })


''' sample new coord, check overlap, and get data --- rtree (option1) '''
def get_random_patches_rtree(idx, collections, bands, crops, sampler, dates, radius, debug=False, rtree_obj=None):
    ## (lon,lat) of top-10000 cities
    coords = sampler.sample_point(idx)
    
    # use rtree to avoid strong overlap
    try:
        new_coord = (coords[0],coords[1])
        for i in rtree_obj.nearest(new_coord, objects=True):
            distance = np.sqrt(sampler.deg2km(abs(new_coord[0]-i.bbox[2]))**2 + sampler.deg2km(abs(new_coord[1]-i.bbox[3]))**2)
            if distance < (1.5 * radius/1000):
                raise OverlapError
        rtree_obj.insert(len(rtree_obj)-1, (new_coord[0], new_coord[1], new_coord[0], new_coord[1]))

    except OverlapError:
        patches_s2c, center_coord = get_random_patches_rtree(idx, collections, bands, crops, sampler, dates, radius, debug, rtree_obj)
    
    ## random +- 30 days of random days within 1 year from the reference dates
    #fix_random_seeds(idx)
    #delta = timedelta(days=np.random.randint(365))
    periods = [get_period(date, days=60) for date in dates]

    collection_s2c = collections['TOA']
    bands_s2c = bands['TOA']
    crop_s2c = crops['TOA']

    try:        
        filtered_collections_s2c = [filter_collection(collection_s2c, coords, p) for p in periods]
        patches_s2c = [get_patch(c, coords, radius, bands=bands_s2c, crop=crop_s2c) for c in filtered_collections_s2c]        
        center_coord = coords

    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        rtree_obj.insert(len(rtree_obj)-1, (new_coord[0], new_coord[1], new_coord[0], new_coord[1])) # prevent from sampling an old coord that doesn't fit the collection        
        patches_s2c, center_coord = get_random_patches_rtree(idx, collections, bands, crops, sampler, dates, radius, debug, rtree_obj)
   
    return patches_s2c, center_coord

''' sample new coord, check overlap, and get data --- grid (option2) '''
def get_random_patches_grid(idx, collections, bands, crops, sampler, dates, radius, debug=False, grid_dict={}):
    ## (lon,lat) of top-10000 cities
    coords = sampler.sample_point(idx)
    
    new_coord = (coords[0],coords[1])
    gridIndex = (math.floor(new_coord[0]+180),math.floor(new_coord[1]+90))
    # use grid to avoid strong overlap
    try:
        if not gridIndex in grid_dict.keys():
            grid_dict[gridIndex] = {new_coord}
        else:
            for coord in grid_dict[gridIndex]:
                distance = np.sqrt(sampler.deg2km(abs(new_coord[0]-coord[0]))**2 + sampler.deg2km(abs(new_coord[1]-coord[1]))**2)
                if distance < (1.5 * radius/1000):
                    raise OverlapError
            grid_dict[gridIndex].add(new_coord)
       
    except OverlapError:
        patches_s2c, center_coord = get_random_patches_grid(idx, collections, bands, crops, sampler, dates, radius, debug, grid_dict=grid_dict)
       
    ## random +- 30 days of random days within 1 year from the reference dates
    #fix_random_seeds(idx)
    #delta = timedelta(days=np.random.randint(365))
    periods = [get_period(date, days=60) for date in dates]

    collection_s2c = collections['TOA']
    bands_s2c = bands['TOA']
    crop_s2c = crops['TOA']

    try:
        
        filtered_collections_s2c = [filter_collection(collection_s2c, coords, p) for p in periods]
        patches_s2c = [get_patch(c, coords, radius, bands=bands_s2c, crop=crop_s2c) for c in filtered_collections_s2c]               
        center_coord = coords

    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        grid_dict[gridIndex].add(new_coord)
        patches_s2c, center_coord = get_random_patches_grid(idx, collections, bands, crops, sampler, dates, radius, debug, grid_dict=grid_dict)

    return patches_s2c, center_coord

''' sample new coord, check overlap, and get data --- match from existing coords (option3) '''
def get_random_patches_match(idx, collections, bands, crops, sampler, dates, radius, debug=False, match_coords={}):
    ## (lon,lat) of idx patch
    coords = match_coords[str(idx)]
        
    ## random +- 30 days of random days within 1 year from the reference dates
    #fix_random_seeds(idx)
    #delta = timedelta(days=np.random.randint(365))
    periods = [get_period(date, days=60) for date in dates]

    collection_s2c = collections['TOA']
    bands_s2c = bands['TOA']
    crop_s2c = crops['TOA']

    try:
        filtered_collections_s2c = [filter_collection(collection_s2c, coords, p) for p in periods]
        patches_s2c = [get_patch(c, coords, radius, bands=bands_s2c, crop=crop_s2c) for c in filtered_collections_s2c]               

    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        #patches_s2c, center_coord = get_random_patches_match(idx, collections, bands, crops, sampler, dates, radius, debug, ext_coords, rtree_obj)
        return None, coords

    return patches_s2c, coords


''' function to save geotiff '''
def save_geotiff(img, coords, filename):
    #pdb.set_trace()
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(coords[0][0] - xres / 2, coords[0][1] + yres / 2) * Affine.scale(xres, -yres)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': channels,
        'crs': '+proj=latlong',
        'transform': transform,
        'dtype': img.dtype,
        'compress': 'None'
    }
    with rasterio.open(filename, 'w', **profile) as f:
        f.write(img.transpose(2, 0, 1))

''' save patch to file '''
def save_patch(raster, coords, metadata, path):
    patch_id = metadata['properties']['system:index']
    patch_path = os.path.join(path, patch_id)
    os.makedirs(patch_path, exist_ok=True)

    for band, img in raster.items():
        save_geotiff(img, coords, os.path.join(patch_path, f'{band}.tif'))

    with open(os.path.join(patch_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)


''' multiprocessing counter '''
class Counter:

    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value


def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./data/') # dir to save data
    parser.add_argument('--num_cities', type=int, default=10000)
    parser.add_argument('--std', type=int, default=50)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cloud_pct', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_freq', type=int, default=10) # print frequency

    parser.add_argument('--resume', type=str, default=None) # resume from existing coordinates
    # op1: match ssl4eo coordinates and indexes
    parser.add_argument('--match_file', type=str, default=None)
    # op2-3: resample, grid or rtree based overlap check. grid is faster but allows boundary overlap; rtree is slower but completely avoid overlap
    parser.add_argument('--overlap_check', type=str, default='rtree',choices=['grid','rtree',None])
    parser.add_argument('--indices_range', type=int, nargs=2, default=[0,250000]) # range of download indices --> number of locations

    args = parser.parse_args()
    #args = parser.parse_args('')
    #args.match_file = 'center_coords_v1_new.csv'
    #args.overlap_check = None

    fix_random_seeds(seed=42)

    ## initialize ee
    ee.Initialize()
    ## get data collection (remove clouds)
    collection_TOA = get_collection(cloud_pct=args.cloud_pct)
    collections = {'TOA': collection_TOA}

    ## initialize sampler
    sampler = GaussianSampler(num_cities=args.num_cities, std=args.std)
    # sampler = BoundedUniformSampler()

    reference = date.fromisoformat('2021-09-22')
    date1 = date.fromisoformat('2021-06-21')
    date2 = date.fromisoformat('2021-03-20')
    date3 = date.fromisoformat('2020-12-21')
    dates = [reference, date1, date2, date3] # 4 reference dates for 4 seasons, with 2-year buffer

    # Q: image size
    radius = 1320 # corresponds to a 2640mx2640m patch in SSL4EO-S12
    crop30 = (88, 88) # patch size
    crop15 = (176, 176) 
    #crop_SR = {'SR_B1': crop30, 'SR_B2': crop30, 'SR_B3': crop30, 'SR_B4': crop30, 'SR_B5': crop30, 'SR_B6': crop30, 'SR_B7': crop30, 'QA_PIXEL': None}
    crop_TOA = {'B1': crop30, 'B2': crop30, 'B3': crop30, 'B4': crop30, 'B5': crop30, 'B6': crop30, 'B7': crop30, 'B8': crop15, 'B9': crop15, 'B10': crop30, 'B11': crop30, 'QA_PIXEL': None}

    crops = {'TOA': crop_TOA}
    bands = {'TOA': ALL_BANDS_TOA}

    ### if resume
    ext_coords = {}
    ext_flags = {}
    if args.resume:
        ext_path = args.resume   
        with open(ext_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2) # lon, lat
                ext_flags[key] = int(row[3]) # success or not
    else:
        ext_path = os.path.join(args.save_path,'checked_locations.csv')
    
    ### if match from exisiting coords (e.g. SSL4EO-S12)
    if args.match_file:
        match_coords = {}
        with open(args.match_file, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                match_coords[key] = (val1, val2) # lon, lat
    ### else need to check overlap, build the grid or rtree from existing coordinates
    elif args.overlap_check is not None:
        grid_dict = {}
        rtree_coords = index.Index()
        if args.resume:
            print('Load existing locations.')
            for i, key in enumerate(tqdm(ext_coords.keys())):
                c = ext_coords[key]
                rtree_coords.insert(i, (c[0], c[1], c[0], c[1]))
                gridIndex = (math.floor(c[0]+180),math.floor(c[1]+90))
                if not gridIndex in grid_dict.keys():
                    grid_dict[gridIndex] = {c}
                else:
                    grid_dict[gridIndex].add(c)
    else:
        raise NotImplementedError


    start_time = time.time()
    counter = Counter()

    ### downloading, define worker for multiprocess     
    def worker(idx):

        if str(idx) in ext_coords.keys():
            if args.match_file: # skip all processed ids
                #print('Already processed:',idx)
                return
            else:
                if ext_flags[str(idx)]!=0: # only skip downloaded ids
                    return

        if args.match_file:
            patches_SR, center_coord = get_random_patches_match(idx,collections, bands, crops, sampler, dates, radius=radius, debug=args.debug, match_coords=match_coords)
        elif args.overlap_check=='rtree':
            patches_SR, center_coord = get_random_patches_rtree(idx,collections, bands, crops, sampler, dates, radius=radius, debug=args.debug, rtree_obj=rtree_coords)
        elif args.overlap_check=='grid':
            patches_SR, center_coord = get_random_patches_grid(idx,collections, bands, crops, sampler, dates, radius=radius, debug=args.debug, grid_dict=grid_dict)
        else:
            raise NotImplementedError

        if patches_SR is not None:
            if args.save_path is not None:
                location_path_SR = os.path.join(args.save_path, 'TOA', f'{idx:06d}')
                os.makedirs(location_path_SR, exist_ok=True)
                for patch in patches_SR:
                    save_patch(
                        raster=patch['raster'],
                        coords=patch['coords'],
                        metadata=patch['metadata'],
                        path=location_path_SR,
                    )                    
            count = counter.update(1)
            if count % args.log_freq == 0:
                print(f'Downloaded {count} locations in {time.time() - start_time:.3f}s.')
        else:
            print('no suitable image for location %d.' % (idx))

        ## add to existing checked locations            
        with open(ext_path, 'a') as f:
            writer = csv.writer(f)
            if patches_SR is not None:
                if args.match_file:
                    success = 2
                else:
                    success = 1
            else:
                success = 0
            data = [idx, center_coord[0], center_coord[1], success]
            writer.writerow(data)

        return
          
    ### set indices
    if args.match_file is not None:
        indices = []
        for key in match_coords.keys():
            indices.append(int(key))
        indices = indices[args.indices_range[0]:args.indices_range[1]]
    elif args.indices_range is not None:
        indices = range(args.indices_range[0], args.indices_range[1])
    else:
        print('Please set up indices.')
        raise NotImplementedError

    ### run
    print('Start downloading..')
    if args.num_workers == 0:
        for i in tqdm(indices):
            worker(i)
    else:
        ## parallelism data
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)      