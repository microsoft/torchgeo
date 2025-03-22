#!/usr/bin/env python3
"""
Download and spatio-temporally align Earth observation data from Google Earth Engine.
"""
__version__ = '0.0.1'
__date__ = 'Mar 22, 2025'
__authors__ = ['Nassim Ait Ali Braham', 'Conrad M Albrecht',]
__contact__ = 'https://github.com/cmalbrec'
__copyright__ = '(c) 2025'
__license__ = 'to be determined'
__status__ = 'experimental'

# Standard library imports
import argparse
import datetime
import logging
import os
import sys
from collections.abc import Callable
from multiprocessing import Pool
from typing import Annotated, Any

# Third-party imports
import ee
import geopandas
import numpy
import pandas
import rasterio
import rasterio.enums
import rasterio.warp
import requests  # type: ignore
import shapely.geometry

# define units
Percent = Annotated[float, "%"]
Degree = Annotated[float, "°"]
Meters = Annotated[float, "m"]
Days = Annotated[float, "d"]
Seconds = Annotated[int, "s"]
Milliseconds = Annotated[int, "ms"]

# set up logging
logger = logging.getLogger(__name__)

# parameters
SENTINEL2_CLOUDCOVER_META_NAME = 'CLOUDY_PIXEL_PERCENTAGE'
SSL4EO_GEE_DATA = {
    'sentinel_2_l2a': {
        'GEE_name': 'COPERNICUS/S2',
        'temporal_resolution_in_days': 5.0,
        'layers': [
            {'name': 'B1', 'description': 'coastal aerosols (60m resolution)',},
            {'name': 'B2', 'description': 'blue band (10m resolution)',},
            {'name': 'B3', 'description': 'green band (10m resolution)',},
            {'name': 'B4', 'description': 'red band (10m resolution)',},
            {'name': 'B5', 'description': 'vegetation red edge +-700nm (20m resolution)',},
            {'name': 'B6', 'description': 'vegetation red edge +-740nm (20m resolution)',},
            {'name': 'B7', 'description': 'vegetation red edge +-780nm (20m resolution)',},
            {'name': 'B8', 'description': 'near infrared (10m resolution)',},
            {'name': 'B8A', 'description': 'near infrared narrow bandwidth (20m resolution)',},
            {'name': 'B9', 'description': 'water vapor (60m resolution)',},
            {'name': 'B10', 'description': 'short wave infrared +-1375nm narrow bandwidth (60m resolution)',},
            {'name': 'B11', 'description': 'short wave infrared +-1610nm  (20m resolution)',},
            {'name': 'B12', 'description': 'short wave infrared +-2190nm  (20m resolution)',},
        ],
    },
    'sentinel_1_grd': {
        'GEE_name': 'COPERNICUS/S1_GRD',
        'temporal_resolution_in_days': 5.0,
        'layers': [
            {'name': 'VV', 'description': 'vertical-vertical polarization channel',},
            {'name': 'VH', 'description': 'vertical-horizontal polarization channel',},
            {'name': 'angle', 'description': 'SAR illumination angle',},
        ],
    },
}


def get_cloudfree_sentinel2_timestamps(
    geeCollection:ee.imagecollection.ImageCollection,
    maxCloudCover:Percent = 10.0,
) -> list[Milliseconds]:
    """
    Query GEE to obtain timestamps of Sentinel-2 data with a cloud coverage below a given threshold.

    Args:
        eeCollection:  Sentinel-2 GEE collection to filter wrt. cloud coverage metadata
        maxCloudCover: maximum cloud cover in percent to filter

    Returns:
       list of Unix epoch times where Sentinel-2 data for collection is available with given maximum cloud cover
    """
    logger.debug('get_cloudfree_sentinel2_timestamps: Reaching out to GEE for time series computation.')
    toReturn: list[Milliseconds] = geeCollection.filter(
        ee.Filter.lt(SENTINEL2_CLOUDCOVER_META_NAME, maxCloudCover),
    ).aggregate_array("system:time_start").getInfo()
    return toReturn

def get_utm_bounding_box(
    latitude:Degree, 
    longitude:Degree, 
    radius:Meters, 
) -> tuple[tuple[float,float,float,float],str]:
    """
    Function to generate UTM bounding box from a center coordinate in EPSG:4326.

    Args:
        latitude:   latitude of point of interest
        longitude:  longitude of point of interest
        radius:     radius of area of interest

    Returns:
        square: bounding box with (latitude, longitude) as center and edge length 2*radius,
        crsUTM: EPGS string of UTM coordinate system of square bounding box
    """
    # Get UTM zone
    utm_zone = int((longitude + 180) / 6.) + 1
    hemisphere = '6' if latitude > 0 else '7'
    utmCRS = f'EPSG:32{hemisphere}{utm_zone:02d}'
    # project coordinate into UTM zone
    x, y = (
        geopandas.GeoSeries(
            [shapely.geometry.Point(longitude,latitude)],
            crs = {'init': 'epsg:4326'},
        )
        .to_crs({'init': utmCRS})[0]
        .coords.xy,
    )
    x, y = x[0], y[0]

    return (x - radius, y - radius, x + radius, y + radius), utmCRS


def construct_layer_file_name_base(
    latitude: Degree,
    longitude: Degree,
    timestamp: Seconds,
    product: str,
    layer: str | list[str],
    directoriesUNIX: bool = False,
    timestampAsDate: bool = True,
) -> str:
    """
    Define SSL4EO file name schema.

    Args:
        latitude:       latitude of point of interest for spatial indexing
        longitude:      longitude of point of interest for spatial indexing
        timestamp:      timestamp in UNIX epoch time (UTC timezone)
        product:        product descriptor, e.g., `sentinel_2_l2a`
        layer:          raster descriptor, e.g., `B2`
        directoriesUNIX determines if the data gets saved in UNIX subdirectories

    Returns:
        fileName: of GeoTiff to store without extension with latitude and longitude truncated to meter-precision for indexing
    """
    # settings
    separator = '/' if directoriesUNIX else '_'

    # Remove '/' from product name
    product = product.replace('/', '_')

    # input data check
    assert -180 <= longitude <= 180
    assert -90 <= latitude  <= 90
    assert type(timestamp) is int

    # timestamp formatting
    datetimeTimestamp = datetime.datetime.fromtimestamp(timestamp)
    timestring = (
        f'{datetimeTimestamp.year:04}{datetimeTimestamp.month:02}{datetimeTimestamp.day:02}'
        if timestampAsDate else str(timestamp)
    )

    return f"""\
lat{int(latitude*1e5):08}lon{int(longitude*1e5):08}{separator}\
time{timestring}{separator}\
{product}{separator if isinstance(layer,str) else ''}\
{layer if isinstance(layer,str) else ''}"""


def random_select_four_seasons_from_timeseries(
    timestamps:list[Milliseconds],
) -> list[int]:
    """
    Pick a random year in which random timestamps are picked for the 4 seasons of a year.

    Args:
        timestamps: series of timestamps to pick from

    Returns:
        indices: 4 indices of the timestamp list picked as seasons
    """
    # define seasons
    wintermonths = [12,1,2,]
    springmonths = [3,4,5,]
    summermonths = [6,7,8,]
    fallmonths = [9,10,11,]

    # format raw UNIX epoch timestamps into proper datetime timestamp series
    s = pandas.to_datetime(pandas.Series(timestamps), unit='ms')

    # pick a random year from time series
    indices = []
    for year in numpy.random.permutation(s.dt.year.unique()):
        sry = s[s.dt.year == year]
        try:
            indices = [
                numpy.random.choice(sry[sry.dt.month.isin(seasonmonths)].index)
                for seasonmonths in [
                    wintermonths,
                    springmonths,
                    summermonths,
                    fallmonths
                ]
            ]
            assert len(indices) == 4
            break
        except Exception as e:
            logger.debug(e)

    return indices


def download_layer_geotiff(
    geeImage: ee.image.Image,
    boundBox: ee.geometry.Geometry,
    layerName: str,
    outputPath: str,
) -> None:
    """
    Download a GeoTiff from Google Earth Engine image.

    Args:
        geeImage:   GEE image object to locally save on disk.
        layerName:  name of GEE band to select
        boundBox:   area of interest to safe
        outputPath: GeoTiff path to locally store raster layer.

    Raises:
        Exception: on failed download
    """
    logger.info(f'Downloading image to {outputPath}.')

    # compute download URL
    try:
        downloadURL = geeImage.getDownloadURL(
            {
                'format': 'GEO_TIFF',
                'crs': geeImage.projection().crs(),
                'region': boundBox,
                'scale': geeImage.projection().nominalScale(),
                #'crsTransform': geeImage.projection().transform(),
                # notes:
                # - attention: the commented line above seems to upscale the hightest native resolution, probably a GEE issue
                # - on potentially half-pixel inconsistencies with GEE and scale parameter:
                # https://developers.google.com/earth-engine/guides/exporting_images#setting_scale
            }
        )
        logger.debug(f'Downloading data from URL: {downloadURL}.')
    except Exception as e:
        logger.error(e)

    # download data
    r = requests.get(downloadURL)
    if r.status_code == 200:
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        with open(outputPath, 'wb') as fp:
            fp.write(r.content)
        r.close()
    else:
        r.close()
        raise Exception(f'Failed downloading to {outputPath}: {r.status_code}')


def download_data_from_gee(
    downloadDir:str,
    centerCoords:pandas.DataFrame,
    layerNames:list[str] | str,
    geeCollection:ee.imagecollection.ImageCollection,
    collectionName:str,
    spatialBuffer:Meters,
    temporalBuffer:Days,
    temporal_sampling_strategy:Callable[[list[Milliseconds]], list[int]] \
        = random_select_four_seasons_from_timeseries,
    saveInSubdirs:bool = True,
    reprojectLayerName: str | None = None,
) -> pandas.DataFrame:
    """
    Browse spatio(-temporal) anchors to download Google Earth Engine data.

    Args:
        downloadDir:                directory to save downloaded data to (assumed to exist)
        centerCoords:               spatio(-temporal) coordinates as anchors for downloading, column names:
                                    latitude,longitude,timestamp_anchor,timestamp_data
                                    notes:
                                    - not providing temporal information triggers seasonal random sampling
                                    - temporal information is assumed as UNIX epoch timestamps in seconds
        layerNames:                 names of layers in GEE collection to download
        geeCollection:              GEE collection,
        collectionName:             user name of GEE collection
        spatialBuffer:              spatial extent to download around spatial anchor (AoI)
        temporalBuffer:             temporal buffer to search for data around temporal anchor
        temporal_sampling_strategy: sampling strategy of timestamps available in GEE collection for AoI
        saveInSubdirs:              save data as plain GeoTiff files or into a directory hierarchy
        reprojectLayerName:         download all the GEE layers in a single file upsampled to the resolution of the named layer

    Returns:
        dataframe with spatio-temporal information on data downloaded from GEE collection:
        - latitude, longitude as spatial anchors
        - timestamp_anchor as temporal anchor in UNIX epoch time in milliseconds
        - timestamp_data as (starting) time of data acquisition by satellite in UNIX epoch (milliseconds)
        note: an entry is available if, and only if, all specified layers have been downloaded
    """
    # check that downloading directory exists
    assert os.path.exists(downloadDir)
    layersAtOnce = reprojectLayerName is not None

    # initialize spatio-temporal information on downloaded data
    spatioTemporalIndicesDownloaded = pandas.DataFrame(
        [],
        columns=['latitude', 'longitude', 'timestamp_anchor', 'timestamp_data'],
    )

    # visit geospatial locations as per user request
    for idx, (lat, lon) in centerCoords[['latitude', 'longitude']].iterrows():
        # define GEE spatial area-of-interest
        bboxCoords, crs = get_utm_bounding_box(
            latitude = lat, longitude = lon, radius = spatialBuffer,
        )

        # filter GEE collection by spatial area
        boundBox = ee.Geometry.Rectangle(
            bboxCoords, proj = crs, geodesic = True, evenOdd = False
        )
        geeCollectionBounded = geeCollection.filterBounds(boundBox)

        # filter timestamps according to user needs
        if 'timestamp_anchor' in centerCoords:
            logger.debug('Closest Timestamp Sampling')
            geeCollectionBounded = geeCollectionBounded.filterDate(
                centerCoords.timestamp_anchor.loc[idx]*1e3 - int(24*3600*temporalBuffer*1e3),
                centerCoords.timestamp_anchor.loc[idx]*1e3 + int(24*3600*temporalBuffer*1e3),
            ).filter(
                ee.Filter.contains('.geo', boundBox)
            )
            timestampsMilliseconds = geeCollectionBounded.aggregate_array("system:time_start").getInfo()
            logger.debug('Reaching out to GEE for time series computation.')
            timestampIndices = (
                [
                    numpy.abs(
                        numpy.array(timestampsMilliseconds)/1000.
                        - centerCoords.timestamp_anchor.loc[idx]
                    ).argmin()
                ]
                if len(timestampsMilliseconds) > 0
                else []
            )
        else:
            logger.debug('Seasonal Sampling')
            timestampsMilliseconds = geeCollectionBounded.filter(
                ee.Filter.contains(
                    '.geo',
                    boundBox
                ) # ATTENTION: potential inefficiency in performance over requirement of tile overlap with area-of-interest
            )
            .aggregate_array("system:time_start")
            .getInfo()
            logger.debug('Reaching out to GEE for time series computation.')
            timestampIndices = temporal_sampling_strategy(timestampsMilliseconds)

        # download the data as GeoTIFFs (if any)
        if len(timestampIndices) == 0:
            logger.warning(f'No timestamps available in GEE collection at location (lat,lon)=({lat},{lon}).')
        for timestampIndex in timestampIndices:
            logger.debug(f'timestamp index {timestampIndex}')
            if layersAtOnce and type(layerNames) is str:
                 layerNamesIter = [layerNames]
            else:
                 layerNamesIter = layerNames
            for layer in layerNamesIter:
                logger.debug(f'layer {layer}')
                try:
                    # convert GEE collection into GEE image selecting relevant layers
                    geeImage = ee.Image(
                        geeCollectionBounded.select(layer).toList(
                            geeCollectionBounded.size()
                        ).get(int(timestampIndex))
                    )
                    # upscale GEE image in case it contains multiple layers
                    if layersAtOnce:
                        reprojectCRS = geeImage.select(reprojectLayerName).projection()
                        geeImage = geeImage.resample().reproject(
                                reprojectCRS,
                                scale = reprojectCRS.nominalScale(),
                        )
                    # download GEE image
                    download_layer_geotiff(
                        geeImage = geeImage,
                        layerName = layer,
                        boundBox = boundBox,
                        outputPath = os.path.join(
                            downloadDir,
                            construct_layer_file_name_base(
                                lat,
                                lon,
                                timestampsMilliseconds[int(timestampIndex)] // 1000,
                                collectionName,
                                layer,
                                directoriesUNIX = saveInSubdirs,
                           ) + f'_{int(2*spatialBuffer)}m.tif',
                        )
                    )
                    write = True
                except Exception as e:
                    write = False
                    logger.error(
                        f"Unable to download {layer} of {collectionName} at (lat,lon,time)=({lat},{lon},{timestampsMilliseconds[timestampIndex]//1000}): {e}",
                        exc_info=True,
                    )

                    break
            if write:
                # record spatio-temporal information on downloaded data
                timestamp = timestampsMilliseconds[int(timestampIndex)]/1e3
                spatioTemporalIndicesDownloaded.loc[
                    len(spatioTemporalIndicesDownloaded)
                ] = [
                    lat,
                    lon,
                    timestamp if 'timestamp_anchor' not in centerCoords else centerCoords.timestamp_anchor.loc[idx],
                    timestamp,
                ]

    return spatioTemporalIndicesDownloaded



def spatial_align_rasters(
    referenceRasterPath:str,
    rasterPathes:list[str],
    scalefactor:float = 1,
    resamplingMethod:rasterio.enums.Resampling = rasterio.enums.Resampling.nearest,
) -> None:
    """
    Geospatially harmonize rasters to a common reference grid.

    Args:
        referenceRasterPath:    path to geospatial raster data as reference to reproject into
        rasterPathes:           pathes to geospatial rasters to reproject
        scalefactor:            factor of rescaling the reference raster before reprojection
        resamplingMethod:       resampling method for reprojection
    """
    # load reference data and rescale as required
    referenceRaster = rasterio.open(referenceRasterPath)
    if scalefactor != 1:
        logger.debug("Rescaling {referenceRasterPath} by factor {scalefactor}.")
        data = referenceRaster.read(
            out_shape = (
                referenceRaster.count,
                int(referenceRaster.height * scalefactor),
                int(referenceRaster.width  * scalefactor),
            ),
            resampling = resamplingMethod,
        )
        # write upscaled reference raster
        kwargs = referenceRaster.meta.copy()
        kwargs.update(
            transform = referenceRaster.transform * referenceRaster.transform.scale(
                (referenceRaster.width  / data.shape[-1]),
                (referenceRaster.height / data.shape[-2]),
            ),
            width = referenceRaster.width * scalefactor,
            height = referenceRaster.width * scalefactor,
        )
        referenceRasterRescaledPath = f'{os.path.splitext(referenceRasterPath)[0]}-scaled{scalefactor}.tif'
        with rasterio.open(referenceRasterRescaledPath, 'w', **kwargs,) as dst:
            dst.write(data)
        referenceRaster.close()
        referenceRaster = rasterio.open(referenceRasterRescaledPath)

    # reproject all rasters to the reference raster
    for rasterPath in rasterPathes:
        logger.debug("Reprojecting {rasterPath} to {referenceRasterPath}.")
        with rasterio.open(rasterPath) as raster:
            # reproject raster data
            data, transform = rasterio.warp.reproject(
                source = raster.read(),
                src_transform = raster.transform,
                src_crs = raster.crs,
                src_nodata = raster.nodata,
                destination = numpy.empty(
                    (
                        raster.count,
                        referenceRaster.width,
                        referenceRaster.height,
                    ),
                    dtype = raster.dtype[0],
                ),
                dst_transform = referenceRaster.transform,
                dst_crs = referenceRaster.crs,
                dst_nodata = raster.nodata,
                resampling = resamplingMethod,
            )
            # adjust metadata
            kwargs = referenceRaster.meta.copy()
            kwargs.update(
                count = raster.count,
                nodata = raster.nodata,
                transform = transform,
                dtype = raster.dtype[0],
            )
            # write reprojected data to GeoTIFF
            with rasterio.open(
                f'{os.path.splitext(rasterPath)[0]}-TO-{os.path.splitext(os.path.basename(referenceRasterPath))[0]}.tif',
                'w', **kwargs,
            ) as out:
                out.write(data)
    # garbage collection
    referenceRaster.close()



def crop_geotiff2ssl4eo_datacube(
    input_path:str,
    output_path:str,
    cubesize:int     = 264,
) -> None:
    """
    Take a georeferenced image and crop it to SSL4EO-S12 data cube size.

    Args:
        input_path:  path to reference image to crop
        output_path: path to write cropped image to
        cubesize:    SSL4EO-S12 square datacube of size `cubesize x cubesize` pixels
    """
    with rasterio.open(input_path) as src:
        # determine crop window
        window = rasterio.windows.Window(
            (src.width  - cubesize) // 2,
            (src.height - cubesize) // 2,
            cubesize, cubesize,
        )

        # determine metadata
        meta = src.meta
        meta.update(
            {
                'height': cubesize,
                'width': cubesize,
                'transform': src.window_transform(window),
            }
        )

        # write cropped GeoTIFF
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(src.read(window=window))


def stack_ssl4eo_geotiffs(
    ssl4eo_pathes:list[str],
    output_path:str,
) -> None:
    """
    Take SSL4EO georeferenced, multiband data cubes and stack them.

    Args:
        ssl4eo_pathes:  list of pathes with georeferenced files of same location and raster resolution for stacking into a SSL4EO datacube
        output_path:    path to write GeoTIFF to containing a single SSL4EO data cube 
    """
    # generate metadata
    with rasterio.open(ssl4eo_pathes[0]) as src:
        meta = src.meta
    meta.update(
        count = sum(rasterio.open(path).count for path in ssl4eo_pathes)
    )
    
    # create empty SSL4EO datacube
    rasters = numpy.empty(
        (meta['count'], meta['height'], meta['width']),
        dtype = meta['dtype'],
    )

    # populate SSL4EO datacube
    b_off = 0
    with rasterio.open(output_path, 'w', **meta) as dst:
        for path in ssl4eo_pathes:
            with rasterio.open(path) as src:
                rasters[b_off:b_off+src.count, ...] = src.read()
                b_off += src.count
        dst.write(rasters)


def save_results(results: Any, output_csv_path: str) -> None:
    """
    Save results to a CSV file, overwriting the existing file each time.
    """
    df = pandas.concat(results, ignore_index=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f'Data saved to {output_csv_path}')


def download_for_coords(args: Any) -> pandas.DataFrame:
    """
    Helper function to download data for a single set of coordinates.
    
    Args:
        args (tuple): Contains the arguments to pass to the download function.
    """
    try:
        download_directory, row, layers, image_collection, collection_id, \
        spatial_buffer, time_buffer, reproject_layer_name, creds = args
        ee.Initialize(creds)
        result_df = download_data_from_gee(
            download_directory, pandas.DataFrame([row]), layers, image_collection, collection_id,
            spatial_buffer, time_buffer, saveInSubdirs=True, reprojectLayerName=reproject_layer_name
        )
        return result_df
    except Exception as e:
        logger.error(
            f"Error processing {row['latitude']}, {row['longitude']}: {e}",
            exc_info = True,
        )
        # Return an empty DataFrame indicating failure
        return pandas.DataFrame()


def main(
    download_directory: str,
    input_csv_path: str,
    output_csv_path: str,
    collection_id: str,
    checkpoint_csv_path: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    cloud_cover_meta_name: str | None = None,
    cloud_cover_threshold: float | None = None,
    layers: list[str] | None = None,
    spatial_buffer: int = 1000,
    time_buffer: float | None = None,
    reproject_layer_name: str | None = None,
    num_workers: int = 4,
    gcloud_service_creds: tuple[str, str] | None = None,
) -> None:
    """
    Main function to orchestrate the downloading of satellite images based on specified parameters.

    Args:
        download_directory (str):               Directory to save the downloaded images.
        input_csv_path (str):                   Path to the CSV file with the center coordinates.
        output_csv_path (str):                  Path to save the output CSV with download metadata: coordinates + timestamps.
        collection_id (str):                    GEE collection ID.
        checkpoint_csv_path (str, optional):    Path to the checkpoint CSV file to resume download.
        start_date (str, optional):             Start date for the data collection (YYYY-MM-DD format).
        end_date (str, optional):               End date for the data collection (YYYY-MM-DD format).
        cloud_cover_meta_name (str, optional):  Metadata field name for cloud coverage.
        cloud_cover_threshold (float, optional):Maximum allowed cloud coverage percentage.
        layers (list of str, optional):         List of layer (bands) names to download.
        spatial_buffer (int):                   Spatial buffer in meters around the center coordinates.
        time_buffer (int, optional):            Temporal buffer in days for closest timestamp matching.
        reproject_layer_name (str, optional):   If provided, all layers will be reprojected to this layer's resolution.
        num_workers (int):                      Number of parallel processes to use.
        gcloud_service_creds (str, str):        Google cloud credentials: ('service@project.iam.gserviceaccount.com', '/path/to/key').
    """
    # Initialize the Google Earth Engine API.
    # note: Authentication needs to happen through a service account in non-interactive mode, cf.
    # https://developers.google.com/earth-engine/guides/auth#credentials_for_service_accounts_and_compute_engine
    if gcloud_service_creds is None:
        ee.Authenticate()
        ee.Initialize()
    else:
        creds = ee.ServiceAccountCredentials(*gcloud_service_creds)
        ee.Initialize(creds)

    # Create download directory if it does not exist.
    os.makedirs(download_directory, exist_ok=True)

    # Define the Image Collection from GEE.
    image_collection = ee.ImageCollection(collection_id)

    # Apply date filtering if specified.
    if start_date and end_date:
        image_collection = image_collection.filterDate(start_date, end_date)

    # Apply cloud cover filtering if specified.
    if cloud_cover_meta_name and cloud_cover_threshold is not None:
        image_collection = image_collection.filter(
            ee.Filter.lt(cloud_cover_meta_name, cloud_cover_threshold)
        )

    # Read the CSV with center coordinates.
    center_coords = pandas.read_csv(input_csv_path)

    results = []
    # Resume from checkpoint if provided
    if checkpoint_csv_path and os.path.exists(checkpoint_csv_path):
        processed_coords = pandas.read_csv(checkpoint_csv_path)
        # Initialize the content of results with the processed entries
        results = [processed_coords]
        # Merge and filter out processed entries
        merge_columns = ['latitude', 'longitude', 'timestamp_anchor']
        combined = pandas.merge(
            center_coords,
            processed_coords,
            on = merge_columns,
            how = 'left',
            indicator = True,
        )
        remaining_coords = combined[combined['_merge'] == 'left_only'].drop(
            columns='_merge',
        )
    else:
        remaining_coords = center_coords

    # Download the data with multiprocessing.
    counter = 0  
    with Pool(num_workers) as pool:
        task_generator = (
            (
                download_directory,
                row,
                layers,
                image_collection,
                collection_id,
                spatial_buffer,
                time_buffer,
                reproject_layer_name,
                creds,
            )
            for index, row in remaining_coords.iterrows()
        )
        for result in pool.imap(download_for_coords, task_generator):
            results.append(result)
            counter += 1
            if counter % 1000 == 0:
                save_results(results, output_csv_path)

    if results:
        save_results(results, output_csv_path)

    logger.info(f"Data has been successfully downloaded and saved to {output_csv_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download satellite images from Google Earth Engine.")
    parser.add_argument(
        "download_directory",
        type = str,
        help = "Directory to save the downloaded images.",
    )
    parser.add_argument(
        "input_csv_path",
        type = str,
        help = "CSV file path for input coordinates.",
    )
    parser.add_argument(
        "output_csv_path",
        type = str,
        help = "CSV file path to save the download metadata.",
    )
    parser.add_argument(
        "collection_id",
        type = str,
        help = "Google Earth Engine collection ID.",
    )
    parser.add_argument(
        "--checkpoint_csv_path",
        type = str,
        default = None,
        help = "Optional: CSV file path for checkpoint to resume download.",
    )
    parser.add_argument(
        "--start_date",
        type = str,
        help = "Start date (YYYY-MM-DD) for the data collection.",
        default = None,
    )
    parser.add_argument(
        "--end_date",
        type = str,
        help = "End date (YYYY-MM-DD) for the data collection.",
        default = None,
    )
    parser.add_argument(
        "--cloud_cover_meta_name",
        type = str,
        help = "Metadata field name for cloud cover.",
        default = None,
    )
    parser.add_argument(
        "--cloud_cover_threshold",
        type = float,
        help = "Maximum cloud cover percentage.",
        default = None,
    )
    parser.add_argument(
        "--layers",
        nargs = '+',
        help = "List of layer names to download.",
        default = None,
    )
    parser.add_argument(
        "--spatial_buffer",
        type = int,
        help = "Spatial buffer in meters.",
        default = 1000,
    )
    parser.add_argument(
        "--time_buffer",
        type = float,
        help = "Temporal buffer in days for closest timestamp matching.",
        default = None,
    )
    parser.add_argument(
        "--reproject_layer_name",
        type = str,
        help = "Reproject all layers to this layer's resolution.",
        default = None,
    )
    parser.add_argument(
        "--num_workers",
        type = int,
        help = "Number of parallel processes to use.",
        default = 4,
    )
    parser.add_argument(
        "--gcloud_service_email",
        type = str,
        help = "Google Cloud Service email. The corresponding credentials are picked from the environment variable `$GOOGLE_APPLICATION_CREDENTIALS`",
        default = None,
    )

    args = parser.parse_args()

    # assemble Google Cloud access info
    gcloud_service_credential_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    logger.debug(f"Google Cloud credential file set to: '{gcloud_service_credential_path}'")
    if gcloud_service_credential_path is not None and args.gcloud_service_email is not None:
        gcloud_service_creds = (
            args.gcloud_service_email,
            gcloud_service_credential_path,
        )

    main(
        args.download_directory,
        args.input_csv_path,
        args.output_csv_path,
        args.collection_id,
        args.checkpoint_csv_path,
        args.start_date,
        args.end_date,
        args.cloud_cover_meta_name,
        args.cloud_cover_threshold,
        args.layers,
        args.spatial_buffer,
        args.time_buffer,
        args.reproject_layer_name,
        args.num_workers,
        gcloud_service_creds,
    )

    sys.exit(os.EX_OK)
