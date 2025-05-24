#!/bin/bash

# DOCUMENTATION
# sample bash script demonstrating how to download 4 seasons of
# Sentinel-2 imagery (TOA, top-of-the-atmosphere reflectance)
# from Google Earth Engine, https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED
# through a set of coordinates provided as `test.csv` file in format:
# ```CSV
# latitude,longitude
# 49.6767,-70.34428366596
# ```

# SETUP
DOWNLOAD_DIRECTORY="./downloads"
INPUT_CSV_PATH="./test.csv"
OUTPUT_CSV_PATH="./output_metadata.csv"
COLLECTION_ID="COPERNICUS/S2_HARMONIZED"
START_DATE="2020-01-01"
END_DATE="2022-12-31"
CLOUD_COVER_META_NAME="CLOUDY_PIXEL_PERCENTAGE"
CLOUD_COVER_THRESHOLD=10
LAYERS="B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12"
SPATIAL_BUFFER=2000
TIME_BUFFER=30
REPROJECT_LAYER_NAME="B2"
NUM_WORKERS=16

# EXECUTION
python download_ssl4eo.py \
    $DOWNLOAD_DIRECTORY \
    $INPUT_CSV_PATH \
    $OUTPUT_CSV_PATH \
    $COLLECTION_ID \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --cloud_cover_meta_name "$CLOUD_COVER_META_NAME" \
    --cloud_cover_threshold $CLOUD_COVER_THRESHOLD \
    --layers $LAYERS \
    --spatial_buffer $SPATIAL_BUFFER \
    --time_buffer $TIME_BUFFER \
    --reproject_layer_name "$REPROJECT_LAYER_NAME" \
    --num_workers $NUM_WORKERS