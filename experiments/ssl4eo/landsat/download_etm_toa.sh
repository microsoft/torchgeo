#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

# User-specific parameters
ROOT_DIR=data
SAVE_PATH="$ROOT_DIR/ssl4eo_l_etm_toa"
MATCH_FILE="$ROOT_DIR/ssl4eo_l_30/sampled_locations.csv"
NUM_WORKERS=40
START_INDEX=0
END_INDEX=10

# Satellite-specific parameters
COLLECTION=LANDSAT/LE07/C02/T1_TOA
QA_BAND=QA_PIXEL
QA_CLOUD_BIT=3
META_CLOUD_NAME=CLOUD_COVER
YEAR=2002  # SLC-on
BANDS=(B1 B2 B3 B4 B5 B6_VCID_1 B6_VCID_2 B7 B8)
ORIGINAL_RESOLUTIONS=(30 30 30 30 30 60 60 30 15)
NEW_RESOLUTIONS=30

# Generic parameters
SCRIPT_DIR=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd)
CLOUD_PCT=20
SIZE=264
DTYPE=float32
LOG_FREQ=1000

time python3 "$SCRIPT_DIR/download_ssl4eo.py" \
    --save-path "$SAVE_PATH" \
    --collection $COLLECTION \
    --qa-band $QA_BAND \
    --qa-cloud-bit $QA_CLOUD_BIT \
    --meta-cloud-name $META_CLOUD_NAME \
    --cloud-pct $CLOUD_PCT \
    --dates $YEAR-03-20 $YEAR-06-21 $YEAR-09-23 $YEAR-12-21 \
    --radius $(($NEW_RESOLUTIONS * $SIZE / 2)) \
    --bands ${BANDS[@]} \
    --original-resolutions ${ORIGINAL_RESOLUTIONS[@]} \
    --new-resolutions $NEW_RESOLUTIONS \
    --dtype $DTYPE \
    --num-workers $NUM_WORKERS \
    --log-freq $LOG_FREQ \
    --match-file "$MATCH_FILE" \
    --indices-range $START_INDEX $END_INDEX \
    --debug
