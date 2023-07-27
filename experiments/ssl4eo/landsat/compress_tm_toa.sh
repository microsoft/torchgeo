#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

# User-specific parameters
ROOT_DIR=data
SRC_DIR="$ROOT_DIR/ssl4eo_l_tm_toa"
DST_DIR="$ROOT_DIR/ssl4eo_l_tm_toa_v2"
NUM_WORKERS=40

# Satellite-specific parameters
# https://www.usgs.gov/faqs/how-do-i-use-scale-factor-landsat-level-2-science-products
# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA
R_MIN=0
R_MAX=0.4

# https://earthobservatory.nasa.gov/global-maps/MOD_LSTD_M
T_MIN=$(echo "273.15 - 25" | bc -l)
T_MAX=$(echo "273.15 + 45" | bc -l)

MIN=($R_MIN $R_MIN $R_MIN $R_MIN $R_MIN $T_MIN $R_MIN)
MAX=($R_MAX $R_MAX $R_MAX $R_MAX $R_MAX $T_MAX $R_MAX)

# Generic parameters
SCRIPT_DIR=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd)

time python3 "$SCRIPT_DIR/compress_dataset.py" \
    "$SRC_DIR" \
    "$DST_DIR" \
    --min ${MIN[@]} \
    --max ${MAX[@]} \
    --num-workers $NUM_WORKERS
