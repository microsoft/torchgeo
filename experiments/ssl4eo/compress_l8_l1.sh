#!/usr/bin/env bash

set -euo pipefail

# User-specific parameters
ROOT_DIR=data
SRC_DIR="$ROOT_DIR/ssl4eo-l8-l1"
DST_DIR="$ROOT_DIR/ssl4eo-l8-l1-v2"
NUM_WORKERS=40

# Satellite-specific parameters
# https://www.usgs.gov/faqs/how-do-i-use-scale-factor-landsat-level-2-science-products
# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA
R_MIN=0
R_MAX=0.4

# https://earthobservatory.nasa.gov/global-maps/MOD_LSTD_M
T_MIN=$(echo "273.15 - 25" | bc -l)
T_MAX=$(echo "273.15 + 45" | bc -l)

MIN=($R_MIN $R_MIN $R_MIN $R_MIN $R_MIN $R_MIN $R_MIN $R_MIN $R_MIN $T_MIN $T_MIN)
MAX=($R_MAX $R_MAX $R_MAX $R_MAX $R_MAX $R_MAX $R_MAX $R_MAX $R_MAX $T_MAX $T_MAX)

# Generic parameters
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

time python3 "$SCRIPT_DIR/compress_dataset.py" \
    "$SRC_DIR" \
    "$DST_DIR" \
    --min ${MIN[@]} \
    --max ${MAX[@]} \
    --num-workers $NUM_WORKERS
