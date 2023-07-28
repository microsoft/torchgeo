#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

# User-specific parameters
ROOT_DIR=data
SRC_DIR="$ROOT_DIR/ssl4eo_l_oli_sr"
DST_DIR="$ROOT_DIR/ssl4eo_l_oli_sr_v2"
NUM_WORKERS=40

# Satellite-specific parameters
# https://www.usgs.gov/faqs/how-do-i-use-scale-factor-landsat-level-2-science-products
# https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
R_MIN=$(echo "(0   + 0.2) / 0.0000275" | bc -l)
R_MAX=$(echo "(0.3 + 0.2) / 0.0000275" | bc -l)

MIN=$R_MIN
MAX=$R_MAX

# Generic parameters
SCRIPT_DIR=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd)

time python3 "$SCRIPT_DIR/compress_dataset.py" \
    "$SRC_DIR" \
    "$DST_DIR" \
    --min ${MIN[@]} \
    --max ${MAX[@]} \
    --num-workers $NUM_WORKERS
