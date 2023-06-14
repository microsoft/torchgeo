#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

# User-specific parameters
SAVE_PATH=data/ssl4eo_l_conus
START_INDEX=0
END_INDEX=10

# Generic parameters
SCRIPT_DIR=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd)
SIZE=264
RES=30

time python3 "$SCRIPT_DIR/sample_conus.py" \
    --save-path "$SAVE_PATH" \
    --size $(($RES * $SIZE / 2)) \
    --indices-range $START_INDEX $END_INDEX
