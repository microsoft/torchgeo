#!/usr/bin/env bash

set -euo pipefail

# User-specific parameters
SAVE_PATH=data/ssl4eo-l-conus
START_INDEX=0
END_INDEX=500

# Generic parameters
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
SIZE=1320

time python3 "$SCRIPT_DIR/sample_ssl4eo.py" \
    --save-path "$SAVE_PATH" \
    --size $(($RES * $SIZE / 2)) \
    --indices-range $START_INDEX $END_INDEX
