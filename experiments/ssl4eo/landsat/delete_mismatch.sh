#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

# User-specific parameters
ROOT_DIR=data
L5_L1="$ROOT_DIR/ssl4eo_l_tm_toa/imgs"
L7_L1="$ROOT_DIR/ssl4eo_l_etm_toa/imgs"
L7_L2="$ROOT_DIR/ssl4eo_l_etm_sr/imgs"
L8_L1="$ROOT_DIR/ssl4eo_l_oli_tirs_toa/imgs"
L8_L2="$ROOT_DIR/ssl4eo_l_oli_sr/imgs"

# Generic parameters
SCRIPT_DIR=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd)

time python3 "$SCRIPT_DIR/delete_mismatch.py" "$L7_L1" "$L7_L2" --delete-different-locations --delete-different-dates
time python3 "$SCRIPT_DIR/delete_mismatch.py" "$L8_L1" "$L8_L2" --delete-different-locations --delete-different-dates
