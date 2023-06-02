#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -euo pipefail

# User-specific parameters
ROOT_DIR=data
L5_L1="$ROOT_DIR/ssl4eo-l5-l1/imgs"
L7_L1="$ROOT_DIR/ssl4eo-l7-l1/imgs"
L7_L2="$ROOT_DIR/ssl4eo-l7-l2/imgs"
L8_L1="$ROOT_DIR/ssl4eo-l8-l1/imgs"
L8_L2="$ROOT_DIR/ssl4eo-l8-l2/imgs"

# Generic parameters
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

time python3 "$SCRIPT_DIR/delete_mismatch.py" "$L7_L1" "$L7_L2" --delete-different-locations --delete-different-dates
time python3 "$SCRIPT_DIR/delete_mismatch.py" "$L8_L1" "$L8_L2" --delete-different-locations --delete-different-dates
time python3 "$SCRIPT_DIR/delete_mismatch.py" "$L5_L1" "$L7_L1" "$L7_L2" "$L8_L1" "$L8_L2" --delete-different-locations
