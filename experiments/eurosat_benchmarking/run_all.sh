#!/usr/bin/env bash
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
#
# Reproduce the EuroSAT SSL benchmark end to end.
#
# Run from this directory. Expects EuroSAT at ./data/eurosat (a symlink to a
# shared copy is fine) and a Python environment with torchgeo installed whose
# torch build matches the installed NVIDIA driver.
set -euo pipefail

PYTHON="${PYTHON:-python}"

# SimCLR and MoCo are scheduled before BYOL because BYOL's augmentation pipeline
# is roughly 7x slower per step, so it is the least likely to finish. Within each
# task, ResNet-50 is scheduled before ViT-S/16.
ORDER='configs/resnet50/simclr*.yaml,configs/resnet50/moco*.yaml'
ORDER+=',configs/vit_small/simclr*.yaml,configs/vit_small/moco*.yaml'
ORDER+=',configs/resnet50/byol*.yaml,configs/vit_small/byol*.yaml'

"$PYTHON" scripts/make_configs.py
"$PYTHON" scripts/run_sweep.py --configs "$ORDER" --gpus "${GPUS:-0,1,2,3,4,5,6,7}"
"$PYTHON" scripts/eval_knn.py --all
"$PYTHON" scripts/make_table.py
