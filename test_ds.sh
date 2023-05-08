#!/usr/bin/env bash

#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --job-name=testds
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:A100:1

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali


cd ~/torchgeo_trainer/torchgeo/
python3 test_ds.py
