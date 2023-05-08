#!/usr/bin/env bash

#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --job-name=trainl7
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

cd ~/torchgeo_trainer/torchgeo

python3 experiments/run_l7irish_experiments.py
