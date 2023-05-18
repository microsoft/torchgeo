#!/usr/bin/env bash

#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --job-name=trainl7
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:A100:1

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

cd ~/torchgeo

python3 experiments/torchgeo/run_l7irish_experiments.py
