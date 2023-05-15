#!/usr/bin/env bash

#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --job-name=cog
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:A100:1

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

cd ~/torchgeo_trainer/torchgeo/
python3 convert_cog.py
