#!/usr/bin/env bash

#SBATCH --time=3-00:00:00
#SBATCH --job-name=ag-50
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%x-%j.out

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

python3 experiments/ssl4eo/run_agrifieldnet.py
