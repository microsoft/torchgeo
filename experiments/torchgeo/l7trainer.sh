#!/usr/bin/env bash

#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --job-name=trainl7
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:A100:1
#SBATCH --mail-user=yichia3@illinois.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

. /projects/dali/spack/share/spack/setup-env.sh
spack env activate dali

python3 ~/torchgeo/experiments/torchgeo/run_l7irish_experiments.py
