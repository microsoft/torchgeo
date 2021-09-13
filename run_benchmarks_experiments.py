"""Script for running the benchmark script over a sweep of different options."""
import itertools
import subprocess
import time
from typing import List

NUM_BATCHES = 100

SEED_OPTIONS = [0, 1, 2]
CACHE_OPTIONS = [True, False]
BATCH_SIZE_OPTIONS = [16, 32, 64, 128, 256, 512]

total_num_experiments = len(SEED_OPTIONS) * len(CACHE_OPTIONS) * len(BATCH_SIZE_OPTIONS)

if __name__ == "__main__":

    tic = time.time()
    for i, (cache, batch_size, seed) in enumerate(
        itertools.product(CACHE_OPTIONS, BATCH_SIZE_OPTIONS, SEED_OPTIONS)
    ):
        print(f"{i}/{total_num_experiments} -- {time.time() - tic}")
        tic = time.time()
        command: List[str] = [
            "python",
            "benchmark.py",
            "--landsat-root",
            "/datadrive/landsat",
            "--cdl-root",
            "/datadrive/cdl",
            "-w",
            "6",
            "--batch-size",
            str(batch_size),
            "--num-batches",
            str(NUM_BATCHES),
            "--seed",
            str(seed),
            "--verbose",
        ]

        if cache:
            command.append("--cache")

        subprocess.call(command)
