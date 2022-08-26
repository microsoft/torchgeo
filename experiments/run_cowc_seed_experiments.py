#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""

import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = range(1)
DRY_RUN = True  # if True then print out the commands to be run, if False then run
DATA_DIR = ""  # path to the COWC data directory

# Hyperparameter options
model_options = ["resnet18", "resnet50"]
pretrained_options = [True]
lr_options = [1e-4]
seeds = range(10)


def do_work(work: "Queue[str]", gpu_idx: int) -> bool:
    """Process for each ID in GPUS."""
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))
    return True


if __name__ == "__main__":
    work: "Queue[str]" = Queue()

    for (model, lr, pretrained, seed) in itertools.product(
        model_options, lr_options, pretrained_options, seeds
    ):
        experiment_name = f"{model}_{lr}_{pretrained}_{seed}"

        output_dir = os.path.join("output", "cowc_seed_experiments")
        log_dir = os.path.join(output_dir, "logs")
        config_file = os.path.join("conf", "cowc_counting.yaml")

        if not os.path.exists(os.path.join(output_dir, experiment_name)):
            command = (
                "python train.py"
                + f" config_file={config_file}"
                + f" experiment.name={experiment_name}"
                + f" experiment.module.model={model}"
                + f" experiment.module.learning_rate={lr}"
                + f" experiment.module.pretrained={pretrained}"
                + f" program.output_dir={output_dir}"
                + f" program.log_dir={log_dir}"
                + f" program.data_dir={DATA_DIR}"
                + f" program.seed={seed}"
                + " trainer.gpus=[GPU]"
            )
            command = command.strip()

            work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
