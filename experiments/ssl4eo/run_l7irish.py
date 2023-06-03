#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0, 0, 1, 1, 2, 2, 3, 3]
DRY_RUN = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
model_options = ["unet", "fcn"]
backbone_options = ["resnet18"]
lr_options = [0.001, 0.0003, 0.0001, 0.00003]
loss_options = ["ce"]
wd_options = [0, 0.1, 0.01]
weight_options = [True, False]
seed_options = [0, 1, 2]


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

    for model, backbone, lr, loss, wd, weights, seed in itertools.product(
        model_options,
        backbone_options,
        lr_options,
        loss_options,
        wd_options,
        weight_options,
        seed_options,
    ):
        if model == "fcn" and not weights:
            continue

        if model != "unet":
            experiment_name = f"{model}_{backbone}_{lr}_{loss}_{wd}_{weights}_{seed}"
        else:
            experiment_name = f"{model}_{lr}_{loss}_{wd}_{weights}_{seed}"

        config_file = os.path.join("conf", "l7irishtile.yaml")

        command = (
            "python train.py"
            + f" config_file={config_file}"
            + f" module.model={model}"
            + f" module.backbone={backbone}"
            + f" module.learning_rate={lr}"
            + f" module.loss={loss}"
            + f" module.weight_decay={wd}"
            + f" module.weights={weights}"
            + f" program.seed={seed}"
            + f" program.experiment_name={experiment_name}"
            + " trainer.devices=[GPU]"
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
