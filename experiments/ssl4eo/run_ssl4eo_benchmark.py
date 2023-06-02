#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
DEVICE = [1, 2, 3, 4]
DRY_RUN = False  # if False then print out the commands to be run, if True then run
conf_file_name = "etm_sr_cdl.yaml"

# Hyperparameter options
model_options = ["unet"]
backbone_options = ["resnet18", "resnet50"]
lr_options = [0.003, 0.0001]
loss_options = ["ce"]
wd_options = [0, 0.001]
weight_options = [True, False]


def do_work(work: "Queue[str]", gpu_idx: int) -> bool:
    """Process for each ID in DEVICE."""
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not DRY_RUN:
            subprocess.call(experiment.split(" "))
    return True


if __name__ == "__main__":
    work: "Queue[str]" = Queue()

    for model, backbone, lr, loss, wd, weights in itertools.product(
        model_options,
        backbone_options,
        lr_options,
        loss_options,
        wd_options,
        weight_options,
    ):
        if model == "fcn" and not weights:
            continue

        experiment_name = (
            f"{conf_file_name.split('.')[0]}_{model}_{backbone}_"
            f"{lr}_{loss}_{wd}_{weights}"
        )

        config_file = os.path.join("conf", conf_file_name)

        command = (
            "python train.py"
            + f" config_file={config_file}"
            + f" module.model={model}"
            + f" module.backbone={backbone}"
            + f" module.learning_rate={lr}"
            + f" module.loss={loss}"
            + f" module.weight_decay={wd}"
            + f" module.weights={weights}"
            + f" program.experiment_name={experiment_name}"
            + " trainer.devices=[GPU]"
        )
        command = command.strip()

        work.put(command)

    processes = []
    for gpu_idx in DEVICE:
        p = Process(target=do_work, args=(work, gpu_idx))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
