#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# from torchgeo.models import ResNet18_Weights
# from torchgeo.models import ResNet50_Weights

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0]
DRY_RUN = False  # if False then print out the commands to be run, if True then run
conf_file_name = "agrifieldnet.yaml"

# Hyperparameter options
model_options = ["unet"]
backbone_options = ["resnet18"]
lr_options = [0.001, 0.0003, 0.0001, 0.00003]
loss_options = ["ce"]
weight_options = [False, True]
seed_options = [0, 1]
weight_decay_options = [0, 1e-3]


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

    for model, backbone, lr, loss, weights, weight_decay, seed in itertools.product(
        model_options,
        backbone_options,
        lr_options,
        loss_options,
        weight_options,
        weight_decay_options,
        seed_options,
    ):
        if model == "fcn" and not weights:
            continue

        experiment_name = (
            f"{conf_file_name.split('.')[0]}_"
            f"{model}_"
            f"{backbone}_"
            f"{lr}_"
            f"{loss}_"
            f"{weights}_"
            f"{weight_decay}_"
            f"{seed}"
        )

        config_file = os.path.join("conf", conf_file_name)

        command = (
            "python train.py"
            + f" config_file={config_file}"
            + f" module.model={model}"
            + f" module.backbone={backbone}"
            + f" module.learning_rate={lr}"
            + f" module.loss={loss}"
            + f" module.weights={weights}"
            + f" program.experiment_name={experiment_name}"
            + f" program.seed={seed}"
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