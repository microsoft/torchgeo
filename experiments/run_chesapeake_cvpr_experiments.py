#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0]
DRY_RUN = False  # if False then print out the commands to be run, if True then run
DATA_DIR = ""  # path to the ChesapeakeCVPR data directory

# Hyperparameter options
training_set_options = ["de"]
model_options = ["unet"]
backbone_options = ["resnet18", "resnet50"]
lr_options = [1e-2, 1e-3, 1e-4]
loss_options = ["ce", "jaccard"]
weight_init_options = ["null", "imagenet"]


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

    for (train_state, model, backbone, lr, loss, weight_init) in itertools.product(
        training_set_options,
        model_options,
        backbone_options,
        lr_options,
        loss_options,
        weight_init_options,
    ):

        experiment_name = f"{train_state}_{model}_{backbone}_{lr}_{loss}_{weight_init}"

        output_dir = os.path.join("output", "chesapeake-cvpr_experiments")
        log_dir = os.path.join(output_dir, "logs")
        config_file = os.path.join("conf", "chesapeake_cvpr.yaml")

        if not os.path.exists(os.path.join(output_dir, experiment_name)):

            command = (
                "python train.py"
                + f" config_file={config_file}"
                + f" experiment.name={experiment_name}"
                + f" experiment.module.model={model}"
                + f" experiment.module.backbone={backbone}"
                + f" experiment.module.weights={weight_init}"
                + f" experiment.module.learning_rate={lr}"
                + f" experiment.module.loss={loss}"
                + " experiment.module.class_set=7"
                + f" experiment.datamodule.train_splits=['{train_state}-train']"
                + f" experiment.datamodule.val_splits=['{train_state}-val']"
                + f" experiment.datamodule.test_splits=['{train_state}-test']"
                + f" program.output_dir={output_dir}"
                + f" program.log_dir={log_dir}"
                + f" program.data_dir={DATA_DIR}"
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
