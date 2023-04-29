#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
<<<<<<< HEAD
<<<<<<< HEAD
GPUS = [0, 1]
=======
GPUS = [0]
>>>>>>> 7fd28499ac1019ca7f63fd3b4739a9c2fae5db88
=======
GPUS = [0]
>>>>>>> 7fd28499ac1019ca7f63fd3b4739a9c2fae5db88
DRY_RUN = False  # if False then print out the commands to be run, if True then run
DATA_DIR = "/projects/dali/data/l7irish"  # path to the L7Irish data directory

# Hyperparameter options
model_options = ["unet"]
lr_options = [1e-2]
loss_options = ["ce"]
backbone_options = ["resnet18"]
weight_init_options = ["null"]


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

    for model, backbone, lr, loss, weight_init in itertools.product(
        model_options, backbone_options, lr_options, loss_options, weight_init_options
    ):
        experiment_name = f"{model}_{backbone}_{lr}_{loss}_{weight_init}"

        output_dir = os.path.join("output", "l7irish_experiments")
        log_dir = os.path.join(output_dir, "logs")
        config_file = os.path.join("conf", "l7irish.yaml")

        if not os.path.exists(os.path.join(output_dir, experiment_name)):
            command = (
                "python train.py"
                + f" config_file={config_file}"
                + f" experiment.name={experiment_name}"
                + f" experiment.module.segmentation_model={model}"
                + f" experiment.module.learning_rate={lr}"
                + f" experiment.module.loss={loss}"
                + f" experiment.module.backbone={backbone}"
                + f" experiment.module.weights={weight_init}"
                + f" program.output_dir={output_dir}"
                + f" program.log_dir={log_dir}"
                + f" program.data_dir={DATA_DIR}"
                # + " trainer.gpus=[GPU]"
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
