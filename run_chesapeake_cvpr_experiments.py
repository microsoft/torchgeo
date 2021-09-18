# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0, 1, 2]
TEST_MODE = False  # if False then print out the commands to be run, if True then run

# Hyperparameter options
training_set_options = ["de"]
# model_options = ["unet", "deeplabv3+"]
model_options = ["fcn"]
lr_options = [1e-2, 1e-3, 1e-4]
loss_options = ["ce", "jaccard", "focal"]


def do_work(work: "Queue[str]", gpu_idx: int) -> bool:
    """Process for each ID in GPUS."""
    while not work.empty():
        experiment = work.get()
        experiment = experiment.replace("GPU", str(gpu_idx))
        print(experiment)
        if not TEST_MODE:
            subprocess.call(experiment.split(" "))
    return True


def main() -> None:
    """Main."""
    work: Queue[str] = Queue()

    for (train_state, model, lr, loss,) in itertools.product(
        training_set_options,
        model_options,
        lr_options,
        loss_options,
    ):

        experiment_name = f"{train_state}_{model}_{lr}_{loss}_random"

        output_dir = "output/caaken_experiments/"

        if not os.path.exists(os.path.join(output_dir, experiment_name)):

            command = (
                "python train.py"
                + " config_file=conf/chesapeake_cvpr.yaml"
                + f" experiment.name={experiment_name}"
                + f" experiment.module.segmentation_model={model}"
                + f" experiment.module.learning_rate={lr}"
                + f" experiment.module.loss={loss}"
                + f" experiment.datamodule.train_splits=['{train_state}-train']"
                + f" experiment.datamodule.val_splits=['{train_state}-val']"
                + f" experiment.datamodule.test_splits=['{train_state}-test']"
                + f" program.output_dir={output_dir}"
                + f" program.log_dir={output_dir}/logs"
                + " program.data_dir=/home/caleb/cvpr_chesapeake_landcover"
                + " trainer.gpus=[GPU]"
            )
            command = command.strip()

            work.put(command)

    processes = []
    for gpu_idx in GPUS:
        p = Process(
            target=do_work,
            args=(
                work,
                gpu_idx,
            ),
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
