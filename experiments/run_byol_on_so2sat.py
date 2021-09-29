"""Runs the train script with a grid of hyperparameters."""
import itertools
import os
import subprocess
from multiprocessing import Process, Queue

# list of GPU IDs that we want to use, one job will be started for every ID in the list
GPUS = [0]
DRY_RUN = False  # if False then print out the commands to be run, if True then run
DATA_DIR = ""  # path to the So2Sat data directory

# Hyperparameter options
encoder_options = ["resnet18", "resnet50"]
lr_options = [.2, 1e-2, 1e-3]
imagenet_pretraining_options = [True, False]


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

    for (encoder, lr, imagenet_pretraining) in itertools.product(
        encoder_options,
        lr_options,
        imagenet_pretraining_options,
    ):

        experiment_name = f"{encoder}_{lr}_{imagenet_pretraining}"

        output_dir = os.path.join("output", "byol_so2sat_experiments")
        log_dir = os.path.join(output_dir, "logs")
        config_file = os.path.join("conf", "byol.yaml")

        if not os.path.exists(os.path.join(output_dir, experiment_name)):

            command = (
                "python train.py"
                + f" config_file={config_file}"
                + f" experiment.name={experiment_name}"
                + f" experiment.module.encoder={encoder}"
                + f" experiment.module.lr={lr}"
                + f" experiment.module.imagenet_pretraining={imagenet_pretraining}"
                + f" program.output_dir={output_dir}"
                + f" program.log_dir={log_dir}"
                + f" program.data_dir={DATA_DIR}"
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