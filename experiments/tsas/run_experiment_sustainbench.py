import subprocess


if __name__ == "__main__":
    for i in range(10):
        subprocess.run([
            "torchgeo", "fit",
            "--config", "experiments/tsas/sustainbench.yaml",
            "--seed_everything", str(i),
            "--trainer.default_root_dir", f"logs/sustainbench/resnet18_{i}",
        ])

    for i in range(10):
        subprocess.run([
            "torchgeo", "fit",
            "--config", "experiments/tsas/sustainbench.yaml",
            "--seed_everything", str(i),
            "--trainer.default_root_dir", f"logs/sustainbench/resnet50_{i}",
            "--model.model", "resnet50",
        ])
