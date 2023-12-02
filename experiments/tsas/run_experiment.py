import subprocess


if __name__ == "__main__":
    for i in range(10):
        subprocess.run([
            "torchgeo", "fit",
            "--config", "experiments/tsas/cyclone.yaml",
            "--seed_everything", str(i),
            "--trainer.default_root_dir", f"logs/cyclone/resnet18_{i}",
        ])
