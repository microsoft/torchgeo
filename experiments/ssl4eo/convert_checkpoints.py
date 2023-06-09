import glob
import os

import torch

paths = glob.glob(os.path.join("**", "checkpoint-epoch=199.ckpt"), recursive=True)
for path in paths:
    ckpt = torch.load(path, map_location=torch.device("cpu"))

    backbone = os.path.dirname(path).split(os.sep)[0].split("-")[-1]
    if "resnet" in backbone:
        state_dict = {
            key.replace("backbone.", "model.backbone.model."): val
            for key, val in ckpt["state_dict"].items()
            if key.startswith("backbone.")
        }
        state_to_save = {
            "state_dict": state_dict,
            "hyper_parameters": {"backbone": backbone},
        }
    elif "vits16" in backbone:
        state_dict = {
            key.replace("backbone.", "model."): val
            for key, val in ckpt["state_dict"].items()
            if key.startswith("backbone.")
        }
        state_to_save = {
            "state_dict": state_dict,
            "hyper_parameters": {"backbone": backbone},
        }
    save_path = os.path.join(os.getcwd(), os.path.dirname(path), "backbone.ckpt")
    torch.save(
        state_to_save,
        os.path.join(os.getcwd(), path.split(os.sep)[0] + "_backbone.ckpt"),
    )
