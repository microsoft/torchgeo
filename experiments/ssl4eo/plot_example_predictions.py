#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

# import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from torchgeo.datamodules import L7IrishDataModule
from torchgeo.datasets import unbind_samples

device = torch.device("cpu")

# Load weights
path = "data/l7irish/checkpoint-epoch=26-val_loss=0.68.ckpt"
state_dict = torch.load(path, map_location=device)["state_dict"]
state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

# Initialize model
model = smp.Unet(encoder_name="resnet18", in_channels=9, classes=5)
model.to(device)
model.load_state_dict(state_dict)

# Initialize data loaders
datamodule = L7IrishDataModule(
    root="data/l7irish", crs="epsg:3857", download=True, batch_size=1, patch_size=224
)
datamodule.setup("test")

i = 0
for batch in datamodule.test_dataloader():
    image = batch["image"]
    mask = batch["mask"]
    image.to(device)

    # Skip nodata pixels
    if 0 in mask:
        continue

    # Skip boring images
    if len(mask.unique()) < 4:
        continue

    # Make a prediction
    prediction = model(image)
    prediction = prediction.argmax(dim=1)
    prediction.detach().to("cpu")

    batch["prediction"] = prediction

    for sample in unbind_samples(batch):
        # Plot
        # datamodule.test_dataset.plot(sample)
        # plt.show()
        path = f"data/l7irish_predictions/{i}"
        print(f"Saving {path}...")
        os.makedirs(path, exist_ok=True)
        for key in ["image", "mask", "prediction"]:
            data = sample[key]
            if key == "image":
                data = data[[2, 1, 0]].permute(1, 2, 0).numpy().astype("uint8")
                Image.fromarray(data, "RGB").save(f"{path}/{key}.png")
            else:
                data = data * 255 / 4
                data = data.numpy().astype("uint8").squeeze()
                Image.fromarray(data, "L").save(f"{path}/{key}.png")
        i += 1
