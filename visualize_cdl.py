import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch

from torchgeo.datamodules import Sentinel2CDLDataModule
from torchgeo.datasets import unbind_samples
from torchgeo.models import ResNet18_Weights

device = torch.device("cpu")

# Load weights
# path = "data/l7irish/checkpoint-epoch=26-val_loss=0.68.ckpt"
# state_dict = torch.load(path, map_location=device)["state_dict"]
# state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

weights = ResNet18_Weights.SENTINEL2_ALL_MOCO


# Initialize model
model = smp.Unet(encoder_name="resnet18", encoder_weights=weights, in_channels=13, classes=13)
model.to(device)
# model.load_state_dict(state_dict)

# Initialize data loaders
# datamodule = Sentinel2CDLDataModule(
#     root="data/l7irish", crs="epsg:3857", batch_size=64, patch_size=224
# )
# datamodule.setup("test")

# for batch in datamodule.test_dataloader():
#     image = batch["image"]
#     mask = batch["mask"]
#     image.to(device)

#     # Make a prediction
#     prediction = model(image)
#     prediction = prediction.argmax(dim=1)
#     prediction.detach().to("cpu")

#     batch["prediction"] = prediction

#     for sample in unbind_samples(batch):
#         # Skip nodata pixels
#         if 0 in sample["mask"]:
#             continue

#         # Skip boring images
#         if len(sample["mask"].unique()) < 3:
#             continue

#         # Plot
#         datamodule.test_dataset.plot(sample)
#         plt.show()