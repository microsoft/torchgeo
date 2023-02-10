import os
import matplotlib.pyplot as plt
import torch
from .geo import RasterDataset
from .utils import percentile_normalization


class PlanetscopeAnalyticSR(RasterDataset):

    filename_glob = "*AnalyticMS_SR*.tif"
    # filename_regex = r"^.(?P<date>\d{8}_\d{6})_"
    # date_format = "%Y%m%d_%H%M%S"
    is_image = True
    separate_files = False
    all_bands = ["B", "G", "R", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

            # Reorder and rescale the image
            image = sample["image"][rgb_indices].permute(1, 2, 0)
            image = torch.clamp(image / 10000, min=0, max=1).numpy()
        #         image = np.rollaxis(sample["image"][:3].numpy(), 0, 3)
        #         image = percentile_normalization(image, axis=(0, 1))

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig
