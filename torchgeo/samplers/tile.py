import numpy as np
import rasterio
import rasterio.io
import rasterio.merge
import rasterio.windows

from ..datasets import TileDataset
from torch.utils.data import Sampler

class RandomTileGeoSampler(Sampler):

    def __init__(self, dataset: TileDataset, size: int, length: int):
        self.tile_sample_weights = []
        self.tile_heights = []
        self.tile_widths = []
        self.length = length
        self.size = size

        for image_fn in dataset.image_fns:
            with rasterio.open(image_fn) as f:
                image_height, image_width = f.shape
            self.tile_sample_weights.append(image_height * image_width)
            self.tile_heights.append(image_height)
            self.tile_widths.append(image_width)

        self.tile_sample_weights = np.array(self.tile_sample_weights)
        self.tile_sample_weights = (
            self.tile_sample_weights / self.tile_sample_weights.sum()
        )
        self.num_tiles = len(self.tile_sample_weights)

    def __iter__(self):
        for _ in range(len(self)):
            i = np.random.choice(self.num_tiles, p=self.tile_sample_weights)
            y = np.random.randint(0, self.tile_heights[i] - self.size)
            x = np.random.randint(0, self.tile_widths[i] - self.size)

            yield (i, y, x, self.size)

    def __len__(self):
        return self.length


class GridTileGeoSampler(Sampler):

    def __init__(
        self,
        dataset: TileDataset,
        size: int,
        stride=256,
    ):
        self.indices = []
        for i, image_fn in enumerate(dataset.image_fns):
            with rasterio.open(image_fn) as f:
                height, width = f.height, f.width

            for y in list(range(0, height - size, stride)) + [height - size]:
                for x in list(range(0, width - size, stride)) + [width - size]:
                    self.indices.append((i, y, x, size))
        self.num_chips = len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield index

    def __len__(self):
        return self.num_chips
