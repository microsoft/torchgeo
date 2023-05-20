import rasterio
import rasterio.io
import rasterio.merge
import rasterio.windows
import torch
from torch.utils.data import Dataset


class TileDataset(Dataset):

    def __init__(self, image_fns, mask_fns=None, transforms=None, sanity_check=False):
        super().__init__()
        self.image_fns = image_fns
        self.mask_fns = mask_fns
        if self.mask_fns is not None:
            assert len(image_fns) == len(mask_fns)

        if sanity_check and mask_fns is not None:
            for image_fn, mask_fn in zip(image_fns, mask_fns):
                with rasterio.open(image_fn) as f:
                    image_height, image_width = f.shape
                with rasterio.open(mask_fn) as f:
                    mask_height, mask_width = f.shape
                assert image_height == mask_height
                assert image_width == mask_width

        self.transforms = transforms

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        i, y, x, patch_size = index
        assert 0 <= i < len(self.image_fns)

        sample = {
            "y": y,
            "x": x,
        }

        window = rasterio.windows.Window(
            x, y, patch_size, patch_size
        )

        image_fn = self.image_fns[i]
        with rasterio.open(image_fn) as f:
            image = f.read(window=window)
        sample["image"] = torch.from_numpy(image).float()

        if self.mask_fns is not None:
            mask_fn = self.mask_fns[i]
            with rasterio.open(mask_fn) as f:
                mask = f.read(window=window)
            sample["mask"] = torch.from_numpy(mask).long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
