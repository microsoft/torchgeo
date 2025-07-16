from collections import defaultdict
from pathlib import Path

from numba import njit
import numpy as np
import torch.utils.data as tud
import zarr
from zarr.storage import ZipStore


__all__ = ["L2AZarrDataset", "s2_normalize"]


# Constants
maxR = 3.0  # Max reflectance
midR = 0.13
sat = 1.2
gamma = 1.8

# Gamma adjustment precomputations
gOff = 0.01
gOffPow = gOff**gamma
gOffRange = (1 + gOff) ** gamma - gOffPow


@njit
def clip(val, min_val=0.0, max_val=1.0):
    return np.minimum(np.maximum(val, min_val), max_val)


@njit
def adj(a, tx, ty, maxC):
    ar = clip(a / maxC, 0, 1)
    return ar * (ar * (tx / maxC + ty - 1) - ty) / (ar * (2 * tx / maxC - 1) - tx / maxC)


@njit
def adjGamma(b):
    return (np.power((b + gOff), gamma) - gOffPow) / gOffRange


@njit
def sAdj(a):
    return adjGamma(adj(a, midR, 1, maxR))


@njit
def satEnh(r, g, b):
    ret = np.empty((r.shape[0], r.shape[1], 3), dtype=np.float32)
    avgS = (r + g + b) / 3.0 * (1 - sat)
    ret[:, :, 0] = clip(avgS + r * sat)
    ret[:, :, 1] = clip(avgS + g * sat)
    ret[:, :, 2] = clip(avgS + b * sat)

    return ret


@njit
def sRGB(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 0.41666666666) - 0.055)


@njit
def s2_normalize(B04, B03, B02) -> np.ndarray:
    """Normalizes the S2 bands to produce a visible RGB image."""
    r = sAdj(B04)
    g = sAdj(B03)
    b = sAdj(B02)
    rgbLin = satEnh(r, g, b)
    rgb = sRGB(rgbLin)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


class L2AZarrDataset(tud.Dataset):
    def __init__(self, root_path: str | Path):
        root_path = Path(root_path)
        self.root_path = root_path
        self.keys = [
            x.stem.split(".")[0] for x in (root_path / "s2l2a").glob("*.zarr.zip")
        ]

    def __len__(self) -> int:
        return len(self.keys)
    
    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        def read_zarr_zip(zip_path: str) -> np.ndarray:
            try:
                store = ZipStore(zip_path, mode="r")
                z = zarr.open(store=store, mode="r")
                return z["bands"][:]
            finally:
                store.close()       

        key = self.keys[idx]
        # Ignore L1C and S1 data
        s2l2a = read_zarr_zip(self.root_path / "s2l2a" / f"{key}.zarr.zip")
        s2l2a = s2l2a[0] / 10000.0
        
        ims = defaultdict(list)

        for idx in range(4):
            b432 = s2_normalize(s2l2a[idx, 3], s2l2a[idx, 2], s2l2a[idx, 1])  # rgb
            b765 = s2_normalize(s2l2a[idx, 6], s2l2a[idx, 5], s2l2a[idx, 4])  # vegetation
            b8a = s2_normalize(s2l2a[idx, 9] / 3, s2l2a[idx, 8] / 3, s2l2a[idx, 0] / 3)  # nir
            bw = s2_normalize(s2l2a[idx, 11], s2l2a[idx, 10], s2l2a[idx, 3])  # water
            ims["rgb"].append(b432)
            ims["veg"].append(b765)
            ims["nir"].append(b8a)
            ims["water"].append(bw)

        return {k: np.stack(v, axis=0) for k, v in ims.items()}
