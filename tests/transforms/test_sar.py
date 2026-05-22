# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import numpy as np
import pytest
import torch

from torchgeo.datasets.utils import Sample
from torchgeo.transforms import LeeFilter
from torchgeo.transforms.sar import lee_filter

pytest.importorskip('scipy', minversion='1.11.2')

from scipy.ndimage import uniform_filter


@pytest.fixture
def sample() -> Sample:
    rng = np.random.default_rng(0)
    speckle = rng.exponential(scale=1.0, size=(1, 8, 8))
    return {
        'image': torch.from_numpy(speckle).float(),
        'mask': torch.zeros(1, 8, 8, dtype=torch.long),
    }


@pytest.fixture
def batch() -> Sample:
    rng = np.random.default_rng(1)
    speckle = rng.exponential(scale=1.0, size=(2, 1, 8, 8))
    return {
        'image': torch.from_numpy(speckle).float(),
        'mask': torch.zeros(2, 1, 8, 8, dtype=torch.long),
    }


def _numpy_lee_reference(
    image: np.ndarray, window_size: int, num_looks: float, eps: float = 1e-8
) -> np.ndarray:
    """Reference Lee filter using scipy.ndimage on a 2D NumPy array."""
    image = image.astype(np.float64)
    sigma_v_sq = 1.0 / num_looks
    mean_local = uniform_filter(image, size=window_size, mode='mirror')
    mean_sq_local = uniform_filter(image * image, size=window_size, mode='mirror')
    var_local = np.clip(mean_sq_local - mean_local**2, 0.0, None)
    var_signal = np.clip(var_local - sigma_v_sq * mean_local**2, 0.0, None)
    weight = var_signal / (var_signal + sigma_v_sq * mean_local**2 + eps)
    return mean_local + weight * (image - mean_local)


def _make_synthetic_sar(seed: int = 0, size: int = 64) -> np.ndarray:
    """Bright square on dark background with unit-mean exponential speckle."""
    rng = np.random.default_rng(seed)
    signal = np.ones((size, size), dtype=np.float64)
    signal[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 5.0
    speckle = rng.exponential(scale=1.0, size=(size, size))
    return signal * speckle


class TestLeeFilterFunction:
    @pytest.mark.parametrize('window_size', [3, 5, 7])
    @pytest.mark.parametrize('num_looks', [1.0, 5.0])
    def test_matches_numpy_reference(self, window_size: int, num_looks: float) -> None:
        img_np = _make_synthetic_sar()
        ref = _numpy_lee_reference(img_np, window_size, num_looks)
        x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        out = lee_filter(x, window_size=window_size, num_looks=num_looks)
        out_np = out.squeeze().numpy()
        np.testing.assert_allclose(out_np, ref, rtol=1e-5, atol=1e-5)

    def test_preserves_shape(self) -> None:
        x = torch.rand(2, 3, 16, 16)
        out = lee_filter(x, window_size=5)
        assert out.shape == x.shape

    def test_non_negative_output(self) -> None:
        x = torch.rand(1, 1, 16, 16) * 10
        out = lee_filter(x, window_size=7)
        assert (out >= 0).all()

    def test_reduces_variance_in_homogeneous_region(self) -> None:
        rng = np.random.default_rng(42)
        speckle = rng.exponential(scale=1.0, size=(64, 64)) * 3.0
        x = torch.from_numpy(speckle).float().unsqueeze(0).unsqueeze(0)
        out = lee_filter(x, window_size=9).squeeze().numpy()
        assert out.var() < speckle.var() * 0.5

    def test_preserves_edge_contrast_vs_mean(self) -> None:
        img_np = _make_synthetic_sar(seed=1)
        x = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
        lee_out = lee_filter(x, window_size=7).squeeze().numpy()
        mean_out = uniform_filter(img_np, size=7, mode='mirror')
        size = img_np.shape[0]
        strip_lee = lee_out[size // 4 - 4 : size // 4 + 4, :].std()
        strip_mean = mean_out[size // 4 - 4 : size // 4 + 4, :].std()
        assert strip_lee > strip_mean

    @pytest.mark.parametrize('bad', [0, -1, 4, 8])
    def test_rejects_invalid_window_size(self, bad: int) -> None:
        with pytest.raises(ValueError, match='window_size'):
            lee_filter(torch.zeros(1, 1, 8, 8), window_size=bad)

    @pytest.mark.parametrize('bad', [0.0, -1.0])
    def test_rejects_invalid_num_looks(self, bad: float) -> None:
        with pytest.raises(ValueError, match='num_looks'):
            lee_filter(torch.zeros(1, 1, 8, 8), window_size=5, num_looks=bad)


class TestLeeFilter:
    def test_sample(self, sample: Sample) -> None:
        aug = K.AugmentationSequential(
            LeeFilter(window_size=5, p=1.0), keepdim=True, data_keys=None
        )
        output = aug(sample)
        assert output['image'].shape == sample['image'].shape

    def test_batch(self, batch: Sample) -> None:
        aug = K.AugmentationSequential(LeeFilter(window_size=5, p=1.0), data_keys=None)
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    @pytest.mark.parametrize('num_looks', [1.0, 5.0])
    def test_num_looks(self, num_looks: float, batch: Sample) -> None:
        aug = K.AugmentationSequential(
            LeeFilter(window_size=5, num_looks=num_looks, p=1.0), data_keys=None
        )
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    def test_same_on_batch(self, batch: Sample) -> None:
        aug = K.AugmentationSequential(
            LeeFilter(window_size=5, p=1.0, same_on_batch=True), data_keys=None
        )
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    def test_p_zero_is_identity(self, batch: Sample) -> None:
        aug = K.AugmentationSequential(LeeFilter(p=0.0), data_keys=None)
        output = aug(batch)
        assert torch.equal(output['image'], batch['image'])

    @pytest.mark.parametrize('bad', [0, -1, 4, 8])
    def test_rejects_invalid_window_size(self, bad: int) -> None:
        with pytest.raises(ValueError, match='window_size'):
            LeeFilter(window_size=bad)

    @pytest.mark.parametrize('bad', [0.0, -1.0])
    def test_rejects_invalid_num_looks(self, bad: float) -> None:
        with pytest.raises(ValueError, match='num_looks'):
            LeeFilter(num_looks=bad)
