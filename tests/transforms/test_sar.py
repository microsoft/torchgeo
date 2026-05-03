# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import numpy as np
import pytest
import torch

from torchgeo.datasets.utils import Sample
from torchgeo.transforms import LeeFilter, RefinedLeeFilter
from torchgeo.transforms.sar import lee_filter, refined_lee_filter

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


def _make_step_edge(seed: int = 0, size: int = 96) -> np.ndarray:
    """Half-dark / half-bright field with unit-mean exponential speckle.

    Used by the edge-preservation regression test for refined_lee_filter:
    the edge along the central column should survive better through
    refined_lee than through basic lee_filter on the same input.
    """
    rng = np.random.default_rng(seed)
    signal = np.full((size, size), 1.0, dtype=np.float64)
    signal[:, : size // 2] = 5.0
    speckle = rng.exponential(scale=1.0, size=(size, size))
    return signal * speckle


class TestRefinedLeeFilterFunction:
    def test_preserves_shape(self) -> None:
        x = torch.rand(2, 3, 32, 32)
        out = refined_lee_filter(x, num_looks=1.0)
        assert out.shape == x.shape

    def test_non_negative_output(self) -> None:
        x = torch.rand(1, 1, 32, 32) * 10
        out = refined_lee_filter(x, num_looks=1.0)
        assert (out >= 0).all()

    def test_reduces_variance_in_homogeneous_region(self) -> None:
        rng = np.random.default_rng(42)
        speckle = rng.exponential(scale=1.0, size=(64, 64)) * 3.0
        x = torch.from_numpy(speckle).float().unsqueeze(0).unsqueeze(0)
        out = refined_lee_filter(x, num_looks=1.0).squeeze().numpy()
        assert out.var() < speckle.var() * 0.5

    def test_preserves_edge_better_than_basic_lee(self) -> None:
        """The value-prop test: refined_lee preserves the step edge better."""
        img_np = _make_step_edge(seed=0)
        x = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
        size = img_np.shape[1]
        edge_col = size // 2

        basic = lee_filter(x, window_size=7, num_looks=1.0).squeeze().numpy()
        refined = refined_lee_filter(x, num_looks=1.0).squeeze().numpy()

        basic_grad = abs(
            basic[:, edge_col - 4 : edge_col].mean()
            - basic[:, edge_col : edge_col + 4].mean()
        )
        refined_grad = abs(
            refined[:, edge_col - 4 : edge_col].mean()
            - refined[:, edge_col : edge_col + 4].mean()
        )

        assert refined_grad >= basic_grad

    def test_falls_back_to_box_in_homogeneous(self) -> None:
        """With a high edge threshold, refined_lee should match basic lee_filter."""
        rng = np.random.default_rng(7)
        speckle = rng.exponential(scale=1.0, size=(48, 48)) * 2.0
        x = torch.from_numpy(speckle).float().unsqueeze(0).unsqueeze(0)

        basic = lee_filter(x, window_size=7, num_looks=1.0)
        refined_high_threshold = refined_lee_filter(
            x, num_looks=1.0, edge_threshold_sigma=1e6
        )
        np.testing.assert_allclose(
            refined_high_threshold.numpy(), basic.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_edge_threshold_extremes_differ(self) -> None:
        """edge_threshold_sigma=0 (always refined) and large (always box) differ."""
        img_np = _make_step_edge(seed=2)
        x = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)

        always_refined = refined_lee_filter(x, num_looks=1.0, edge_threshold_sigma=0.0)
        always_box = refined_lee_filter(x, num_looks=1.0, edge_threshold_sigma=1e6)
        assert not torch.allclose(always_refined, always_box, rtol=1e-3)

    @pytest.mark.parametrize('num_looks', [1.0, 5.0])
    def test_num_looks_variants(self, num_looks: float) -> None:
        x = torch.rand(1, 1, 32, 32) * 4.0
        out = refined_lee_filter(x, num_looks=num_looks)
        assert out.shape == x.shape
        assert (~torch.isnan(out)).all()

    @pytest.mark.parametrize('bad', [0.0, -1.0])
    def test_rejects_invalid_num_looks(self, bad: float) -> None:
        with pytest.raises(ValueError, match='num_looks'):
            refined_lee_filter(torch.zeros(1, 1, 16, 16), num_looks=bad)

    def test_gradient_flow(self) -> None:
        x = torch.rand(1, 1, 16, 16, requires_grad=True)
        out = refined_lee_filter(x, num_looks=1.0)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_constant_image_returns_constant(self) -> None:
        """All-ones input must not produce NaNs (zero variance edge case)."""
        x = torch.ones(1, 1, 32, 32)
        out = refined_lee_filter(x, num_looks=1.0)
        assert (~torch.isnan(out)).all()
        np.testing.assert_allclose(out.numpy(), 1.0, atol=1e-5)


class TestRefinedLeeFilter:
    def test_sample(self, sample: Sample) -> None:
        aug = K.AugmentationSequential(
            RefinedLeeFilter(p=1.0), keepdim=True, data_keys=None
        )
        output = aug(sample)
        assert output['image'].shape == sample['image'].shape

    def test_batch(self, batch: Sample) -> None:
        aug = K.AugmentationSequential(RefinedLeeFilter(p=1.0), data_keys=None)
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    @pytest.mark.parametrize('num_looks', [1.0, 5.0])
    def test_num_looks(self, num_looks: float, batch: Sample) -> None:
        aug = K.AugmentationSequential(
            RefinedLeeFilter(num_looks=num_looks, p=1.0), data_keys=None
        )
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    @pytest.mark.parametrize('edge_threshold_sigma', [0.0, 2.0, 4.0])
    def test_edge_threshold_sigma(
        self, edge_threshold_sigma: float, batch: Sample
    ) -> None:
        aug = K.AugmentationSequential(
            RefinedLeeFilter(edge_threshold_sigma=edge_threshold_sigma, p=1.0),
            data_keys=None,
        )
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    def test_same_on_batch(self, batch: Sample) -> None:
        aug = K.AugmentationSequential(
            RefinedLeeFilter(p=1.0, same_on_batch=True), data_keys=None
        )
        output = aug(batch)
        assert output['image'].shape == batch['image'].shape

    def test_p_zero_is_identity(self, batch: Sample) -> None:
        aug = K.AugmentationSequential(RefinedLeeFilter(p=0.0), data_keys=None)
        output = aug(batch)
        assert torch.equal(output['image'], batch['image'])

    @pytest.mark.parametrize('bad', [0.0, -1.0])
    def test_rejects_invalid_num_looks(self, bad: float) -> None:
        with pytest.raises(ValueError, match='num_looks'):
            RefinedLeeFilter(num_looks=bad)
