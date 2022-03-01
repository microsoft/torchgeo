# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NAIPCluster dataset."""

import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import Tensor

from torchgeo.datasets.geo import VisionDataset


def rolling_window(
    array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True
) -> np.ndarray:
    """Method for extracting a rolling window of values around each entry in an ndarray.

    Copied from: https://gist.github.com/seberg/3866040

    Create a view of `array` which for every point gives the n-dimensional neighbourhood
    of size window. New dimensions are added at the end of `array` or after the
    corresponding original dimension.

    Args:
        array (int or tuple): Array to which the rolling window is applied.
        window: Either a single integer to create a window of only the last axis or a
            tuple to create it for the last len(window) axes. 0 can be used as a to
            ignore a dimension in the window.
        asteps (tuple): Aligned at the last axis, new steps for the original array, i.e.
            for creation of non-overlapping windows. (Equivalent to slicing result)
        wsteps (int or tuple (the same size as window)): steps for the added window
            dimensions. These can be 0 to repeat values along the axis.
        axes (int or tuple): If given, must have the same size as window. In this case
            window is interpreted as the size in the dimension given by axes. I.e. a
            window of (2, 1) is equivalent to window=2 and axis=-2.
        toend (bool): If False, the new dimensions are right after the corresponding
            original dimension, instead of at the end of the array. Adding the new axes
            at the end makes it easier to get the neighborhood, however toend=False will
            give a more intuitive result if you view the whole array.

    Returns:
        A view on `array` which is smaller to fit the windows and has windows added
            dimensions (0s not counting), ie. every point of `array` is an array of size
            window.
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)  # maybe crude to cast to int...

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps) :] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1  # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError(
            "`window` * `wsteps` larger then `array` in at least one dimension."
        )

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window) :] = window
        _window = _.copy()
        _[-len(window) :] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def apply_model_to_data(image: np.ndarray, r: int, kmeans: KMeans) -> np.ndarray:
    """Runs a KMeans model on an image and returns a mask of cluster indices.

    Args:
        image: an image with shape HxWxC
        r: radius of pixels included in the feature representation when applying the
            KMeans model
        kmeans: a KMeans model, should accept samples with (2*r+1)**2 * C features

    Returns:
        a mask corresponding to `image` where each pixel has been replaced with its
            (cluster index + 1) by the KMeans model and pixels in a border of size `r`
            around the image are set to 0
    """
    height, width, _ = image.shape

    windowed_data = rolling_window(image, (2 * r + 1, 2 * r + 1, 0))
    windowed_data = np.rollaxis(windowed_data, 2, 5)
    windowed_data = windowed_data.reshape(int((height - 2 * r) * (width - 2 * r)), -1)

    labels = kmeans.predict(windowed_data)
    labels = labels.reshape(height - 2 * r, width - 2 * r)

    labels = np.pad(labels + 1, pad_width=r, constant_values=0)

    return labels


class NAIPCluster(VisionDataset):
    """NAIPCluster dataset.

    This dataset contains 50,000 256x256 patches of NAIP imagery sampled uniformly at
    random from the Microsoft Planetary Computer NAIP archive and masks that are
    generated on-the-fly based on a KMeans clustering of pixels in the imagery.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        root: str,
        num_clusters: int = 64,
        cluster_radius: int = 1,
        num_cluster_samples: int = 10000,
        transform: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """Initialize a new NAIPCluster dataset instance.

        Args:
            root: root directory of NAIP pre-sampled dataset
            num_clusters: the number of clusters to use in the KMeans model
            cluster_radius: the radius to use when clustering pixels. E.g. a radius of 0
                will create a KMeans model that just uses the R, G, B, NIR values at a
                single pixel, while a radius of 1 will consider all the spectral values
                in a 3x3 window centered at a pixel.
            num_cluster_samples: number of points used to fit the KMeans model
            transform: usual torch transform to apply to a sample
        """
        self.root = root
        self.num_clusters = num_clusters
        self.cluster_radius = cluster_radius
        self.transform = transform

        self.fns = []
        self.fps = []
        for i in range(10):
            fn = os.path.join(root, f"sample_{i}.npy")
            assert os.path.exists(fn)
            self.fns.append(fn)
            self.fps.append(np.load(fn, mmap_mode="r"))

        # We assume that each file has the same number of samples
        self.num_samples_per_file = int(self.fps[0].shape[0])
        for fp in self.fps:
            assert self.num_samples_per_file == fp.shape[0]

        # We will sample pixels for training KMeans from the first set of NAIP samples
        samples = self.fps[0]
        _, height, width, num_channels = samples.shape

        R = cluster_radius
        x_all = np.zeros(
            (num_cluster_samples, 2 * R + 1, 2 * R + 1, num_channels), dtype=float
        )

        for idx in range(num_cluster_samples):
            i = np.random.randint(self.num_samples_per_file)
            x = np.random.randint(R, width - R)
            y = np.random.randint(R, height - R)
            x_all[idx] = samples[i, y - R : y + R + 1, x - R : x + R + 1].copy()

        x_all_flat = x_all.reshape((num_cluster_samples, -1))

        self.kmeans = KMeans(n_clusters=num_clusters)
        self.kmeans = self.kmeans.fit(x_all_flat)
        del samples, x_all_flat

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and mask at that index with image of dimension 3x1024x1024
            and mask of dimension 1024x1024
        """
        file_idx = index // self.num_samples_per_file
        file_offset = index % self.num_samples_per_file

        img = self.fps[file_idx][file_offset].copy()
        mask = apply_model_to_data(img, self.cluster_radius, self.kmeans)

        img = torch.tensor(np.rollaxis(img, 2, 0))  # type: ignore[attr-defined]
        mask = torch.tensor(mask).long()  # type: ignore[attr-defined]

        sample = {"image": img, "mask": mask}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return self.num_samples_per_file * len(self.fns)

    def recolor_mask(self, mask: np.ndarray) -> np.ndarray:
        """Converts a mask of cluster indices into a reconstructed version of the image.

        Args:
            mask: a mask from a sample returned by :meth:`__getitem__`

        Returns:
            An image reconstructed from the mask and the cluster centroids of the KMeans
                model.
        """
        idxs = (
            mask[
                self.cluster_radius : -self.cluster_radius,
                self.cluster_radius : -self.cluster_radius,
            ]
            - 1
        )
        central_pixel_idx = ((2 * self.cluster_radius + 1) ** 2 // 2) * 4
        img = self.kmeans.cluster_centers_[idxs]
        img = img[:, :, central_pixel_idx : central_pixel_idx + 4]
        img = np.pad(img, pad_width=[(1, 1), (1, 1), (0, 0)], constant_values=0)
        return img / 255.0

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        mask = sample["mask"].squeeze()
        ncols = 2

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 3

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        axs[0].imshow(image[:, :, :3])
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=self.num_clusters + 1)
        axs[1].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(pred, vmin=0, vmax=self.num_clusters + 1)
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
