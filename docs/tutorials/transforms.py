# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="DYndcZst_kdr"
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the MIT License.

# + [markdown] id="ZKIkyiLScf9P"
# # Transforms

# + [markdown] id="PevsPoE4cY0j"
# In this tutorial, we demonstrate how to use TorchGeo's data augmentation transforms and provide examples of how to utilize them in your experiments with multispectral imagery.
#
# It's recommended to run this notebook on Google Colab if you don't have your own GPU. Click the "Open in Colab" button above to get started.

# + [markdown] id="fsOYw-p2ccka"
# ## Setup

# + [markdown] id="VqdMMzvacOF8"
# Install TorchGeo

# + colab={"base_uri": "https://localhost:8080/"} id="wOwsb8KT_uXR" outputId="e3e5f561-81a8-447b-f149-3e0e8305c598"
# %pip install torchgeo

# + [markdown] id="u2f5_f4X_-vV"
# ## Imports

# + id="cvPMr76K_9uk"
import os
import tempfile

import kornia.augmentation as K
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from torchgeo.datasets import EuroSAT100
from torchgeo.transforms import AugmentationSequential, indices

# + [markdown] id="oR3BCeV2AAop"
# ## Custom Transforms

# + [markdown] id="oVgqhF2udp4z"
# Here we create a transform to show an example of how you can chain custom operations along with TorchGeo and Kornia transforms/augmentations. Note how our transform takes as input a Dict of Tensors. We specify our data by the keys ["image", "mask", "label", etc.] and follow this standard across TorchGeo datasets.


# + id="3mixIK7mAC9G"
class MinMaxNormalize(K.IntensityAugmentationBase2D):
    """Normalize channels to the range [0, 1] using min/max values."""

    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {'mins': mins.view(1, -1, 1, 1), 'maxs': maxs.view(1, -1, 1, 1)}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, int],
        transform: Tensor | None = None,
    ) -> Tensor:
        return (input - flags['mins']) / (flags['maxs'] - flags['mins'] + 1e-10)


# + [markdown] id="2ESh5W05AE3Y"
# ## Dataset Bands and Statistics

# + [markdown] id="WFTBPWUo9b5o"
# Below we have min/max values calculated across the dataset per band. The values were clipped to the interval [0, 98] to stretch the band values and avoid outliers influencing the band histograms.

# + id="vRnMovSrAHgU"
mins = torch.tensor(
    [
        1013.0,
        676.0,
        448.0,
        247.0,
        269.0,
        253.0,
        243.0,
        189.0,
        61.0,
        4.0,
        33.0,
        11.0,
        186.0,
    ]
)
maxs = torch.tensor(
    [
        2309.0,
        4543.05,
        4720.2,
        5293.05,
        3902.05,
        4473.0,
        5447.0,
        5948.05,
        1829.0,
        23.0,
        4894.05,
        4076.05,
        5846.0,
    ]
)
bands = {
    'B01': 'Coastal Aerosol',
    'B02': 'Blue',
    'B03': 'Green',
    'B04': 'Red',
    'B05': 'Vegetation Red Edge 1',
    'B06': 'Vegetation Red Edge 2',
    'B07': 'Vegetation Red Edge 3',
    'B08': 'NIR 1',
    'B8A': 'NIR 2',
    'B09': 'Water Vapour',
    'B10': 'SWIR 1',
    'B11': 'SWIR 2',
    'B12': 'SWIR 3',
}

# + [markdown] id="qQ40TIOG9qVZ"
# The following variables can be used to control the dataloader.

# + id="mKGubYto9qVZ" nbmake={"mock": {"batch_size": 1, "num_workers": 0}}
batch_size = 4
num_workers = 2

# + [markdown] id="hktYHfQHAJbs"
# ## Load the EuroSat MS dataset and dataloader

# + [markdown] id="sUavkZSxeqCA"
# We will use the [EuroSAT](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#eurosat) dataset throughout this tutorial. Specifically, a subset containing only 100 images.

# + colab={"base_uri": "https://localhost:8080/", "height": 269, "referenced_widgets": ["4a45768962f84bfea3828b331f67dbf5", "b87d2d75851848c1b33a00e9cee8bee3", "7e8eb99543654b02b879432d5e286d29", "3858c8543b2349d1aa06410d7f48d49f", "61259e090b754bf4bb53e6be6e3f2233", "b7e748408e824ba8829624c929cae8a9", "d932459633764ddb85dd7fcc3f92c8e2", "77c4e614779f41d39a1e5e23e2a0cb7e", "47b2ca53cd7e4bdeae64192d747eba93", "cc74eb4a618b414aae643b63e316e8be", "f0cdff4785c54dfc9acbbb7052ce9311", "b62e3f13bb584269a8388ebee3de3078", "96e238e381034669b39fbf9d5483e1bb", "f541d671cfb941b0972f622e06eb097d", "bdf19d6aab3b4ec4aa34babe3c9701ce", "f1f2301bee4448cea3e95846f53b7d6f", "a3e02ee9526b4d02b1b47d3705aacaa8", "d1a8127d743741fba825d3378aeb5062", "d3d7851f634c4f3eb7ff0b5f7d433b10", "8aefd08090d540e6ae2b7ec96a91dde0", "c59ae4a8e136486793e1f5aef4b17fb9", "2547bdf147874b9d8c636e0368839149", "c11032f873ea4b4c8edac43bd3caa46c", "9728941761214d74941716810364d0ec", "9eca059d8a7d45178bec2a740c1134a1", "f55f29996c2b466590402d490b8545f6", "99c41c0bfda24961b461844314206525", "37c8dd6f0525490281ac762b8d8469e4", "399a9ea7fed342b7be22c508efe1fd28", "4df7f51c9c4241c29b4857e1c110fc8f", "5d0a21b260d14ea88455d39bebbc6a87", "3f56e822e3a349458b0859b530ec890a", "56c35731ea54485284a5396b89039aba", "8dcaee61833c42ce896b606572ca5ebe", "56d125f8e4ff4923ac5c15a1b803529a", "0914f9a57f914351bc03d9bd6babdc27", "60bf416d480b45c5993d816c46ce19f8", "009688664acf46449e72fd92879f4268", "1c1ff62fb40a41199e82f93f374d7b0b", "11bd6ccbf258495eb177dbed11ecac1a", "2095235bf0024396b23886fc877ca322", "ab01d70248a1450bab0895b4447b1deb", "77595a1376d340aa920a7ac27ab19fc4", "82879fe9b81a47f3899f08513abb42be"]} id="VHVgiNA4t5Tl" outputId="abe91979-9f23-4eed-9589-daa0d69f5458"
root = os.path.join(tempfile.gettempdir(), 'eurosat100')
dataset = EuroSAT100(root, download=True)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
dataloader = iter(dataloader)
print(f'Number of images in dataset: {len(dataset)}')
print(f'Dataset Classes: {dataset.classes}')

# + [markdown] id="ovckKTXpA78o"
# ## Load a sample and batch of images and labels

# + [markdown] id="BKYU2A3weY82"
# Here we test our dataset by loading a single image and label. Note how the image is of shape (13, 64, 64) containing a 64x64 shape with 13 multispectral bands.

# + colab={"base_uri": "https://localhost:8080/"} id="3lhG1yM_v7Mi" outputId="c689890e-80ac-47f9-8779-62f187a6e761"
sample = dataset[0]
x, y = sample['image'], sample['label']
print(x.shape, x.dtype, x.min(), x.max())
print(y, dataset.classes[y])

# + [markdown] id="uw8F17tcAKPY"
# Here we test our dataloader by loading a single batch of images and labels. Note how the image is of shape (4, 13, 64, 64) containing 4 samples due to our batch_size.

# + colab={"base_uri": "https://localhost:8080/"} id="0faJA5UiAJmK" outputId="7448880b-fb51-4c01-c335-767b93868257"
batch = next(dataloader)
x, y = batch['image'], batch['label']
print(x.shape, x.dtype, x.min(), x.max())
print(y, [dataset.classes[i] for i in y])

# + [markdown] id="x8-uLsPdfz0o"
# ## Transforms Usage

# + [markdown] id="p28C8cTGE3dP"
# Transforms are able to operate across batches of samples and singular samples. This allows them to be used inside the dataset itself or externally, chained together with other transform operations using `nn.Sequential`.

# + colab={"base_uri": "https://localhost:8080/"} id="pJXUycffEjNX" outputId="d029826c-a546-4c8e-e254-db680c5045e8"
transform = MinMaxNormalize(mins, maxs)
print(x.shape)
x = transform(x)
print(x.dtype, x.min(), x.max())

# + [markdown] id="KRjb-u0EEmDf"
# Indices can also be computed on batches of images and appended as an additional band to the specified channel dimension. Notice how the number of channels increases from 13 -> 14.

# + colab={"base_uri": "https://localhost:8080/"} id="HaG-1tvi9RKS" outputId="8cbf5fc7-0e34-4670-bf03-700270a041c8"
transform = indices.AppendNDVI(index_nir=7, index_red=3)
batch = next(dataloader)
x = batch['image']
print(x.shape)
x = transform(x)
print(x.shape)

# + [markdown] id="q6WFG8UuGcF8"
# This makes it incredibly easy to add indices as additional features during training by chaining multiple Appends together.

# + colab={"base_uri": "https://localhost:8080/"} id="H_EaAyfnGblR" outputId="b3c7c8c9-1e8b-4125-bf72-69f4973878da"
transforms = nn.Sequential(
    MinMaxNormalize(mins, maxs),
    indices.AppendNDBI(index_swir=11, index_nir=7),
    indices.AppendNDSI(index_green=3, index_swir=11),
    indices.AppendNDVI(index_nir=7, index_red=3),
    indices.AppendNDWI(index_green=2, index_nir=7),
)

batch = next(dataloader)
x = batch['image']
print(x.shape)
x = transforms(x)
print(x.shape)

# + [markdown] id="w4ZbjxPyHoiB"
# It's even possible to chain indices along with augmentations from Kornia for a single callable during training.

# + colab={"base_uri": "https://localhost:8080/"} id="ZKEDgnX0Hn-d" outputId="129a7706-70b8-4d12-8d8c-ff60dc8d44e3"
transforms = AugmentationSequential(
    MinMaxNormalize(mins, maxs),
    indices.AppendNDBI(index_swir=11, index_nir=7),
    indices.AppendNDSI(index_green=3, index_swir=11),
    indices.AppendNDVI(index_nir=7, index_red=3),
    indices.AppendNDWI(index_green=2, index_nir=7),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=['image'],
)

batch = next(dataloader)
print(batch['image'].shape)
batch = transforms(batch)
print(batch['image'].shape)

# + [markdown] id="IhKin8a2GPoI"
# All of our transforms are `nn.Modules`. This allows us to push them and the data to the GPU to see significant gains for large scale operations.

# + colab={"base_uri": "https://localhost:8080/"} id="4QhMOtYzLmVK" outputId="94b8b24a-80a2-4300-df37-aa833a6dde1c" tags=["raises-exception"]
# !nvidia-smi

# + id="4zokGELhGPF8"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = AugmentationSequential(
    MinMaxNormalize(mins, maxs),
    indices.AppendNDBI(index_swir=11, index_nir=7),
    indices.AppendNDSI(index_green=3, index_swir=11),
    indices.AppendNDVI(index_nir=7, index_red=3),
    indices.AppendNDWI(index_green=2, index_nir=7),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomAffine(degrees=(0, 90), p=0.25),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
    data_keys=['image'],
)

transforms_gpu = AugmentationSequential(
    MinMaxNormalize(mins.to(device), maxs.to(device)),
    indices.AppendNDBI(index_swir=11, index_nir=7),
    indices.AppendNDSI(index_green=3, index_swir=11),
    indices.AppendNDVI(index_nir=7, index_red=3),
    indices.AppendNDWI(index_green=2, index_nir=7),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomAffine(degrees=(0, 90), p=0.25),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
    K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
    data_keys=['image'],
).to(device)


def get_batch_cpu():
    return dict(image=torch.randn(64, 13, 512, 512).to('cpu'))


def get_batch_gpu():
    return dict(image=torch.randn(64, 13, 512, 512).to(device))


# + colab={"base_uri": "https://localhost:8080/"} id="vo43CqJ4IIXE" outputId="75e438f7-5ab1-47f4-9de9-b444b5b759f6"
# %%timeit -n 1 -r 5
_ = transforms(get_batch_cpu())

# + colab={"base_uri": "https://localhost:8080/"} id="ICKXYZYrJCeh" outputId="9335cd58-90a6-4b8f-d27c-8bc833e76600"
# %%timeit -n 1 -r 5
_ = transforms_gpu(get_batch_gpu())

# + [markdown] id="nkGy_g6tBAtF"
# ## Visualize Images and Labels

# + [markdown] id="3k4W98v27NtL"
# This is a Google Colab browser for the EuroSAT dataset. Adjust the slider to visualize images in the dataset.

# + id="O_6k7tcxz17x"
transforms = AugmentationSequential(MinMaxNormalize(mins, maxs), data_keys=['image'])
dataset = EuroSAT100(root, transforms=transforms)

# + colab={"base_uri": "https://localhost:8080/", "height": 290} id="Uw8xDeg3BY-u" outputId="2b5f94c4-3aa8-4f30-a38d-b701b2332967"
# @title EuroSat Multispectral (MS) Browser  { run: "auto", vertical-output: true }
idx = 21  # @param {type:"slider", min:0, max:59, step:1}
sample = dataset[idx]
rgb = sample['image'][0, 1:4]
image = T.ToPILImage()(rgb)
print(f"Class Label: {dataset.classes[sample['label']]}")
image.resize((256, 256), resample=Image.BILINEAR)
