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

# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the MIT License.

# + [markdown] id="NdrXRgjU7Zih"
# # Getting Started
#
# In this tutorial, we demonstrate some of the basic features of TorchGeo and show how easy it is to use if you're already familiar with other PyTorch domain libraries like torchvision.
#
# It's recommended to run this notebook on Google Colab if you don't have your own GPU. Click the "Open in Colab" button above to get started.

# + [markdown] id="lCqHTGRYBZcz"
# ## Setup
#
# First, we install TorchGeo.
# -

# %pip install torchgeo

# + [markdown] id="dV0NLHfGBMWl"
# ## Imports
#
# Next, we import TorchGeo and any other libraries we need.

# + id="entire-albania"
import os
import tempfile

from torch.utils.data import DataLoader

from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

# + [markdown] id="5rLknZxrBEMz"
# ## Datasets
#
# For this tutorial, we'll be using imagery from the [National Agriculture Imagery Program (NAIP)](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip) and labels from the [Chesapeake Bay High-Resolution Land Cover Project](https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/land-cover-data-project/). First, we manually download a few NAIP tiles and create a PyTorch Dataset.

# + colab={"base_uri": "https://localhost:8080/", "height": 232, "referenced_widgets": ["d00a2177bf4b4b8191bfc8796f0e749f", "17d6b81aec50455989276b595457cc7f", "06ccd130058b432dbfa025c102eaeb27", "6bc5b9872b574cb5aa6ebd1d44e7a71f", "f7746f028f874a85b6101185fc9a8efc", "f7ef78d6f87a4a2685788e395525fa7c", "5b2450e316e64b4ba432c78b63275124", "d3bbd6112f144c77bc68e5f2a7a355ff", "a0300b1252cd4da5a798b55c15f8f5fd", "793c2851b6464b398f7b4d2f2f509722", "8dd61c8479d74c95a55de147e04446b3", "b57d15e6c32b4fff8994ae67320972f6", "9a34f8907a264232adf6b0d0543461dd", "e680eda3c84c440083e2959f04431bea", "a073e33fd9ae4125822fc17971233770", "87faaa32454a42939d3bd405e726228c", "b3d4c9c99bec4e69a199e45920d52ce4", "a215f3310ea543d1a8991f57ec824872", "569f60397fd6440d825e8afb83b4e1ae", "b7f604d2ba4e4328a451725973fa755f", "737fa148dfae49a18cc0eabbe05f2d0f", "0b6613adbcc74165a9d9f74988af366e", "b25f274c737d4212b3ffeedb2372ba22", "ef0fc75ff5044171be942a6b3ba0c2da", "612d84013a6e4890a48eb229f6431233", "9a689285370646ab800155432ea042a5", "014ed48a23234e8b81dd7ac4dbf95817", "93c536a27b024728a00486b1f68b4dde", "8a8538a91a74439b81e3f7c6516763e3", "caf540562b484594bab8d6210dd7c2c1", "99cd2e65fb104380953745f2e0a93fac", "c5b818707bb64c5a865236a46399cea2", "54f5db9555c44efa9370cbb7ab58e142", "1d83b20dbb9c4c6a9d5c100fe4770ba4", "c51b2400ca9442a9a9e0712d5778cd9a", "bd2e44a8eb1a4c19a32da5a1edd647d1", "0f9feea4b8344a7f8054c9417150825e", "31acb7a1ca8940078e1aacd72e547f47", "0d0ca8d64d3e4c2f88d87342808dd677", "54402c5f8df34b83b95c94104b26e2c6", "910b98584fa74bb5ad308fe770f5b40e", "b2dce834ee044d69858389178b493a2b", "237f2e31bcfe476baafae8d922877e07", "43ac7d95481b4ea3866feef6ace2f043"]} id="e3138ac3" outputId="11589c46-eee6-455d-839b-390f2934d834"
naip_root = os.path.join(tempfile.gettempdir(), 'naip')
naip_url = (
    'https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/'
)
tiles = [
    'm_3807511_ne_18_060_20181104.tif',
    'm_3807511_se_18_060_20181104.tif',
    'm_3807512_nw_18_060_20180815.tif',
    'm_3807512_sw_18_060_20180815.tif',
]
for tile in tiles:
    download_url(naip_url + tile, naip_root)

naip = NAIP(naip_root)

# + [markdown] id="HQVji2B22Qfu"
# Next, we tell TorchGeo to automatically download the corresponding Chesapeake labels.

# + colab={"base_uri": "https://localhost:8080/"} id="2Ah34KAw2biY" outputId="03b7bdf0-78c1-4a13-ac56-59de740d7f59"
chesapeake_root = os.path.join(tempfile.gettempdir(), 'chesapeake')
os.makedirs(chesapeake_root, exist_ok=True)
chesapeake = ChesapeakeDE(chesapeake_root, crs=naip.crs, res=naip.res, download=True)

# + [markdown] id="OWUhlfpD22IX"
# Finally, we create an IntersectionDataset so that we can automatically sample from both GeoDatasets simultaneously.

# + id="WXxy8F8l2-aC"
dataset = naip & chesapeake

# + [markdown] id="yF_R54Yf3EUd"
# ## Sampler
#
# Unlike typical PyTorch Datasets, TorchGeo GeoDatasets are indexed using lat/long/time bounding boxes. This requires us to use a custom GeoSampler instead of the default sampler/batch_sampler that comes with PyTorch.

# + id="RLczuU293itT"
sampler = RandomGeoSampler(dataset, size=1000, length=10)

# + [markdown] id="OWa-mmYd8S6K"
# ## DataLoader
#
# Now that we have a Dataset and Sampler, we can combine these into a single DataLoader.

# + id="jfx-9ZmU8ZTc"
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

# + [markdown] id="HZIfqqW58oZe"
# ## Training
#
# Other than that, the rest of the training pipeline is the same as it is for torchvision.

# + id="7sGmNvBy8uIg"
for sample in dataloader:
    image = sample['image']
    target = sample['mask']
