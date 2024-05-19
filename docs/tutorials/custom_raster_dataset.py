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

# + [markdown] id="iiqWbXISOEAQ"
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the MIT License.

# + [markdown] id="8zfSLrVHOgwv"
# # Custom Raster Datasets
#
# In this tutorial, we'll describe how to write a custom dataset in TorchGeo. There are many types of datasets that you may encounter, from image data, to segmentation masks, to point labels. We'll focus on the most common type of dataset: a raster file containing an image or mask. Let's get started!

# + [markdown] id="HdTsXvc8UeSS"
# ## Choosing a base class
#
# In TorchGeo, there are two _types_ of datasets:
#
# * `GeoDataset`: for uncurated raw data with geospatial metadata
# * `NonGeoDataset`: for curated benchmark datasets that lack geospatial metadata
#
# If you're not sure which type of dataset you need, a good rule of thumb is to run `gdalinfo` on one of the files. If `gdalinfo` returns information like the bounding box, resolution, and CRS of the file, then you should probably use `GeoDataset`.

# + [markdown] id="S86fPV92Wdc8"
# ### GeoDataset
#
# In TorchGeo, each `GeoDataset` uses an [R-tree](https://en.wikipedia.org/wiki/R-tree) to store the spatiotemporal bounding box of each file or data point. To simplify this process and reduce code duplication, we provide two subclasses of `GeoDataset`:
#
# * `RasterDataset`: recursively search for raster files in a directory
# * `VectorDataset`: recursively search for vector files in a directory
#
# In this example, we'll be working with raster images, so we'll choose `RasterDataset` as the base class.

# + [markdown] id="C3fDQJdvWfsW"
# ### NonGeoDataset
#
# `NonGeoDataset` is almost identical to [torchvision](https://pytorch.org/vision/stable/index.html)'s `VisionDataset`, so we'll instead focus on `GeoDataset` in this tutorial. If you need to add a `NonGeoDataset`, the following tutorials may be helpful:
#
# * [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
# * [Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
# * [Developing Custom PyTorch DataLoaders](https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html)
#
# Of course, you can always look for similar datasets that already exist in TorchGeo and copy their design when creating your own dataset.

# + [markdown] id="ABkJW-3FOi4N"
# ## Setup
#
# First, we install TorchGeo and a couple of other dependencies for downloading data from Microsoft's Planetary Computer.

# + colab={"base_uri": "https://localhost:8080/"} id="aGYEmPNONp8W" outputId="5f8c773c-b1e4-471a-fa99-31c5d7125eb4"
# %pip install torchgeo planetary_computer pystac

# + [markdown] id="MkycnrRMOBso"
# ## Imports
#
# Next, we import TorchGeo and any other libraries we need.

# + id="9v1QN3-mOrdt"
import os
import tempfile
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import planetary_computer
import pystac
import torch
from torch.utils.data import DataLoader

from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler

# %matplotlib inline
plt.rcParams['figure.figsize'] = (12, 12)

# + [markdown] id="W6PAktuVVoSP"
# ## Downloading
#
# Let's download some data to play around with. In this example, we'll create a dataset for loading Sentinel-2 images. Yes, TorchGeo already has a built-in class for this, but we'll use it as an example of the steps you would need to take to add a dataset that isn't yet available in TorchGeo. We'll show how to download a few bands of Sentinel-2 imagery from the Planetary Computer. This may take a few minutes.

# + colab={"base_uri": "https://localhost:8080/", "height": 432, "referenced_widgets": ["ecb1d85ebf264d04885a013a6e7a069c", "4b513ad43aae431dadc2467f370a91c7", "0af96038d774446f99181eece9a172e2", "6c7ead8397fa418d9a67d7b1f68d794a", "0a60e5df954c4874aaed83607e2f20ae", "934ab0cb728541628ee8b09f0a60838f", "66bfa56ececd4da1b1b10c4076c41ca3", "549911c845fa4fbe82725416db96ff76", "381bd014879f4b8cb9713fa640e6e744", "fcedd561ee1e49f79e4e2ef140e34b8d", "5d57d7756e4240e0a7243d817b4af6bd", "77ed782e7b2343829235222ac932f845", "dd23af3773714c009f8996c3c0f84ace", "cc525ae67812410ab3a5135fce36df54", "1651e0cd746e45699df3f9d6c8c7abef", "cb44bd94cdc442ccb0b98f71bffb610f", "c7fc3889e6224808bf919eeb70cde2e7", "16b0250afcb94ce589af663dc7cd9b64", "ed8b05db83c84c9aaca84a35c49b35e1", "957435d60d7e40f0a945c46446943771", "39f7a22b7a9f4844bfc6c89e9d3a94aa", "06e5ccfc007b44b082af8cc4b418a69b", "30a6013e4722448d94e9db91ad6d3e6f", "5149cfb5beab412b90cb27d7662d0230", "512d1ecfc2a74739a9945583a54e9d22", "31cd3702e66c4d678aa9d7be5b672e8c", "59f956f7423541f8b9f63df601422c95", "7eb86dddca194032b143669997c8ee86", "5b27ac2e02874fe1bf22f4af9a026488", "11e0951fb0b440d3a5052487b75a5866", "fac336dcb5424ad1884a6d458b19a05e", "7e775c3f4f1d4f579900e62e235c4cc2", "be81c4fc26f046af93ebc85ed2b9b049", "652084f413184219ae276a6ce73b8fc2", "fa8c6a2f39b94480b42e0739b740dd17", "7605de5a54ef44c9a429763c9dae26d4", "3af584f242aa4283bc4089f461b88bc4", "fbee2fa720764ad28898c7076fa47515", "3d873afcb0a147b5877212e70c14f428", "82a5dfe7afeb4ce1be3c5bc0e79c03c2", "6e25b5a3a9b74736b88cc55fdcad16e7", "8de1e9d8038b4143add60c9c53407a42", "7c62146cce7e41b79f31d1518f4b025f", "7dccd3ba53154811a0e8dd63c4b36b11", "88e4e6e8039e4b80a06b53ccaadfc7db", "845c622bc12a4211bb3cb1bfd93830a3", "26e81ed27e7e4b5bbc68accae6c6051c", "8822d92bc7a6449a851e5140f97f1eb6", "0a19c6e153e54b2c86b3f74a3f9ebc78", "ffe4cc5f5dff4cdd90e1d1afa7f4a210", "f12a91cdbe9c4c269f9111468b4d4473", "5b379fc2a31b4c07a12198d98c9ec48d", "8f0c13446f3846aeac06eba2e2a90a77", "b0addc94caeb4f3ca6ed21f773d68725", "e6564598832544878aa3f90a37cad7ae", "f7e5fd99610a4fa9ab7ac076fcbe9cc1", "ff7f7a7cb5cc4b8bb8a9ee5c8189b9b8", "c53c376437f9408c902554d7ec58dfa5", "5e22879c720f47f59f63b40a7f45a28d", "9a7cac58ce4c4cdf8f7ea9bf348852f9", "0d79a004bedc4ab0beb05410d69edcc5", "45d6cfb69a2f43e3b411f3974ed258a0", "94738bc89cd24297904d8aa71ffab7fa", "e3ccaf4e4ccb40cb89ec0e1bc1bdf6aa", "dc2e555542834dca9807927b31c5ea60", "f1d46cc6d8cd466bae0aafe2345c34eb", "1c943575cfb840b7b6602b7bf384baac", "f4be36be86e7414782716bbf2ea97714", "6e7fb39128f94190ab538ecc5cd2a529", "a339df96ccfd4234805c35a19d4f6be1", "e7874f8bfb4948afb2a9a94495223b5a", "e980b11f57654baf9deb114d22d7b165", "ccfca83232a246b183f1d8887b67bca5", "d8748855e166467e9d8a3c6ce50b8426", "90f125983bd74e04a8ff9ad250919884", "d9ac7ee67f9c491d817c9b95e3ea4735", "af59e0800a7e4788b7626f556d876017", "8050a04b8a9c49598da21dfec7ee7992", "ac013a8894884dcbafbd6046bcc69ace", "bab46fc09bd94f59a3a514f7f5c86298", "6ca58f93c56e435296b7659d454caa71", "0ac7f49774444de48f8d1237c5842cbf", "9291ab2d297d400c9bb89a1479506005", "bccd758b413742b5a661a5842dd42e93", "316708d30a5d4f34b66e423d683ee760", "c44d19c7347e4a15993df5ee72f397fc", "8d1f66bc4d2d4341afc437030a2baafa", "ec3a76861bb74ee998c6b180db50104c"]} id="re1vuzCQfNvr" outputId="ac86ad5f-a0c4-4d80-cbdd-4869cb73b827"
root = os.path.join(tempfile.gettempdir(), 'sentinel')
item_urls = [
    'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20220902T090559_R050_T40XDH_20220902T181115',
    'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20220718T084609_R107_T40XEJ_20220718T175008',
]

for item_url in item_urls:
    item = pystac.Item.from_file(item_url)
    signed_item = planetary_computer.sign(item)
    for band in ['B02', 'B03', 'B04', 'B08']:
        asset_href = signed_item.assets[band].href
        filename = urlparse(asset_href).path.split('/')[-1]
        download_url(asset_href, root, filename)

# + [markdown] id="Hz3uPKcsPLAz"
# This downloads the following files:

# + colab={"base_uri": "https://localhost:8080/"} id="zcBoq3RWPQhn" outputId="ab32f780-a43a-4725-d609-4d4ea35d3ccc"
sorted(os.listdir(root))


# + [markdown] id="Pt-BP66NRkc7"
# As you can see, each spectral band is stored in a different file. We have downloaded 2 total scenes, each with 4 spectral bands.

# + [markdown] id="5BX_C8dJSCZT"
# ## Defining a dataset
#
# To define a new dataset class, we subclass from `RasterDataset`. `RasterDataset` has several class attributes used to customize how to find and load files.
#
# ### `filename_glob`
#
# In order to search for files that belong in a dataset, we need to know what the filenames look like. In our Sentinel-2 example, all files start with a capital `T` and end with `_10m.tif`. We also want to make sure that the glob only finds a single file for each scene, so we'll include `B02` in the glob. If you've never used Unix globs before, see Python's [fnmatch](https://docs.python.org/3/library/fnmatch.html) module for documentation on allowed characters.
#
# ### `filename_regex`
#
# Rasterio can read the geospatial bounding box of each file, but it can't read the timestamp. In order to determine the timestamp of the file, we'll define a `filename_regex` with a group labeled "date". If your files don't have a timestamp in the filename, you can skip this step. If you've never used regular expressions before, see Python's [re](https://docs.python.org/3/library/re.html) module for documentation on allowed characters.
#
# ### `date_format`
#
# The timestamp can come in many formats. In our example, we have the following format:
#
# * 4 digit year (`%Y`)
# * 2 digit month (`%m`)
# * 2 digit day (`%d`)
# * the letter T
# * 2 digit hour (`%H`)
# * 2 digit minute (`%M`)
# * 2 digit second (`%S`)
#
# We'll define the `date_format` variable using [datetime format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).
#
# ### `is_image`
#
# If your data only contains model inputs (such as images), use `is_image = True`. If your data only contains ground truth model outputs (such as segmentation masks), use `is_image = False` instead.
#
# ### `dtype`
#
# Defaults to float32 for `is_image == True` and long for `is_image == False`. This is what you want for 99% of datasets, but can be overridden for tasks like pixel-wise regression (where the target mask should be float32).
#
# ### `resampling`
#
# Defaults to bilinear for float Tensors and nearest for int Tensors. Can be overridden for custom resampling algorithms.
#
# ### `separate_files`
#
# If your data comes with each spectral band in a separate files, as is the case with Sentinel-2, use `separate_files = True`. If all spectral bands are stored in a single file, use `separate_files = False` instead.
#
# ### `all_bands`
#
# If your data is a multispectral image, you can define a list of all band names using the `all_bands` variable.
#
# ### `rgb_bands`
#
# If your data is a multispectral image, you can define which bands correspond to the red, green, and blue channels. In the case of Sentinel-2, this corresponds to B04, B03, and B02, in that order.
#
# Putting this all together into a single class, we get:

# + id="8sFb8BTTTxZD"
class Sentinel2(RasterDataset):
    filename_glob = 'T*_B02_10m.tif'
    filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    date_format = '%Y%m%dT%H%M%S'
    is_image = True
    separate_files = True
    all_bands = ['B02', 'B03', 'B04', 'B08']
    rgb_bands = ['B04', 'B03', 'B02']


# + [markdown] id="a1AlbJp7XUEa"
# We can now instantiate this class and see if it works correctly.

# + colab={"base_uri": "https://localhost:8080/"} id="NXvg9EL8XZAk" outputId="134235ee-b108-4861-f864-ea3d8960b0ce"
dataset = Sentinel2(root)
print(dataset)


# + [markdown] id="msbeAkVOX-iJ"
# As expected, we have a GeoDataset of size 2 because there are 2 scenes in our root data directory.

# + [markdown] id="IUjv7Km7YDpH"
# ## Plotting
#
# A great test to make sure that the dataset works correctly is to try to plot an image. We'll add a plot function to our dataset to help visualize it. First, we need to modify the image so that it only contains the RGB bands, and ensure that they are in the correct order. We also need to ensure that the image is in the range 0.0 to 1.0 (or 0 to 255). Finally, we'll create a plot using matplotlib.

# + id="7PNFOy9mYq6K"
class Sentinel2(RasterDataset):
    filename_glob = 'T*_B02_10m.tif'
    filename_regex = r'^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])'
    date_format = '%Y%m%dT%H%M%S'
    is_image = True
    separate_files = True
    all_bands = ['B02', 'B03', 'B04', 'B08']
    rgb_bands = ['B04', 'B03', 'B02']

    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample['image'][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig


# + [markdown] id="sF8HBA9gah3z"
# Let's plot an image to see what it looks like. We'll use `RandomGeoSampler` to load small patches from each image.

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="I6lv4YcVbAox" outputId="e6ee643f-66bd-457e-f88c-bbedf092e19d"
torch.manual_seed(1)

dataset = Sentinel2(root)
sampler = RandomGeoSampler(dataset, size=4096, length=3)
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

for batch in dataloader:
    sample = unbind_samples(batch)[0]
    dataset.plot(sample)
    plt.axis('off')
    plt.show()


# + [markdown] id="ALLYUzhXKkfS"
# For those who are curious, these are glaciers on Novaya Zemlya, Russia.

# + [markdown] id="R_qrQkBCEvEl"
# ## Custom parameters
#
# If you want to add custom parameters to the class, you can override the `__init__` method. For example, let's say you have imagery that can be automatically downloaded. The `RasterDataset` base class doesn't support this, but you could add support in your subclass. Simply copy the parameters from the base class and add a new `download` parameter.

# + id="TxODAvIHFKNt"
class Downloadable(RasterDataset):
    def __init__(self, root, crs, res, transforms, cache, download=False):
        super().__init__(root, crs, res, transforms, cache)

        if download:
            # download the dataset
            ...

# + [markdown] id="cI43f8DMF3iR"
# ## Contributing
#
# TorchGeo is an open source ecosystem built from the contributions of users like you. If your dataset might be useful for other users, please consider contributing it to TorchGeo! You'll need a bit of documentation and some testing before your dataset can be added, but it will be included in the next minor release for all users to enjoy. See the [Contributing](https://torchgeo.readthedocs.io/en/stable/user/contributing.html) guide to get started.
