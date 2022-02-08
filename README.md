<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/logo/logo-color.svg" width="400" alt="TorchGeo"/>

TorchGeo is a [PyTorch](https://pytorch.org/) domain library, similar to [torchvision](https://pytorch.org/vision), that provides datasets, transforms, samplers, and pre-trained models specific to geospatial data.

The goal of this library is to make it simple:

1. for machine learning experts to use geospatial data in their workflows, and
2. for remote sensing experts to use their data in machine learning workflows.

See our [installation instructions](#installation), [documentation](#documentation), and [examples](#example-usage) to learn how to use TorchGeo.

External links:
[![docs](https://readthedocs.org/projects/torchgeo/badge/?version=latest)](https://torchgeo.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/microsoft/torchgeo/branch/main/graph/badge.svg?token=oa3Z3PMVOg)](https://codecov.io/gh/microsoft/torchgeo)
[![pypi](https://badge.fury.io/py/torchgeo.svg)](https://pypi.org/project/torchgeo/)
[![conda](https://anaconda.org/conda-forge/torchgeo/badges/version.svg)](https://anaconda.org/conda-forge/torchgeo)
[![spack](https://img.shields.io/spack/v/py-torchgeo)](https://spack.readthedocs.io/en/latest/package_list.html#py-torchgeo)

Tests:
[![style](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml)
[![tests](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml)

## Installation

The recommended way to install TorchGeo is with [pip](https://pip.pypa.io/):

```console
$ pip install torchgeo
```

For [conda](https://docs.conda.io/) and [spack](https://spack.io/) installation instructions, see the [documentation](https://torchgeo.readthedocs.io/en/latest/user/installation.html).

## Documentation

You can find the documentation for TorchGeo on [ReadTheDocs](https://torchgeo.readthedocs.io).

## Example Usage

The following sections give basic examples of what you can do with TorchGeo. For more examples, check out our [tutorials](https://torchgeo.readthedocs.io/en/latest/tutorials/getting_started.html).

First we'll import various classes and functions used in the following sections:

```python
from torch.utils.data import DataLoader
from torchgeo.datasets import CDL, COWCDetection, Landsat7, Landsat8, stack_samples
from torchgeo.samplers import RandomGeoSampler
```

### Benchmark datasets

TorchGeo includes a number of [*benchmark*](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#non-geospatial-datasets) datasets, datasets that include both input images and target labels. This includes datasets for tasks like image classification, regression, semantic segmentation, object detection, instance segmentation, change detection, and more.

If you've used [torchvision](https://pytorch.org/vision) before, these datasets should seem very familiar. In this example, we'll create a dataset for the Cars Overhead With Context (COWC) car detection dataset. This dataset can be automatically downloaded, checksummed, and extracted, just like with torchvision.

```python
dataset = COWCDetection(root="...", split="train", download=True, checksum=True)
```

This dataset can then be passed to a PyTorch data loader.

```python
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
```

The only difference between a benchmark dataset in TorchGeo and a similar dataset in torchvision is that each dataset returns a dictionary with keys for each PyTorch Tensor.

```python
for batch in dataloader:
    image = batch["image"]
    label = batch["label"]

    # train a model, or make predictions using a pre-trained model
```

### Geospatial datasets

Many remote sensing applications involve working with [*generic*](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#geospatial-datasets) geospatial data. This data can be challenging to work with due to the sheer variety of data. Geospatial imagery is often multispectral with a different number of spectral bands and spatial resolution for every satellite. In addition, each file may be in a different coordinate reference system (CRS), requiring the data to be reprojected into a matching CRS.

In this example, we show how easy it is to work with geospatial data and to sample small image patches from a combination of Landsat and Cropland Data Layer (CDL) data using TorchGeo. First, we assume that the user has Landsat 7 and 8 imagery downloaded. Since Landsat 8 has more spectral bands than Landsat 7, we'll only use the bands that both satellites have in common. We'll create a single dataset including all images from both Landsat 7 and 8 data by taking the union between these two datasets.

```python
landsat7 = Landsat7(root="...")
landsat8 = Landsat8(root="...", bands=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"])
landsat = landsat7 | landsat8
```

Next, we take the intersection between this dataset and the Cropland Data Layer (CDL) dataset. We want to take the intersection instead of the union to ensure that we only sample from regions that have both Landsat and CDL data. Note that we can automatically download and checksum CDL data. Also note that each of these datasets may contain files in different coordinate reference systems (CRS) or resolutions, but TorchGeo automatically ensures that a matching CRS and resolution is used.

```python
cdl = CDL(root="...", download=True, checksum=True)
dataset = landsat & cdl
```

This dataset can now be used with a PyTorch data loader. Unlike benchmark datasets, geospatial datasets often include very large images. For example, the CDL dataset consists of a single image covering the entire continental United States. In order to sample from these datasets using geospatial coordinates, TorchGeo defines a number of [*samplers*](https://torchgeo.readthedocs.io/en/latest/api/samplers.html). In this example, we'll use a random sampler that returns 256x256 pixel images and an epoch length of 10,000 images. We also use a custom collation function to combine each sample dictionary into a mini-batch of samples.

```python
sampler = RandomGeoSampler(dataset, size=256, length=10000)
dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, collate_fn=stack_samples)
```

This data loader can now be used in your normal training/evaluation pipeline.

```python
for batch in dataloader:
    image = batch["image"]
    mask = batch["mask"]

    # train a model, or make predictions using a pre-trained model
```

### Train and test models using our PyTorch Lightning-based training script

We provide a script, `train.py` for training models using a subset of the datasets. We do this with the PyTorch Lightning `LightningModule`s and `LightningDataModule`s implemented under the `torchgeo.trainers` namespace.
The `train.py` script is configurable via the command line and/or via YAML configuration files. See the [conf/](conf/) directory for example configuration files that can be customized for different training runs.

```console
$ python train.py config_file=conf/landcoverai.yaml
```

## Citation

If you use this software in your work, please cite [our paper](https://arxiv.org/abs/2111.08872):
```
@article{Stewart_TorchGeo_deep_learning_2021,
    author = {Stewart, Adam J. and Robinson, Caleb and Corley, Isaac A. and Ortiz, Anthony and Lavista Ferres, Juan M. and Banerjee, Arindam},
    journal = {arXiv preprint arXiv:2111.08872},
    month = {11},
    title = {{TorchGeo: deep learning with geospatial data}},
    url = {https://github.com/microsoft/torchgeo},
    year = {2021}
}
```

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our [Contribution Guide](https://torchgeo.readthedocs.io/en/latest/user/contributing.html) for more information.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
