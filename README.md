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
[![docs](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml)
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
from torchgeo.datasets import CDL, Landsat7, Landsat8, TropicalCycloneWindEstimation, stack_samples
from torchgeo.samplers import RandomGeoSampler
```

### Composing geospatial datasets

Many use cases involve working with geospatial data and combining it in intelligent ways. In this example, we assume that the user has Landsat 7 and 8 imagery downloaded. We first create a single dataset containing all Landsat imagery. We only use the Landsat 7 bands for Landsat 8 and take the union between both datasets.

```python
landsat7 = Landsat7(root="...")
landsat8 = Landsat8(root="...", bands=landsat7.bands)
landsat = landsat7 | landsat8
```

Next, we take the intersection between this dataset and the Cropland Data Layer (CDL) dataset. We want to take the intersection instead of the union to ensure that we only sample from regions that have both Landsat and CDL data. Note that we can automatically download and checksum CDL data. Also note that each of these datasets may contain files in different coordinate reference systems (CRS) or resolutions, but TorchGeo automatically ensures that a matching CRS and resolution is used.

```python
cdl = CDL(root="...", download=True, checksum=True)
dataset = landsat & cdl
```

This dataset can now be used with a PyTorch data loader. In order to sample from this dataset using geospatial coordinates, we create a random sampler class.

```python
sampler = RandomGeoSampler(dataset, size=256, length=10000)
dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, collate_fn=stack_samples)
```

This data loader can now be used in your normal training/evaluation pipeline.

```python
for batch in dataloader:
    # train a model, or make predictions using a pre-trained model
```

### Download and use the Tropical Cyclone Wind Estimation Competition dataset

This dataset is from a competition hosted by [Driven Data](https://www.drivendata.org/) in collaboration with [Radiant Earth](https://www.radiant.earth/). See [here](https://www.drivendata.org/competitions/72/predict-wind-speeds/) for more information.

Using this dataset in TorchGeo is as simple as importing and instantiating the appropriate class.

```python
import torchgeo.datasets

dataset = torchgeo.datasets.TropicalCycloneWindEstimation(split="train", download=True)
print(dataset[0]["image"].shape)
print(dataset[0]["label"])
```

### Train and test models using our PyTorch Lightning based training script

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
