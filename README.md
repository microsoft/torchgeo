<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/logo/logo-color.svg" width="400" alt="TorchGeo"/>

TorchGeo is a [PyTorch](https://pytorch.org/) domain library, similar to [torchvision](https://pytorch.org/vision), that provides datasets, transforms, samplers, and pre-trained models specific to geospatial data.

The goal of this library is to make it simple:

1. for machine learning experts to use geospatial data in their workflows, and
2. for remote sensing experts to use their data in machine learning workflows.

See our [installation instructions](#installation-instructions), [documentation](#documentation), and [examples](#example-usage) to learn how to use torchgeo.

External links:
[![docs](https://readthedocs.org/projects/torchgeo/badge/?version=latest)](https://torchgeo.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/microsoft/torchgeo/branch/main/graph/badge.svg?token=oa3Z3PMVOg)](https://codecov.io/gh/microsoft/torchgeo)

Tests:
[![docs](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml)
[![style](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml)
[![tests](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml)

## Installation instructions

The recommended way to install TorchGeo is with [pip](https://pip.pypa.io/):

```console
$ pip install torchgeo
```

For [conda](https://docs.conda.io/) and [spack](https://spack.io/) installation instructions, see the [documentation](https://torchgeo.readthedocs.io/en/latest/user/installation.html).

## Documentation

You can find the documentation for torchgeo on [ReadTheDocs](https://torchgeo.readthedocs.io).

## Example usage

The following sections give basic examples of what you can do with torchgeo. For more examples, check out our [tutorials](https://torchgeo.readthedocs.io/en/latest/tutorials/getting_started.html).

### Train and test models using our PyTorch Lightning based training script

We provide a script, `train.py` for training models using a subset of the datasets. We do this with the PyTorch Lightning `LightningModule`s and `LightningDataModule`s implemented under the `torchgeo.trainers` namespace.
The `train.py` script is configurable via the command line and/or via YAML configuration files. See the [conf/](conf/) directory for example configuration files that can be customized for different training runs.

```console
$ python train.py config_file=conf/landcoverai.yaml
```

### Download and use the Tropical Cyclone Wind Estimation Competition dataset

This dataset is from a competition hosted by [Driven Data](https://www.drivendata.org/) in collaboration with [Radiant Earth](https://www.radiant.earth/). See [here](https://www.drivendata.org/competitions/72/predict-wind-speeds/) for more information.

Using this dataset in torchgeo is as simple as importing and instantiating the appropriate class.

```python
import torchgeo.datasets

dataset = torchgeo.datasets.TropicalCycloneWindEstimation(split="train", download=True)
print(dataset[0]["image"].shape)
print(dataset[0]["label"])
```

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our [Contribution Guide](https://torchgeo.readthedocs.io/en/latest/user/contributing.html) for more information.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
