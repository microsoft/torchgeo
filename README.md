<img src="logo/logo-color.svg" width="400" alt="TorchGeo"/>

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
$ pip install git+https://github.com/microsoft/torchgeo.git
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
print(dataset[0]["wind_speed"])
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

### Guide for new contributors

To make a contribution to torchgeo follow these steps:

- Fork and clone the repo
- Setup a development environment as shown [above](#installation-instructions)
- Make a great contribution
  - If you are contributing new code, try to also add unit tests (see `tests/` for examples) that check the correctness of your contribution
  - Make sure your contribution passes the [linters](#linters)
- Commit/push your code to your fork of the repo
- Open a pull request
  - Describe your contribution in the description 
  - If you are a new contributor you will be prompted by the CLA bot as described above 
  - Our continuous integration setup (through GitHub actions) will run various linters described below and let you know if there are any problems
  - We'll review and discuss your contribution, hopefully merge it into torchgeo, and thank you for your help in building a library for combining deep learing and geospatial imagery!

If you have questions, feel free to open an Issue or Discussion within the repo and we'll get to it as soon as we can!

### Linters

We use [black](https://github.com/psf/black), [flake8](https://github.com/PyCQA/flake8), and [isort](https://github.com/PyCQA/isort) for style checks, [mypy](https://github.com/python/mypy) for static type analysis, and [pytest](https://github.com/pytest-dev/pytest) for unit tests.
Further, we use [pydocstyle](https://github.com/PyCQA/pydocstyle) and [sphinx](https://github.com/sphinx-doc/sphinx) for documentation. These tools can be daunting to use if you are new to Python software development, but are helpful for maintaining a high quality codebase. The following is a brief tutorial for running these against the torchgeo codebase:

**Option 1** - Run the commands manually

If you setup your development environment as described in the [installation instructions](#installation-instructions), then you can run the following commands in the terminal:
```bash
black --check .
isort . --check --diff
flake8 .
pydocstyle .
mypy .
pytest --cov=. --cov-report=term-missing
```

If any of these fail, then the linters are *not* passing. For example, if `black` is complaining, then running `black .` from the root torchgeo directory will auto format all python files to fix formatting problems. Similarly, `isort .` will reorder all of the `import` statements in an opnionated way.

**Option 2** - Run the commands using [pre-commit](https://pre-commit.com/)

`pre-commit` is a tool that automatically runs linters locally, so that you don't have to remember to run them manually and then have your code flagged by CI. You can setup `pre-commit` with:
```
pip install pre-commit  # install the pre-commit package
pre-commit install  # installs pre-commit hooks in the local repo
pre-commit run --all-files  # runs pre-commit against all files in the repo (and sets up the environment)
```

Now, every time you run `git commit`, then `pre-commit` will run and let you know if any of the files that you *changed* fail the linters. If `pre-commit` passes then your code should be ready (style-wise) for a pull request. Note that you will need to run `pre-commit run --all-files` if any of the hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) change, see [here](https://pre-commit.com/#4-optional-run-against-all-the-files).

**Option 3** - Let our CI setup run these checks for you

Whenever you push to an existing open pull request then the continuous integration workflows will be run against your commit. These run `black`, `isort`, etc. in a containerized environment and will report any problems that they encounter. 

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
