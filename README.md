<img src="logo/logo-color.svg" width="400" alt="TorchGeo"/>

Torchgeo is a PyTorch based library, similar to torchvision, that provides datasets, transforms, samplers, and pre-trained models specifically for geospatial data.

Our goals for this library are to make it simple 1.) for machine learning experts to use geospatial data in their workflows, and 2.) for remote sensing experts to use their data in machine learning workflows.

See our [installation instructions](#installation-instructions), [documentation](#documentation), and [examples](#example-usage) to learn how to use torchgeo.

[![docs](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/docs.yaml)
[![style](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml)
[![tests](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml)


## Installation instructions

Until the first release, you can install an environment compatible with torchgeo with `conda`, `pip`, or `spack` as shown below.

### Conda

**Note**: we assume you have access to a GPU and include the `pytorch-gpu` package from the conda-forge channel in "environment.yml".

```bash
conda config --set channel_priority strict
conda env create --file environment.yml
conda activate torchgeo

# verify that the PyTorch can use the GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Pip

With Python 3.6 or later:

```bash
pip install -r requirements.txt
```

### Spack

```bash
TODO
```

## Documentation

You can find the documentation for torchgeo on ReadTheDocs [TODO](TODO).


## Example usage

The following sections give basic examples of what you can do with torchgeo. For more examples, check out our documentation [TODO](TODO).

### Train and test models using our PyTorch Lightning based training script

```bash
# run the training script with a config file
python train.py config_file=conf/landcoverai.yaml
```

### Download and use the Tropical Cyclone Wind Estimation Competition dataset

```python
import torchgeo


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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
