<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/logo/logo-color.svg" width="400" alt="TorchGeo logo"/>

TorchGeo is a [PyTorch](https://pytorch.org/) domain library, similar to [torchvision](https://pytorch.org/vision), providing datasets, samplers, transforms, and pre-trained models specific to geospatial data.

The goal of this library is to make it simple:

1. for machine learning experts to work with geospatial data, and
2. for remote sensing experts to explore machine learning solutions.

Testing:
[![docs](https://readthedocs.org/projects/torchgeo/badge/?version=latest)](https://torchgeo.readthedocs.io/en/stable/)
[![style](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml)
[![tests](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/microsoft/torchgeo/branch/main/graph/badge.svg?token=oa3Z3PMVOg)](https://codecov.io/gh/microsoft/torchgeo)

Packaging:
[![pypi](https://badge.fury.io/py/torchgeo.svg)](https://pypi.org/project/torchgeo/)
[![conda](https://anaconda.org/conda-forge/torchgeo/badges/version.svg)](https://anaconda.org/conda-forge/torchgeo)
[![spack](https://img.shields.io/spack/v/py-torchgeo)](https://spack.readthedocs.io/en/latest/package_list.html#py-torchgeo)

## Installation

The recommended way to install TorchGeo is with [pip](https://pip.pypa.io/):

```console
$ pip install torchgeo
```

For [conda](https://docs.conda.io/) and [spack](https://spack.io/) installation instructions, see the [documentation](https://torchgeo.readthedocs.io/en/stable/user/installation.html).

## Documentation

You can find the documentation for TorchGeo on [ReadTheDocs](https://torchgeo.readthedocs.io). This includes API documentation, contributing instructions, and several [tutorials](https://torchgeo.readthedocs.io/en/stable/tutorials/getting_started.html). For more details, check out our [paper](https://arxiv.org/abs/2111.08872) and [blog](https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/).

## Example Usage

The following sections give basic examples of what you can do with TorchGeo.

First we'll import various classes and functions used in the following sections:

```python
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.datasets import CDL, Landsat7, Landsat8, VHR10, stack_samples
from torchgeo.samplers import RandomGeoSampler
from torchgeo.trainers import SemanticSegmentationTask
```

### Geospatial datasets and samplers

Many remote sensing applications involve working with [*geospatial datasets*](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#geospatial-datasets)—datasets with geographic metadata. These datasets can be challenging to work with due to the sheer variety of data. Geospatial imagery is often multispectral with a different number of spectral bands and spatial resolution for every satellite. In addition, each file may be in a different coordinate reference system (CRS), requiring the data to be reprojected into a matching CRS.

<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/images/geodataset.png" alt="Example application in which we combine Landsat and CDL and sample from both"/>

In this example, we show how easy it is to work with geospatial data and to sample small image patches from a combination of [Landsat](https://www.usgs.gov/landsat-missions) and [Cropland Data Layer (CDL)](https://data.nal.usda.gov/dataset/cropscape-cropland-data-layer) data using TorchGeo. First, we assume that the user has Landsat 7 and 8 imagery downloaded. Since Landsat 8 has more spectral bands than Landsat 7, we'll only use the bands that both satellites have in common. We'll create a single dataset including all images from both Landsat 7 and 8 data by taking the union between these two datasets.

```python
landsat7 = Landsat7(root="...")
landsat8 = Landsat8(root="...", bands=Landsat8.all_bands[1:-2])
landsat = landsat7 | landsat8
```

Next, we take the intersection between this dataset and the CDL dataset. We want to take the intersection instead of the union to ensure that we only sample from regions that have both Landsat and CDL data. Note that we can automatically download and checksum CDL data. Also note that each of these datasets may contain files in different coordinate reference systems (CRS) or resolutions, but TorchGeo automatically ensures that a matching CRS and resolution is used.

```python
cdl = CDL(root="...", download=True, checksum=True)
dataset = landsat & cdl
```

This dataset can now be used with a PyTorch data loader. Unlike benchmark datasets, geospatial datasets often include very large images. For example, the CDL dataset consists of a single image covering the entire continental United States. In order to sample from these datasets using geospatial coordinates, TorchGeo defines a number of [*samplers*](https://torchgeo.readthedocs.io/en/stable/api/samplers.html). In this example, we'll use a random sampler that returns 256 x 256 pixel images and 10,000 samples per epoch. We also use a custom collation function to combine each sample dictionary into a mini-batch of samples.

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

Many applications involve intelligently composing datasets based on geospatial metadata like this. For example, users may want to:

* Combine datasets for multiple image sources and treat them as equivalent (e.g., Landsat 7 and 8)
* Combine datasets for disparate geospatial locations (e.g., Chesapeake NY and PA)

These combinations require that all queries are present in at least one dataset, and can be created using a `UnionDataset`. Similarly, users may want to:

* Combine image and target labels and sample from both simultaneously (e.g., Landsat and CDL)
* Combine datasets for multiple image sources for multimodal learning or data fusion (e.g., Landsat and Sentinel)

These combinations require that all queries are present in both datasets, and can be created using an `IntersectionDataset`. TorchGeo automatically composes these datasets for you when you use the intersection (`&`) and union (`|`) operators.

### Benchmark datasets

TorchGeo includes a number of [*benchmark datasets*](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#non-geospatial-datasets)—datasets that include both input images and target labels. This includes datasets for tasks like image classification, regression, semantic segmentation, object detection, instance segmentation, change detection, and more.

If you've used [torchvision](https://pytorch.org/vision) before, these datasets should seem very familiar. In this example, we'll create a dataset for the Northwestern Polytechnical University (NWPU) very-high-resolution ten-class ([VHR-10](https://github.com/chaozhong2010/VHR-10_dataset_coco)) geospatial object detection dataset. This dataset can be automatically downloaded, checksummed, and extracted, just like with torchvision.

```python
dataset = VHR10(root="...", download=True, checksum=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

for batch in dataloader:
    image = batch["image"]
    label = batch["label"]

    # train a model, or make predictions using a pre-trained model
```

<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/images/vhr10.png" alt="Example predictions from a Mask R-CNN model trained on the NWPU VHR-10 dataset"/>

All TorchGeo datasets are compatible with PyTorch data loaders, making them easy to integrate into existing training workflows. The only difference between a benchmark dataset in TorchGeo and a similar dataset in torchvision is that each dataset returns a dictionary with keys for each PyTorch `Tensor`.

### Reproducibility with PyTorch Lightning

In order to facilitate direct comparisons between results published in the literature and further reduce the boilerplate code needed to run experiments with datasets in TorchGeo, we have created PyTorch Lightning [*datamodules*](https://torchgeo.readthedocs.io/en/stable/api/datamodules.html) with well-defined train-val-test splits and [*trainers*](https://torchgeo.readthedocs.io/en/stable/api/trainers.html) for various tasks like classification, regression, and semantic segmentation. These datamodules show how to incorporate augmentations from the kornia library, include preprocessing transforms (with pre-calculated channel statistics), and let users easily experiment with hyperparameters related to the data itself (as opposed to the modeling process). Training a semantic segmentation model on the [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) dataset is as easy as a few imports and four lines of code.

```python
datamodule = InriaAerialImageLabelingDataModule(root_dir="...", batch_size=64, num_workers=6)
task = SemanticSegmentationTask(segmentation_model="unet", encoder_weights="imagenet", learning_rate=0.1)
trainer = Trainer(gpus=1, default_root_dir="...")

trainer.fit(model=task, datamodule=datamodule)
```

<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/images/inria.png" alt="Building segmentations produced by a U-Net model trained on the Inria Aerial Image Labeling dataset"/>

In our GitHub repo, we provide `train.py` and `evaluate.py` scripts to train and evaluate the performance of models using these datamodules and trainers. These scripts are configurable via the command line and/or via YAML configuration files. See the [conf](https://github.com/microsoft/torchgeo/blob/main/conf) directory for example configuration files that can be customized for different training runs.

```console
$ python train.py config_file=conf/landcoverai.yaml
```

## Citation

If you use this software in your work, please cite our [paper](https://arxiv.org/abs/2111.08872):
```
@article{Stewart_TorchGeo_deep_learning_2021,
    author = {Stewart, Adam J. and Robinson, Caleb and Corley, Isaac A. and Ortiz, Anthony and Lavista Ferres, Juan M. and Banerjee, Arindam},
    journal = {arXiv preprint arXiv:2111.08872},
    month = {11},
    title = {{TorchGeo}: Deep Learning With Geospatial Data},
    url = {https://github.com/microsoft/torchgeo},
    year = {2021}
}
```

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our [Contribution Guide](https://torchgeo.readthedocs.io/en/stable/user/contributing.html) for more information.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
