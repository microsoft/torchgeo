<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/logo/logo-color.svg" width="400" alt="TorchGeo logo"/>

TorchGeo is a [PyTorch](https://pytorch.org/) domain library, similar to [torchvision](https://pytorch.org/vision), providing datasets, samplers, transforms, and pre-trained models specific to geospatial data.

The goal of this library is to make it simple:

1. for machine learning experts to work with geospatial data, and
2. for remote sensing experts to explore machine learning solutions.

Community:
[![slack](https://img.shields.io/badge/slack-join-7C0385?logo=slack)](https://join.slack.com/t/torchgeo/shared_invite/zt-22rse667m-eqtCeNW0yI000Tl4B~2PIw)
[![osgeo](https://img.shields.io/badge/OSGeo-join-4CB05B?logo=osgeo)](https://www.osgeo.org/community/getting-started-osgeo/)
[![pytorch](https://img.shields.io/badge/PyTorch-join-DE3412?logo=pytorch)](https://pytorch.org/ecosystem/join)

Packaging:
[![pypi](https://badge.fury.io/py/torchgeo.svg)](https://pypi.org/project/torchgeo/)
[![conda](https://anaconda.org/conda-forge/torchgeo/badges/version.svg)](https://anaconda.org/conda-forge/torchgeo)
[![spack](https://img.shields.io/spack/v/py-torchgeo)](https://packages.spack.io/package.html?name=py-torchgeo)

Testing:
[![docs](https://readthedocs.org/projects/torchgeo/badge/?version=latest)](https://torchgeo.readthedocs.io/en/stable/)
[![style](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/style.yaml)
[![tests](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml/badge.svg)](https://github.com/microsoft/torchgeo/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/microsoft/torchgeo/branch/main/graph/badge.svg?token=oa3Z3PMVOg)](https://codecov.io/gh/microsoft/torchgeo)

## Installation

The recommended way to install TorchGeo is with [pip](https://pip.pypa.io/):

```console
$ pip install torchgeo
```

For [conda](https://docs.conda.io/) and [spack](https://spack.io/) installation instructions, see the [documentation](https://torchgeo.readthedocs.io/en/stable/user/installation.html).

## Documentation

You can find the documentation for TorchGeo on [ReadTheDocs](https://torchgeo.readthedocs.io). This includes API documentation, contributing instructions, and several [tutorials](https://torchgeo.readthedocs.io/en/stable/tutorials/getting_started.html). For more details, check out our [paper](https://arxiv.org/abs/2111.08872), [podcast episode](https://www.youtube.com/watch?v=ET8Hb_HqNJQ), [tutorial](https://www.youtube.com/watch?v=R_FhY8aq708), and [blog post](https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/).

<p float="left">
    <a href="https://www.youtube.com/watch?v=ET8Hb_HqNJQ">
        <img src="https://img.youtube.com/vi/ET8Hb_HqNJQ/0.jpg" style="width:49%;">
    </a>
    <a href="https://www.youtube.com/watch?v=R_FhY8aq708">
        <img src="https://img.youtube.com/vi/R_FhY8aq708/0.jpg" style="width:49%;">
    </a>
</p>

## Example Usage

The following sections give basic examples of what you can do with TorchGeo.

First we'll import various classes and functions used in the following sections:

```python
from lightning.pytorch import Trainer
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
landsat7 = Landsat7(root="...", bands=["B1", ..., "B7"])
landsat8 = Landsat8(root="...", bands=["B2", ..., "B8"])
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
from torch.utils.data import DataLoader

from torchgeo.datamodules.utils import collate_fn_detection
from torchgeo.datasets import VHR10

# Initialize the dataset
dataset = VHR10(root="...", download=True, checksum=True)

# Initialize the dataloader with the custom collate function
dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn_detection,
)

# Training loop
for batch in dataloader:
    images = batch["image"]  # list of images
    boxes = batch["boxes"]  # list of boxes
    labels = batch["labels"]  # list of labels
    masks = batch["masks"]  # list of masks

    # train a model, or make predictions using a pre-trained model
```

<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/images/vhr10.png" alt="Example predictions from a Mask R-CNN model trained on the VHR-10 dataset"/>

All TorchGeo datasets are compatible with PyTorch data loaders, making them easy to integrate into existing training workflows. The only difference between a benchmark dataset in TorchGeo and a similar dataset in torchvision is that each dataset returns a dictionary with keys for each PyTorch `Tensor`.

### Pre-trained Weights

Pre-trained weights have proven to be tremendously beneficial for transfer learning tasks in computer vision. Practitioners usually utilize models pre-trained on the ImageNet dataset, containing RGB images. However, remote sensing data often goes beyond RGB with additional multispectral channels that can vary across sensors. TorchGeo is the first library to support models pre-trained on different multispectral sensors, and adopts torchvision's [multi-weight API](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/). A summary of currently available weights can be seen in the [docs](https://torchgeo.readthedocs.io/en/stable/api/models.html#pretrained-weights). To create a [timm](https://github.com/huggingface/pytorch-image-models) Resnet-18 model with weights that have been pretrained on Sentinel-2 imagery, you can do the following:

```python
import timm
from torchgeo.models import ResNet18_Weights

weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"], num_classes=10)
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
```

These weights can also directly be used in TorchGeo Lightning modules that are shown in the following section via the `weights` argument. For a notebook example, see this [tutorial](https://torchgeo.readthedocs.io/en/stable/tutorials/pretrained_weights.html).

### Reproducibility with Lightning

In order to facilitate direct comparisons between results published in the literature and further reduce the boilerplate code needed to run experiments with datasets in TorchGeo, we have created Lightning [*datamodules*](https://torchgeo.readthedocs.io/en/stable/api/datamodules.html) with well-defined train-val-test splits and [*trainers*](https://torchgeo.readthedocs.io/en/stable/api/trainers.html) for various tasks like classification, regression, and semantic segmentation. These datamodules show how to incorporate augmentations from the kornia library, include preprocessing transforms (with pre-calculated channel statistics), and let users easily experiment with hyperparameters related to the data itself (as opposed to the modeling process). Training a semantic segmentation model on the [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) dataset is as easy as a few imports and four lines of code.

```python
datamodule = InriaAerialImageLabelingDataModule(root="...", batch_size=64, num_workers=6)
task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet50",
    weights=True,
    in_channels=3,
    num_classes=2,
    loss="ce",
    ignore_index=None,
    lr=0.1,
    patience=6,
)
trainer = Trainer(default_root_dir="...")

trainer.fit(model=task, datamodule=datamodule)
```

<img src="https://raw.githubusercontent.com/microsoft/torchgeo/main/images/inria.png" alt="Building segmentations produced by a U-Net model trained on the Inria Aerial Image Labeling dataset"/>

TorchGeo also supports command-line interface training using [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). It can be invoked in two ways:

```console
# If torchgeo has been installed
torchgeo
# If torchgeo has been installed, or if it has been cloned to the current directory
python3 -m torchgeo
```

It supports command-line configuration or YAML/JSON config files. Valid options can be found from the help messages:

```console
# See valid stages
torchgeo --help
# See valid trainer options
torchgeo fit --help
# See valid model options
torchgeo fit --model.help ClassificationTask
# See valid data options
torchgeo fit --data.help EuroSAT100DataModule
```

Using the following config file:
```yaml
trainer:
  max_epochs: 20
model:
  class_path: ClassificationTask
  init_args:
    model: "resnet18"
    in_channels: 13
    num_classes: 10
data:
  class_path: EuroSAT100DataModule
  init_args:
    batch_size: 8
  dict_kwargs:
    download: true
```

we can see the script in action:
```console
# Train and validate a model
torchgeo fit --config config.yaml
# Validate-only
torchgeo validate --config config.yaml
# Calculate and report test accuracy
torchgeo test --config config.yaml --trainer.ckpt_path=...
```

It can also be imported and used in a Python script if you need to extend it to add new features:

```python
from torchgeo.main import main

main(["fit", "--config", "config.yaml"])
```

See the [Lightning documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) for more details.

## Citation

If you use this software in your work, please cite our [paper](https://dl.acm.org/doi/10.1145/3557915.3560953):
```bibtex
@inproceedings{Stewart_TorchGeo_Deep_Learning_2022,
    address = {Seattle, Washington},
    author = {Stewart, Adam J. and Robinson, Caleb and Corley, Isaac A. and Ortiz, Anthony and Lavista Ferres, Juan M. and Banerjee, Arindam},
    booktitle = {Proceedings of the 30th International Conference on Advances in Geographic Information Systems},
    doi = {10.1145/3557915.3560953},
    month = {11},
    pages = {1--12},
    publisher = {Association for Computing Machinery},
    series = {SIGSPATIAL '22},
    title = {{TorchGeo}: Deep Learning With Geospatial Data},
    url = {https://dl.acm.org/doi/10.1145/3557915.3560953},
    year = {2022}
}
```

## Contributing

This project welcomes contributions and suggestions. If you would like to submit a pull request, see our [Contribution Guide](https://torchgeo.readthedocs.io/en/stable/user/contributing.html) for more information.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
