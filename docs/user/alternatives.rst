Related Libraries
=================

TorchGeo is not the only **geospatial machine learning library** out there, there are a number of alternatives that you can consider using. The goal of this page is to provide an up-to-date listing of these libraries and the features they support in order to help you decide which library is right for you. Criteria for inclusion on this list include:

* **geospatial**: Must be primarily intended for working with geospatial, remote sensing, or satellite imagery data. This rules out libraries like `torchvision`_, which provides little to no support for multispectral data or geospatial transforms.
* **machine learning**: Must provide basic machine learning functionality. This rules out libraries like `GDAL`_, which is useful for data loading but offers no support for machine learning.
* **library**: Must be an actively developed software library with testing and releases on repositories like PyPI or CRAN. This rules out libraries like `TorchSat`_, `RoboSat`_, and `Solaris`_, which have been abandoned and are no longer maintained.

When deciding which library is most useful to you, it is worth considering the features they support, how actively the library is being developed, and how popular the library is, roughly in that order.

.. note::

   Software is a living, breathing organism and is constantly undergoing change. If any of the above information is incorrect or out of date, or if you want to add a new project to this list, please open a PR!

   *Last updated: 31 March 2025*

Features
--------

**Key**: ✅ full support, 🚧 partial support, ❌ no support

.. csv-table::
   :align: center
   :file: metrics/features.csv
   :header-rows: 1
   :widths: auto

\*Support was dropped in newer releases.

**ML Backend**: The machine learning libraries used by the project. For example, if you are a scikit-learn user, eo-learn may be perfect for you, but if you need more advanced deep learning support, you may want to choose a different library.

**I/O Backend**: The I/O libraries used by the project to read data. This gives you a rough idea of which file formats are supported. For example, if you need to work with lidar data, a project that uses laspy may be important to you.

**Spatial Backend**: The spatial library used to perform spatial joins and compute intersections based on geospatial metadata. This may be important to you if you intend to scale up your simulations.

**Transform Backend**: The transform library used to perform data augmentation. For example, Kornia performs all augmentations on PyTorch Tensors, allowing you to run your transforms on the GPU for an entire mini-batch at a time.

**Datasets**: The number of geospatial datasets built into the library. Note that most projects have something similar to TorchGeo's ``RasterDataset`` and ``VectorDataset``, allowing you to work with generic raster and vector files. Collections of datasets are only counted a single time, so data loaders for Landsats 1--9 are a single dataset, and data loaders for SpaceNets 1--8 are also a single dataset.

**Weights**: The number of model weights pre-trained on geospatial data that are offered by the library. Note that most projects support hundreds of model architectures via a library like PyTorch Image Models, and can use models pre-trained on ImageNet. There are far fewer libraries that provide foundation model weights pre-trained on multispectral satellite imagery.

**CLI**: Whether or not the library has a command-line interface. This low-code or no-code solution is convenient for users with limited programming experience, and can offer nice features for reproducing research and fast experimentation.

**Reprojection**: Whether or not the library supports automatic reprojection and resampling of data. Without this, users are forced to manually warp data using a library like GDAL if they want to combine datasets in different coordinate systems or spatial resolutions.

**STAC**: Whether or not the library supports the spatiotemporal asset catalog. STAC is becoming a popular means of indexing into spatiotemporal data like satellite imagery.

**Time-Series**: Whether or not the library supports time-series modeling. For many remote sensing applications, time-series data provide important signals.

GitHub
------

These are metrics that can be scraped from GitHub.

.. csv-table::
   :align: right
   :file: metrics/github.csv
   :header-rows: 1
   :widths: auto

**Contributors**: The number of contributors. This is one of the most important metrics for project development. The more developers you have, the higher the `bus factor <https://en.wikipedia.org/wiki/Bus_factor>`_, and the more likely the project is to survive. More contributors also means more new features and bug fixes.

**Forks**: The number of times the git repository has been forked. This gives you an idea of how many people are attempting to modify the source code, even if they have not (yet) contributed back their changes.

**Watchers**: The number of people watching activity on the repository. These are people who are interested enough to get notifications for every issue, PR, release, or discussion.

**Stars**: The number of people who have starred the repository. This is not the best metric for number of users, and instead gives you a better idea about the amount of *hype* surrounding the project.

**Issues**: The total number of open and closed issues. Although it may seem counterintuitive, the more issues, the better. Large projects like PyTorch have tens of thousands of open issues. This does not mean that PyTorch is broken, it means that it is popular and has enough issues to discover corner cases or open feature requests.

**PRs**: The total number of open and closed pull requests. This tells you how active development of the project has been. Note that this metric can be artificially inflated by bots like dependabot.

**Releases**: The number of software releases. The frequency of releases varies from project to project. The important thing to look for is multiple releases.

**Commits**: The number of commits on the main development branch. This is another metric for how active development has been. However, this can vary a lot depending on whether PRs are merged with or without squashing first.

**Core SLOCs**: The number of source lines of code in the core library, excluding empty lines and comments. This tells you how large the library is, and how long it would take someone to write something like it themselves. We use `scc <https://github.com/boyter/scc>`_ to compute SLOCs and exclude markup languages from the count.

**Test SLOCs**: The number of source lines of code in the testing suite, excluding empty lines and comments. This tells you how well tested the project is. A good goal to strive for is a similar amount of code for testing as there is in the core library itself.

**Test Coverage**: The percentage of the core library that is hit by unit tests. This is especially important for interpreted languages like Python and R where there is no compiler type checking. 100% test coverage is ideal, but 80% is considered good.

**License**: The license the project is distributed under. For commercial researchers, this may be very important and decide whether or not they are able to use the software.

Downloads
---------

These are download metrics for the project. Note that these numbers can be artificially inflated by mirrors and installs during continuous integration. They give you a better idea of the number of projects that depend on a library than the number of users of that library.

.. csv-table::
   :align: right
   :file: metrics/downloads.csv
   :header-rows: 1
   :widths: auto

**PyPI Downloads**: The number of downloads from the Python Packaging Index. PyPI download metrics are computed by `PyPI Stats <https://pypistats.org/>`_ and `PePy <https://www.pepy.tech/>`_.

**CRAN Downloads**: The number of downloads from the Comprehensive R Archive Network. CRAN download metrics are computed by `Meta CRAN <https://cranlogs.r-pkg.org/>`_ and `DataScienceMeta <https://www.datasciencemeta.com/rpackages>`_.

**Conda Downloads**: The number of downloads from Conda Forge. Conda download metrics are computed by `Conda Forge <https://anaconda.org/conda-forge/>`_.

.. _torchvision: https://github.com/pytorch/vision
.. _GDAL: https://github.com/OSGeo/gdal
.. _TorchSat: https://github.com/sshuair/torchsat
.. _RoboSat: https://github.com/mapbox/robosat
.. _Solaris: https://github.com/CosmiQ/solaris

.. _TorchGeo: https://github.com/microsoft/torchgeo
.. _eo-learn: https://github.com/sentinel-hub/eo-learn
.. _Raster Vision: https://github.com/azavea/raster-vision
.. _DeepForest: https://github.com/weecology/DeepForest
.. _samgeo: https://github.com/opengeos/segment-geospatial
.. _TerraTorch: https://github.com/IBM/terratorch
.. _SITS: https://github.com/e-sensing/sits
.. _scikit-eo: https://github.com/yotarazona/scikit-eo
.. _geo-bench: https://github.com/ServiceNow/geo-bench
.. _GeoAI: https://github.com/opengeos/geoai
.. _OTBTF: https://github.com/remicres/otbtf
.. _GeoDeep: https://github.com/uav4geo/GeoDeep
.. _srai: https://github.com/kraina-ai/srai