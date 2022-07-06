Installation
============

TorchGeo is simple and easy to install. We support installation using the `pip <https://pip.pypa.io/>`_, `conda <https://docs.conda.io/>`_, and `spack <https://spack.io/>`_ package managers.

pip
---

Since TorchGeo is written in pure-Python, the easiest way to install it is using pip:

.. code-block:: console

   $ pip install torchgeo


If you want to install a development version, you can use a VCS project URL:

.. code-block:: console

   $ pip install git+https://github.com/microsoft/torchgeo.git


or a local git checkout:

.. code-block:: console

   $ git clone https://github.com/microsoft/torchgeo.git
   $ cd torchgeo
   $ pip install .


By default, only required dependencies are installed. TorchGeo has a number of optional dependencies for specific datasets or development. These can be installed with a comma-separated list:

.. code-block:: console

   $ pip install torchgeo[datasets]
   $ pip install torchgeo[style,tests]


See the ``setup.cfg`` for a complete list of options. See the `pip documentation <https://pip.pypa.io/>`_ for more details.

conda
-----

If you need to install non-Python dependencies like PyTorch, it's better to use a package manager like conda. First, you'll want to configure conda to only use the conda-forge channel:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict


Now, you can install the latest stable release using:

.. code-block:: console

   $ conda install torchgeo

.. note:: The installation of torchgeo in this manner is not supported on Windows since pytorch from the conda-forge channel currently does not support Windows. Users are recommended to create a custom conda environment and install torchgeo as shown below.

Conda does not directly support installing development versions, but you can use conda to install our dependencies, then use pip to install TorchGeo itself.

.. code-block:: console

   $ git clone https://github.com/microsoft/torchgeo.git
   $ cd torchgeo
   $ conda env create --file environment.yml
   $ conda activate torchgeo
   $ pip install .

Conda does not directly support optional dependencies. If you install from conda-forge, only required dependencies will be installed by default. Optional dependencies can be installed afterwards using pip. If you install using the ``environment.yml`` file, all optional dependencies are installed by default.

See the `conda-forge documentation <https://conda-forge.org/>`_ for more details.

spack
-----

If you are working in an HPC environment or want to install your software from source, the easiest way is with spack:

.. code-block:: console

   $ spack install py-torchgeo
   $ spack load py-torchgeo


Our Spack package has a ``main`` version that can be used to install the latest commit:

.. code-block:: console

   $ spack install py-torchgeo@main
   $ spack load py-torchgeo

Optional dependencies can be installed by enabling build variants:

.. code-block:: console

   $ spack install py-torchgeo+datasets
   $ spack install py-torchgeo+style+tests

Run ``spack info py-torchgeo`` for a complete list of variants.

See the `spack documentation <https://spack.readthedocs.io/>`_ for more details.
