Installation
============

TorchGeo is simple and easy to install. We support installation using `uv <https://docs.astral.sh/uv/>`_, `pip <https://pip.pypa.io/en/stable/>`_, `conda <https://docs.conda.io/en/latest/>`_, and `spack <https://spack.io/>`_ package managers.

uv
--

`uv` is a fast Python package installer and resolver. It's built by the same team behind `ruff` and is often significantly quicker than `pip`.

To install TorchGeo using `uv`, ensure `uv` is installed on your system first (e.g., via `pip install uv` or following `uv`'s official installation guide):

.. code-block:: console

   $ uv pip install torchgeo

If you want to install a development version, you can use a VCS project URL:

.. code-block:: console

   $ uv pip install git+https://github.com/microsoft/torchgeo.git

or a local git checkout:

.. code-block:: console

   $ git clone https://github.com/microsoft/torchgeo.git
   $ cd torchgeo
   $ uv pip install .

By default, only required dependencies are installed. TorchGeo has a number of optional dependencies for specific datasets, models, or development. These can be installed with a comma-separated list:

.. code-block:: console

   $ uv pip install torchgeo[datasets,models]
   $ uv pip install torchgeo[style,tests]
   $ uv pip install torchgeo[all]

See the ``pyproject.toml`` for a complete list of options. See the `uv documentation <https://docs.astral.sh/uv/>`_ for more details.

pip
---

Alternatively, you can install TorchGeo using `pip`, the standard Python package installer:

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

By default, only required dependencies are installed. TorchGeo has a number of optional dependencies for specific datasets, models, or development. These can be installed with a comma-separated list:

.. code-block:: console

   $ pip install torchgeo[datasets,models]
   $ pip install torchgeo[style,tests]
   $ pip install torchgeo[all]

See the ``pyproject.toml`` for a complete list of options. See the `pip documentation <https://pip.pypa.io/en/stable/>`_ for more details.

conda
-----

First, you'll want to configure conda to only use the conda-forge channel:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict

Now, you can install the latest stable release using:

.. code-block:: console

   $ conda install torchgeo

Conda does not support development versions or optional dependencies directly through ``conda install`` for this package. If you install from conda-forge, only stable releases and required dependencies will be installed. Development versions or optional dependencies can be installed afterwards using `uv` or `pip`.

See the `conda-forge documentation <https://conda-forge.org/>`_ for more details.

spack
-----

If you are working in an HPC environment or want to install your software from source, the easiest way is with spack:

.. code-block:: console

   $ spack install py-torchgeo
   $ spack load py-torchgeo

Our spack package has a ``main`` version that can be used to install the latest commit:

.. code-block:: console

   $ spack install py-torchgeo@main
   $ spack load py-torchgeo

Optional dependencies can be installed by enabling build variants:

.. code-block:: console

   $ spack install py-torchgeo+datasets+models
   $ spack install py-torchgeo+style+tests

Run ``spack info py-torchgeo`` for a complete list of variants. See the `spack documentation <https://spack.readthedocs.io/en/latest/>`_ for more details.