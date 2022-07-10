Contributing
============

The TorchGeo project welcomes contributions and suggestions! If you think you've found a bug or would like to suggest a new feature, you can `open an issue on GitHub <https://github.com/microsoft/torchgeo/issues/new/choose>`_. TorchGeo is an open-source community-supported project, so we try to address issues in order of severity or impact. If you feel confident, the fastest way to make changes to TorchGeo is to submit a pull request. This guide explains everything you need to know about contributing to TorchGeo.

.. note:: TorchGeo is a library for geospatial datasets, transforms, and models. If you would like to add a new transform or model that doesn't involve geospatial data or isn't specific to the remote sensing domain, you're better off adding it to a general purpose computer vision library like `torchvision <https://github.com/pytorch/vision>`_ or `Kornia <https://github.com/kornia/kornia>`_.


Git
---

All development is done on GitHub. If you would like to submit a pull request, you'll first want to fork https://github.com/microsoft/torchgeo. Then, clone the repository using:

.. code-block:: console

   $ git clone https://github.com/<your-username>/torchgeo.git


From there, you can make any changes you want. Once you are satisfied with your changes, you can commit them and push them back to your fork. If you want to make multiple changes, it's best to create separate branches and pull requests for each change:

.. code-block:: console

   $ git checkout main
   $ git branch <descriptive-branch-name>
   $ git checkout <descriptive-branch-name>
   $ git add <files-you-changed...>
   $ git commit -m "descriptive commit message"
   $ git push


For changes to Python code, you'll need to ensure that your code is :ref:`well-tested <tests>` and all :ref:`linters <linters>` pass. When you're ready, you can `open a pull request on GitHub <https://github.com/microsoft/torchgeo/compare>`_. All pull requests should be made against the ``main`` branch. If it's a bug fix, we will backport it to a release branch for you.

Licensing
---------

TorchGeo is licensed under the MIT License. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

If your pull request adds any new files containing code, including ``*.py`` and ``*.ipynb`` files, you'll need to add the following comment to the top of the file:

.. code-block:: python

   # Copyright (c) Microsoft Corporation. All rights reserved.
   # Licensed under the MIT License.


.. _tests:

Tests
-----

TorchGeo uses `GitHub Actions <https://docs.github.com/en/actions>`_ for Continuous Integration. We run a suite of unit tests on every commit to ensure that pull requests don't break anything. If you submit a pull request that adds or modifies any Python code, we require unit tests for that code before the pull request can be merged.

For example, if you add a new dataset in ``torchgeo/datasets/foo.py``, you'll need to create corresponding unit tests in ``tests/datasets/test_foo.py``. The easiest way to do this is to find unit tests for similar datasets and modify them for your dataset. These tests can then be run with `pytest <https://docs.pytest.org/>`_:

.. code-block:: console

   $ pytest --cov=torchgeo/datasets --cov-report=term-missing tests/datasets/test_foo.py
   ========================= test session starts =========================
   platform darwin -- Python 3.8.11, pytest-6.2.4, py-1.9.0, pluggy-0.13.0
   rootdir: ~/torchgeo, configfile: pyproject.toml
   plugins: mock-1.11.1, anyio-3.2.1, cov-2.8.1, nbmake-0.5
   collected 7 items

   tests/datasets/test_foo.py .......                              [100%]

   ---------- coverage: platform darwin, python 3.8.11-final-0 -----------
   Name                                      Stmts   Miss  Cover   Missing
   -----------------------------------------------------------------------
   torchgeo/datasets/__init__.py                26      0   100%
   torchgeo/datasets/foo.py                    177     62    65%   376-403, 429-496, 504-509
   ...
   -----------------------------------------------------------------------
   TOTAL                                      1709    920    46%

   ========================== 7 passed in 6.20s ==========================


From this output, you can see that all tests pass, but many lines of code in ``torchgeo/datasets/foo.py`` are not being tested, including 376--403, 429--496, etc. In order for this pull request to be merged, additional tests will need to be added until there is 100% test coverage.

These tests require `pytest <https://docs.pytest.org/>`_ and `pytest-cov <https://pytest-cov.readthedocs.io/>`_ to be installed.

.. note:: If you add a new dataset, the tests will require some form of data to run. This data should be stored in ``tests/data/<dataset>``. Please don't include real data, as this may violate the license the data is distributed under, and can involve very large file sizes. Instead, create fake data examples using the instructions found `here <https://github.com/microsoft/torchgeo/blob/main/tests/data/README.md>`__.

.. _linters:

Linters
-------

In order to remain `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_ compliant and maintain a high-quality codebase, we use several linting tools:

* `black <https://black.readthedocs.io/>`_ for code formatting
* `isort <https://pycqa.github.io/isort/>`_ for import ordering
* `flake8 <https://flake8.pycqa.org/>`_ for code formatting
* `pydocstyle <https://www.pydocstyle.org/>`_ for docstrings
* `pyupgrade <https://github.com/asottile/pyupgrade>`_ for code formatting
* `mypy <https://mypy.readthedocs.io/>`_ for static type analysis

All of these tools should be used from the root of the project to ensure that our configuration files are found. Black, isort, and pyupgrade are relatively easy to use, and will automatically format your code for you:

.. code-block:: console

   $ black .
   $ isort .
   $ pyupgrade --py37-plus $(find . -name "*.py")


Flake8, pydocstyle, and mypy won't format your code for you, but they will warn you about potential issues with your code or docstrings:

.. code-block:: console

   $ flake8
   $ pydocstyle
   $ mypy .


If you've never used mypy before or aren't familiar with `Python type hints <https://docs.python.org/3/library/typing.html>`_, this check can be particularly daunting. Don't hesitate to ask for help with resolving any of these warnings on your pull request.

You can also use `git pre-commit hooks <https://pre-commit.com/>`_ to automatically run these checks before each commit. pre-commit is a tool that automatically runs linters locally, so that you don't have to remember to run them manually and then have your code flagged by CI. You can setup pre-commit with:

.. code-block:: console

   $ pip install pre-commit
   $ pre-commit install
   $ pre-commit run --all-files


Now, every time you run ``git commit``, pre-commit will run and let you know if any of the files that you changed fail the linters. If pre-commit passes then your code should be ready (style-wise) for a pull request. Note that you will need to run ``pre-commit run --all-files`` if any of the hooks in ``.pre-commit-config.yaml`` change, see `here <https://pre-commit.com/#4-optional-run-against-all-the-files>`__.

Documentation
-------------

All of our documentation is hosted on `Read the Docs <https://readthedocs.org/>`_. If you make non-trivial changes to the documentation, it helps to build the documentation yourself locally. To do this, make sure the dependencies are installed:

.. code-block:: console

   $ pip install .[docs]
   $ cd docs
   $ pip install -r requirements.txt


Then run the following commands:

.. code-block:: console

   $ make clean
   $ make html


The resulting HTML files can be found in ``_build/html``. Open ``index.html`` in your browser to navigate the project documentation. If you fix something, make sure to run ``make clean`` before running ``make html`` or Sphinx won't rebuild all of the documentation.

Tutorials
---------

TorchGeo has a number of tutorials included in the documentation that can be run in `Google Colab <https://colab.research.google.com/>`_. These Jupyter notebooks are tested before each release to make sure that they still run properly. To test these locally, install `pytest <https://docs.pytest.org/>`_ and `nbmake <https://github.com/treebeardtech/nbmake>`_ and run:

.. code-block:: console

   $ pytest --nbmake docs/tutorials


Datasets
--------

A major component of TorchGeo is the large collection of :mod:`torchgeo.datasets` that have been implemented. Adding new datasets to this list is a great way to contribute to the library. A brief checklist to follow when implementing a new dataset:

* Implement the dataset extending either :class:`~torchgeo.datasets.GeoDataset` or :class:`~torchgeo.datasets.NonGeoDataset`
* Add the dataset definition to ``torchgeo/datasets/__init__.py``
* Add a ``data.py`` script to ``tests/data/<new dataset>/`` that generates test data with the same directory structure/file naming conventions as the new dataset
* Add appropriate tests with 100% test coverage to ``tests/datasets/``
* Add the dataset to ``docs/api/datasets.rst``
* Add the dataset metadata to either ``docs/api/geo_datasets.csv`` or ``docs/api/non_geo_datasets.csv``

A good way to get started is by looking at some of the existing implementations that are most closely related to the dataset that you are implementing (e.g. if you are implementing a semantic segmentation dataset, looking at the LandCover.ai dataset implementation would be a good starting point).
