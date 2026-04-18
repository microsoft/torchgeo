.. _contributing:

Contributing
============

The TorchGeo project welcomes contributions and suggestions! If you think you've found a bug or would like to suggest a new feature, you can `open an issue on GitHub <https://github.com/torchgeo/torchgeo/issues/new/choose>`_. TorchGeo is an open-source community-supported project, so we try to address issues in order of severity or impact. If you feel confident, the fastest way to make changes to TorchGeo is to submit a pull request. This guide explains everything you need to know about contributing to TorchGeo.

.. note:: TorchGeo is a library for geospatial datasets, transforms, and models. If you would like to add a new transform or model that doesn't involve geospatial data or isn't specific to the remote sensing domain, you're better off adding it to a general purpose computer vision library like `torchvision <https://github.com/pytorch/vision>`_ or `Kornia <https://github.com/kornia/kornia>`_.


Git
---

All development is done on GitHub. If you would like to submit a pull request, you'll first want to fork https://github.com/torchgeo/torchgeo. Then, clone the repository using:

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


For changes to Python code, you'll need to ensure that your code is :ref:`well-tested <tests>` and all :ref:`linters <linters>` pass. When you're ready, you can `open a pull request on GitHub <https://github.com/torchgeo/torchgeo/compare>`_. All pull requests should be made against the ``main`` branch. If it's a bug fix, we will backport it to a release branch for you.

Licensing
---------

TorchGeo is licensed under the MIT License. If your pull request adds any new files containing code, including ``*.py`` and ``*.ipynb`` files, you'll need to add the following comment to the top of the file:

.. code-block:: python

   # Copyright (c) TorchGeo Contributors. All rights reserved.
   # Licensed under the MIT License.


.. _tests:

Tests
-----

TorchGeo uses `GitHub Actions <https://docs.github.com/en/actions>`_ for Continuous Integration. We run a suite of unit tests on every commit to ensure that pull requests don't break anything. If you submit a pull request that adds or modifies any Python code, we require unit tests for that code before the pull request can be merged.

For example, if you add a new dataset in ``torchgeo/datasets/foo.py``, you'll need to create corresponding unit tests in ``tests/datasets/test_foo.py``. The easiest way to do this is to find unit tests for similar datasets and modify them for your dataset. These tests can then be run with `pytest <https://docs.pytest.org/en/stable/>`_:

.. code-block:: console

   $ pytest --cov=torchgeo.datasets tests/datasets/test_foo.py
   ========================= test session starts =========================
   platform darwin -- Python 3.10.11, pytest-6.2.4, py-1.9.0, pluggy-0.13.0
   rootdir: ~/torchgeo, configfile: pyproject.toml
   plugins: mock-1.11.1, anyio-3.2.1, cov-2.8.1, nbmake-0.5
   collected 7 items

   tests/datasets/test_foo.py .......                              [100%]

   --------- coverage: platform darwin, python 3.10.11-final-0 -----------
   Name                                      Stmts   Miss  Cover   Missing
   -----------------------------------------------------------------------
   torchgeo/datasets/foo.py                    177     62    65%   376-403, 429-496, 504-509
   -----------------------------------------------------------------------
   TOTAL                                       177     62    65%

   ========================== 7 passed in 6.20s ==========================


From this output, you can see that all tests pass, but many lines of code in ``torchgeo/datasets/foo.py`` are not being tested, including 376--403, 429--496, etc. In order for this pull request to be merged, additional tests will need to be added until there is 100% test coverage.

These tests require `pytest <https://docs.pytest.org/en/stable/>`_ and `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ to be installed.

.. note:: If you add a new dataset, the tests will require some form of data to run. This data should be stored in ``tests/data/<dataset>``. Please don't include real data, as this may violate the license the data is distributed under, and can involve very large file sizes. Instead, create fake data examples using the instructions found `here <https://github.com/torchgeo/torchgeo/blob/main/tests/data/README.md>`__.

.. _linters:

Linters
-------

In order to remain `PEP-8 <https://peps.python.org/pep-0008/>`_ compliant and maintain a high-quality codebase, we use a few linting tools:

* `ruff <https://docs.astral.sh/ruff/>`_ for code formatting
* `ty <https://docs.astral.sh/ty/>`_ for static type analysis
* `prettier <https://prettier.io/docs/>`_ for config file formatting

These tools should be used from the root of the project to ensure that our configuration files are found. Ruff is relatively easy to use, and will automatically fix most issues it encounters:

.. code-block:: console

   $ ruff format
   $ ruff check


Ty won't fix your code for you, but will warn you about potential issues with your code:

.. code-block:: console

   $ ty check


If you've never used ty before or aren't familiar with `Python type hints <https://docs.python.org/3/library/typing.html>`_, this check can be particularly daunting. Don't hesitate to ask for help with resolving any of these warnings on your pull request.

Prettier is a code formatter that helps to ensure consistent code style across a project. It supports various languages.

.. code-block:: console

   $ prettier --write .


You can also use `git pre-commit hooks <https://pre-commit.com/>`_ to automatically run these checks before each commit. pre-commit is a tool that automatically runs linters locally, so that you don't have to remember to run them manually and then have your code flagged by CI. You can set up pre-commit with:

.. code-block:: console

   $ pip install pre-commit
   $ pre-commit install
   $ pre-commit run --all-files


Now, every time you run ``git commit``, pre-commit will run and let you know if any of the files that you changed fail the linters. If pre-commit passes then your code should be ready (style-wise) for a pull request. Note that you will need to run ``pre-commit run --all-files`` if any of the hooks in ``.pre-commit-config.yaml`` change, see `here <https://pre-commit.com/#4-optional-run-against-all-the-files>`__.

Documentation
-------------

All of our documentation is hosted on `Read the Docs <https://about.readthedocs.com/>`_. If you make non-trivial changes to the documentation, it helps to build the documentation yourself locally. To do this, make sure the dependencies are installed:

.. code-block:: console

   $ pip install .[docs]
   $ cd docs

Pandoc must also be installed. You can download it from the `official Pandoc website <https://pandoc.org/installing.html>`_ and ensure it is included in your system PATH.

Then run the following commands:

.. code-block:: console

   $ make clean
   $ make html


The resulting HTML files can be found in ``_build/html``. Open ``index.html`` in your browser to navigate the project documentation. If you fix something, make sure to run ``make clean`` before running ``make html`` or Sphinx won't rebuild all of the documentation.

Tutorials
---------

TorchGeo has a number of tutorials included in the documentation that can be run in `Lightning Studios <https://lightning.ai/studios>`_ and `Google Colab <https://colab.research.google.com/>`_. These Jupyter notebooks are tested before each release to make sure that they still run properly. To test these locally, install `pytest <https://docs.pytest.org/en/stable/>`_ and `nbmake <https://github.com/treebeardtech/nbmake>`_ and run:

.. code-block:: console

   $ pytest --nbmake docs/tutorials


Datasets
--------

A major component of TorchGeo is the large collection of :mod:`torchgeo.datasets` that have been implemented. Adding new datasets to this list is a great way to contribute to the library. A brief checklist to follow when implementing a new dataset:

* Implement the dataset extending either :class:`~torchgeo.datasets.GeoDataset` or :class:`~torchgeo.datasets.NonGeoDataset`
* Add the dataset definition to ``torchgeo/datasets/foo.py``, where *foo* is the name of the dataset
* Add an import alias to this dataset in ``torchgeo/datasets/__init__.py``
* Add a ``tests/data/foo/data.py`` script that generates fake test data with the same directory structure/file naming conventions as the real dataset
* Add appropriate tests with 100% test coverage to ``tests/datasets/test_foo.py``
* Create a new ``docs/api/datasets/foo.rst`` file with the dataset documentation (see existing files for the format)
* Add the dataset metadata to one of ``docs/api/datasets/*.csv``

A good way to get started is by looking at some of the existing implementations that are most closely related to the dataset that you are implementing (e.g., if you are implementing a semantic segmentation dataset, looking at the LandCover.ai dataset implementation would be a good starting point).

I/O Benchmarking
----------------

For PRs that may affect GeoDataset sampling speed, you can test the performance impact as follows. On the main branch (before) and on your PR branch (after), run the following commands:

.. code-block:: console

   $ python -m torchgeo fit --config tests/conf/io_raw.yaml
   $ python -m torchgeo fit --config tests/conf/io_preprocessed.yaml

This code will download a small (1 GB) dataset consisting of a single Landsat 9 scene and CDL file. It will then profile the speed at which various samplers work for both raw data (original downloaded files) and preprocessed data (same CRS, res, TAP, COG). The important output to look out for is the total time taken by ``train_dataloader_next`` (RandomGeoSampler) and ``val_next`` (GridGeoSampler). With this, you can create a table on your PR like:

======  ============  ==========  =====================  ===================
 state  raw (random)  raw (grid)  preprocessed (random)  preprocessed (grid)
======  ============  ==========  =====================  ===================
before        17.223      10.974                 15.685               4.6075
 after        17.360      11.032                  9.613               4.6673
======  ============  ==========  =====================  ===================

In this example, we see a 60% speed-up for RandomGeoSampler on preprocessed data. All other numbers are more or less the same across multiple runs.

Related Libraries
-----------------

TorchGeo maintains a list of :ref:`related-libraries`. New GeoML libraries can be added to this list using the following scripts in ``docs/user/metrics``:

1. Add the library, GitHub repository, Codecov, PyPI/CRAN, and Conda information to ``common.py``.
2. (Optional) Run ``python3 update_timeline.py --api-key GITHUB_TOKEN`` to determine the first and last commit.
3. (Optional) Sort all dictionaries in ``common.py`` in order of first commit.
4. Run ``python3 update_github.py --api-key GITHUB_TOKEN`` to update ``github.csv``.
5. Sort ``index`` in ``common.py`` and ``features.csv`` in the same order as ``github.csv``.
6. Run ``python3 update_downloads.py --api-key PEPY_TOKEN`` to update ``downloads.csv``.
7. Document all library features in ``features.csv``.
8. Add a GitHub link to the bottom of ``../alternatives.rst``.
9. Open a pull request.

Libraries must be primarily geospatial machine learning libraries. Furthermore, libraries should have stable releases, recent commit activity, and at least 100 stars on GitHub. If you have any questions about how to determine any features, do not hesitate to ask for assistance on your pull request.
