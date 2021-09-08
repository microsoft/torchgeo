# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

import pytorch_sphinx_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))

import torchgeo  # noqa: E402

# -- Project information -----------------------------------------------------

project = "torchgeo"
copyright = "2021, Microsoft Corporation"
author = "Adam J. Stewart"
version = ".".join(torchgeo.__version__.split(".")[:2])
release = torchgeo.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# Sphinx 3.0+ required for:
# autodoc_typehints = "description"
needs_sphinx = "3.0"

nitpicky = True
nitpick_ignore = [
    # https://github.com/sphinx-doc/sphinx/issues/8127
    ("py:class", ".."),
    # TODO: can't figure out why this isn't found
    ("py:class", "LightningDataModule"),
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "pytorch_project": "docs",
    "navigation_with_keys": True,
    "analytics_id": "UA-117752657-2",
}

html_favicon = os.path.join("..", "logo", "favicon.ico")

# -- Extension configuration -------------------------------------------------

# sphinx.ext.autodoc
autodoc_default_options = {
    "members": True,
    "special-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytorch-lightning": ("https://pytorch-lightning.readthedocs.io/en/latest/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest/", None),
    "rtree": ("https://rtree.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# nbsphinx
nbsphinx_execute = "never"
# TODO: branch/tag should change depending on which version of docs you look at
# TODO: width option of image directive is broken, see:
# https://github.com/pytorch/pytorch_sphinx_theme/issues/140
nbsphinx_prolog = """
{% set colab = "https://colab.research.google.com" %}
{% set repo = "microsoft/torchgeo" %}
{% set branch = "main" %}

.. image:: {{ colab }}/assets/colab-badge.svg
   :alt: Open in Colab
   :target: {{ colab }}/github/{{ repo }}/blob/{{ branch }}/docs/{{ env.docname }}
   :width: 200
"""
