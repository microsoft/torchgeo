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
extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", "sphinx.ext.napoleon"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "setup.rst",
    "tests*.rst",
    "torchgeo.rst",
]

nitpicky = True
nitpick_ignore = [
    # https://github.com/sphinx-doc/sphinx/issues/8127
    ("py:class", ".."),
    ("py:class", "torch.utils.data.dataset.Dataset"),
    ("py:class", "torch.nn.modules.module.Module"),
    # https://stackoverflow.com/questions/68186141
    ("py:class", "torchgeo.datasets.cowc.COWC"),
    ("py:class", "torchgeo.datasets.geo.GeoDataset"),
    ("py:class", "torchgeo.datasets.geo.VisionDataset"),
    ("py:class", "torchgeo.datasets.geo.ZipDataset"),
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

# -- Extension configuration -------------------------------------------------

autodoc_default_options = {
    "members": True,
    "special-members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"

autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
