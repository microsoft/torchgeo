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

# import pytorch_sphinx_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

import torchgeo

# -- Project information -----------------------------------------------------

project = 'torchgeo'
copyright = '2021, Microsoft Corporation'
author = torchgeo.__author__
version = '.'.join(torchgeo.__version__.split('.')[:2])
release = torchgeo.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# Sphinx 4.0+ required for autodoc_typehints_description_traget
needs_sphinx = '4.0'

nitpicky = True
nitpick_ignore = [
    # Undocumented classes
    ('py:class', 'fiona.model.Feature'),
    ('py:class', 'kornia.augmentation._2d.intensity.base.IntensityAugmentationBase2D'),
    ('py:class', 'kornia.augmentation.base._AugmentationBase'),
    ('py:class', 'lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig'),
    ('py:class', 'segmentation_models_pytorch.base.model.SegmentationModel'),
    ('py:class', 'timm.models.resnet.ResNet'),
    ('py:class', 'timm.models.vision_transformer.VisionTransformer'),
    ('py:class', 'torch.optim.lr_scheduler.LRScheduler'),
    ('py:class', 'torchvision.models._api.WeightsEnum'),
    ('py:class', 'torchvision.models.resnet.ResNet'),
    ('py:class', 'torchvision.models.swin_transformer.SwinTransformer'),
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# conf.py
html_theme = "furo"

html_theme_options = {
    'sidebar_hide_name': True, 'navigation_with_keys': True,
}

html_favicon = os.path.join('..', 'logo', 'favicon.ico')

html_static_path = ['_static']
html_css_files = ['badge-height.css', 'notebook-prompt.css', 'table-scroll.css']

# -- Extension configuration -------------------------------------------------

# sphinx.ext.autodoc
autodoc_default_options = {
    'members': True,
    'special-members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# sphinx.ext.intersphinx
intersphinx_mapping = {
    'kornia': ('https://kornia.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'pyvista': ('https://docs.pyvista.org/version/stable/', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'rtree': ('https://rtree.readthedocs.io/en/stable/', None),
    'segmentation_models_pytorch': ('https://smp.readthedocs.io/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'timm': ('https://huggingface.co/docs/timm/main/en/', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'torchvision': ('https://pytorch.org/vision/stable', None),
}

# nbsphinx
nbsphinx_execute = 'never'
with open(os.path.join('tutorials', 'prolog.rst.jinja')) as f:
    nbsphinx_prolog = f.read()

# Disables requirejs in nbsphinx to enable compatibility with the pytorch_sphinx_theme
# See more information here https://github.com/spatialaudio/nbsphinx/issues/599
# NOTE: This will likely break nbsphinx widgets
nbsphinx_requirejs_path = ''
