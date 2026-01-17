# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

import torchgeo

# -- Project information -----------------------------------------------------

project = 'torchgeo'
copyright = 'TorchGeo Contributors'
author = torchgeo.__author__
version = '.'.join(torchgeo.__version__.split('.')[:2])
release = torchgeo.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# Sphinx 5.3+ required to allow section titles inside autodoc class docstrings
# https://github.com/sphinx-doc/sphinx/pull/10887
needs_sphinx = '5.3'

nitpicky = True
nitpick_ignore = [
    # Undocumented classes
    ('py:class', 'kornia.augmentation._2d.intensity.base.IntensityAugmentationBase2D'),
    ('py:class', 'kornia.augmentation._3d.geometric.base.GeometricAugmentationBase3D'),
    ('py:class', 'kornia.augmentation.base._AugmentationBase'),
    ('py:class', 'lightning.pytorch.utilities.types.LRSchedulerConfig'),
    ('py:class', 'lightning.pytorch.utilities.types.OptimizerConfig'),
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
html_theme = 'pydata_sphinx_theme'

# Define the version we use for matching in the version switcher.
version_match = os.environ.get('READTHEDOCS_VERSION')
json_url = 'https://torchgeo.readthedocs.io/en/latest/_static/switcher.json'

# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit() or version_match == 'latest':
    # For local development, infer the version to match from the package.
    if 'dev' in release or 'rc' in release:
        version_match = 'dev'
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = '_static/switcher.json'
    else:
        version_match = f'v{release}'
elif version_match == 'stable':
    version_match = f'v{release}'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation: https://pydata-sphinx-theme.readthedocs.io/
html_theme_options = {
    'collapse_navigation': False,
    'show_nav_level': 2,
    'show_toc_level': 2,
    'navigation_depth': 4,
    'navbar_align': 'left',
    'header_links_before_dropdown': 6,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/torchgeo/torchgeo',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'Slack',
            'url': 'https://join.slack.com/t/torchgeo/shared_invite/zt-22rse667m-eqtCeNW0yI000Tl4B~2PIw',
            'icon': 'fa-brands fa-slack',
        },
        {
            'name': 'YouTube',
            'url': 'https://www.youtube.com/@TorchGeo',
            'icon': 'fa-brands fa-youtube',
        },
    ],
    'analytics': {'google_analytics_id': 'UA-209075005-1'},
    'logo': {
        'image_light': os.path.join('..', 'logo', 'logo-color.svg'),
        'image_dark': os.path.join('..', 'logo', 'logo-color.svg'),
    },
    'switcher': {'json_url': json_url, 'version_match': version_match},
    'navbar_start': ['navbar-logo', 'version-switcher'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
}

html_favicon = os.path.join('..', 'logo', 'favicon.ico')

html_static_path = ['_static']
html_css_files = ['custom.css']

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
    'einops': ('https://einops.rocks/', None),
    'kornia': ('https://kornia.readthedocs.io/en/stable/', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'pyproj': ('https://pyproj4.github.io/pyproj/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'segmentation_models_pytorch': ('https://smp.readthedocs.io/en/stable/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'timm': ('https://huggingface.co/docs/timm/main/en/', None),
    'torch': ('https://docs.pytorch.org/docs/stable/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'torchvision': ('https://docs.pytorch.org/vision/stable/', None),
}

# myst-parser
suppress_warnings = ['myst.header']

# nbsphinx
nbsphinx_execute = 'never'
with open(os.path.join('tutorials', 'prolog.rst.jinja')) as f:
    nbsphinx_prolog = f.read()
