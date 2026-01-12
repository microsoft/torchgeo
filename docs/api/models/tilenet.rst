TileNet
=======

.. automodule:: torchgeo.models.tilenet

Overview
--------

TileNet is the convolutional encoder introduced in
*Tile2Vec: Unsupervised Representation Learning for Spatial Data*
(Jean et al., 2018).

This implementation exactly matches the original Tile2Vec encoder
architecture, including the additional convolutional branch in each
residual block.

The model produces a fixed-length embedding for an input image tile and
is typically used for self-supervised or transfer learning tasks.

Model API
---------

.. autofunction:: torchgeo.models.tilenet

Weights
-------

.. autoclass:: torchgeo.models.tilenet.TileNet_Weights
   :members: NAIP_ALL_TILE2VEC
