Glossary
========

A collection of common terms used in :mod:`torchgeo` that may be unfamiliar to either:

1. Deep learning researchers who don't know remote sensing
2. Remote sensing researchers who don't know deep learning

|

.. todo:: We shouldn't need to bold these keys ourselves, opened an issue at https://github.com/pytorch/pytorch_sphinx_theme/issues/133

.. glossary::

   **area of interest (AOI)**
       Synonym for :term:`region of interest (ROI)`. A particular spatial area to focus on.

   **chip**
       Synonym for :term:`patch`. A smaller image sampled from a larger :term:`tile`.

   **classification**
       A computer vision task that involves predicting the image class for an entire image or a specific bounding box.

   **coordinate reference system (CRS)**
       Synonym for :term:`spatial reference system (SRS)`. A system that defines how to locate geographic entities on a :term:`projected <projection>` surface.

   **instance segmentation**
       A computer vision task that involves predicting labels for each pixel in an image such that each object has a unique label.

   **object detection**
       A computer vision task that involves predicting bounding boxes around each object in an image.

   **patch**
       Synonym for :term:`chip`. A smaller image sampled from a larger :term:`tile`.

   **projection**
       A geometric transformation for portraying the surface of a 3D Earth onto a 2D planar image.

   **region of interest (ROI)**
       Synonym for :term:`area of interest (AOI)`. A particular spatial region to focus on.

   **regression**
       A computer vision task that involves predicting a real valued number based on an image.

   **semantic segmentation**
       A computer vision task that involves predicting labels for each pixel in an image such that each class has a unique label.

   **spatial reference system (SRS)**
       Synonym for :term:`coordinate reference system (CRS)`. A system that defines how to locate geographic entities on a :term:`projected <projection>` surface.

   **stitching**
       Combining a collection of :term:`patches <patch>` into a single image. This is the reverse operation of :term:`tiling`.

   **swath**
       A set of :term:`tiles <tile>` along a satellite trajectory.

   **tile**
       A single image file taken by a remote sensor like a satellite.

   **tiling**
       Splitting a :term:`tile` into :term:`patches <patch>`. This is the reverse operation of :term:`stitching`.
