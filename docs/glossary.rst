Glossary
========

A collection of common terms used in :mod:`torchgeo` that may be unfamiliar to either:

1. Deep learning researchers who don't know remote sensing
2. Remote sensing researchers who don't know deep learning

.. glossary::

   chip
       Synonym for :term:`patch`. A smaller image sampled from a larger :term:`tile`.

   classification
       A computer vision task that involves predicting the image class for an entire image or a specific bounding box.

   instance segmentation
       A computer vision task that involves predicting labels for each pixel in an image such that each object has a unique label.

   object detection
       A computer vision task that involves predicting bounding boxes around each object in an image.

   patch
       Synonym for :term:`chip`. A smaller image sampled from a larger :term:`tile`.

   regression
       A computer vision task that involves predicting a real valued number based on an image.

   semantic segmentation
       A computer vision task that involves predicting labels for each pixel in an image such that each class has a unique label.

   swath
       A set of :term:`tiles <tile>` along a satellite trajectory.

   tile
       A single image file taken by a remote sensor like a satellite.
