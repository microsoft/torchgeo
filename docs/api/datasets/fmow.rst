.. _FMoW:

FMoW
====

.. currentmodule:: torchgeo.datasets
.. autoclass:: FMoW

Download
--------

The dataset is not downloaded automatically because the RGB distribution is about
200 GB. Install the `AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
and download the required splits from the public fMoW bucket:

.. code-block:: console

   $ aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/train <root>/train --no-sign-request
   $ aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/val <root>/val --no-sign-request

Replace ``<root>`` with the path passed to :class:`FMoW`.
