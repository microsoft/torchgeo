Copernicus-Bench
================

Copernicus-Bench is a comprehensive evaluation benchmark with 15 downstream tasks hierarchically organized across preprocessing (e.g., cloud removal), base applications (e.g., land cover classification), and specialized applications (e.g., air quality estimation). This benchmark enables systematic assessment of foundation model performances across various Sentinel missions on different levels of practical applications.

.. csv-table:: C = classification,  R = regression, S = semantic segmentation, T = time series, CD = change detection, E = embedding
   :widths: 5 15 7 15 20 12 11 12 15 13
   :header-rows: 1
   :align: center
   :file: copernicus_bench.csv

.. currentmodule:: torchgeo.datasets
.. autoclass:: CopernicusBench
.. autoclass:: CopernicusBenchBase
.. autoclass:: CopernicusBenchCloudS2
.. autoclass:: CopernicusBenchCloudS3
.. autoclass:: CopernicusBenchEuroSATS1
.. autoclass:: CopernicusBenchEuroSATS2
.. autoclass:: CopernicusBenchBigEarthNetS1
.. autoclass:: CopernicusBenchBigEarthNetS2
.. autoclass:: CopernicusBenchLC100ClsS3
.. autoclass:: CopernicusBenchLC100SegS3
.. autoclass:: CopernicusBenchDFC2020S1
.. autoclass:: CopernicusBenchDFC2020S2
.. autoclass:: CopernicusBenchFloodS1
.. autoclass:: CopernicusBenchLCZS2
.. autoclass:: CopernicusBenchBiomassS3
.. autoclass:: CopernicusBenchAQNO2S5P
.. autoclass:: CopernicusBenchAQO3S5P
