"""Plotting."""

import sys

from matplotlib import pyplot as plt

from torchgeo.datasets import CopernicusBench

ds = CopernicusBench('eurosat_s1', 'data/l2_eurosat_s1s2')
x = ds[int(sys.argv[1])]
ds.plot(x, suptitle='EuroSAT-S1')
plt.show()

ds = CopernicusBench('eurosat_s2', 'data/l2_eurosat_s1s2')
x = ds[int(sys.argv[1])]
ds.plot(x, suptitle='EuroSAT-S2')
plt.show()
