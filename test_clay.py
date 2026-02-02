"""Clay."""

from matplotlib import pyplot as plt

from torchgeo.datasets import ClayEmbeddings

paths = 'data/clay/01WCN_20190518_20231021_v001.gpq'
# paths = 'data/clayV1/data_01c1fab1-0004-5b1c-0009-b72e01d3104e_222_0_0.parquet'

ds = ClayEmbeddings(paths)
sample = ds[0]
ds.plot(sample)
plt.show()
