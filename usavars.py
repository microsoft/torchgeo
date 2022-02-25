import os
import pandas as pd

from torchgeo.datasets import USAVars
ds = USAVars(download=True)

print(ds[0])
