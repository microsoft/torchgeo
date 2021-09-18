This directory contains fake data used to test torchgeo. Depending on the type of dataset, fake data can be created in multiple ways:

## GeoDataset

GeoDataset data can be created like so. We first open an existing data example and use it to copy the driver/CRS/transform to the fake data.

### Raster data

```python
import os

import numpy as np
import rasterio

ROOT = "data/landsat8"
FILENAME = "LC08_L2SP_023032_20210622_20210629_02_T1_SR_B1.TIF"

src = rasterio.open(os.path.join(ROOT, FILENAME))
Z = np.arange(4, dtype=src.read().dtype).reshape(2, 2)
dst = rasterio.open(FILENAME, "w", driver=src.driver, height=Z.shape[0], width=Z.shape[1], count=src.count, dtype=Z.dtype, crs=src.crs, transform=src.transform)
for i in range(1, dst.count + 1):
    dst.write(Z, i)
```
Optionally, if the dataset has a colormap, this can be copied like so:
```python
cmap = src.colormap(1)
dst.write_colormap(1, cmap)
```

### Vector data

```python
import os
from collections import OrderedDict

import fiona

ROOT = "data/cbf"
FILENAME = "Ontario.geojson"

src = fiona.open(os.path.join(ROOT, FILENAME))
dst = fiona.open(FILENAME, "w", **src.meta)
rec = {"type": "Feature", "id": "0", "properties": OrderedDict(), "geometry": {"type": "Polygon", "coordinates": [[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]]}}
dst.write(rec)
```

## VisionDataset

VisionDataset data can be created like so.

### RGB images

```python
from PIL import Image

img = Image.new("RGB", (1, 1))
img.save("01.png")
```

### Grayscale images

```python
from PIL import Image

img = Image.new("L", (1, 1))
img.save("02.jpg")
```

### Audio wav files

```python
import numpy as np
from scipy.io import wavfile

audio = np.random.randn(1,).astype(np.float32)
wavfile.write("01.wav", rate=22050, data=audio)
```
