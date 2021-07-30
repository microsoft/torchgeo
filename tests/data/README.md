This directory contains fake data used to test torchgeo. Depending on the type of dataset, fake data can be created in one of two ways:

## GeoDataset

GeoDataset data can be created like so. We first open an existing data example and use it to copy the driver/CRS/transform to the fake data.

```python
import os

import numpy as np
import rasterio

ROOT = "/mnt/blobfuse/adam-scratch/landsat8"
FILENAME = "LC08_L2SP_023032_20210622_20210629_02_T1_SR_B1.TIF"

src = rasterio.open(os.path.join(ROOT, FILENAME))
Z = np.array([[1, 2], [3, 4]], dtype=src.read().dtype)
dst = rasterio.open(FILENAME, "w", driver=src.driver, height=Z.shape[0], width=Z.shape[1], count=src.count, dtype=Z.dtype, crs=src.crs, transform=src.transform)
for i in range(dst.count):
    dst.write(Z, i)
```
Optionally, if the dataset has a colormap, this can be copied like so:
```python
cmap = src.colormap(1)
dst.write_colormap(1, cmap)
```

If the dataset expects multiple files, you can simply copy and rename the file you created.

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
