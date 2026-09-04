This directory contains generators and source metadata for fake data used to test TorchGeo. The session-scoped `test_data` pytest fixture runs each requested dataset's `data.py` scripts once in an isolated temporary directory. It returns the generated root, shared by the dataset, datamodule, and task tests in that worker. Generated files are not checked into Git.

Request a generated dataset directory explicitly:

```python
from collections.abc import Callable


def test_dataset(test_data: Callable[[str], str]) -> None:
    dataset = MyDataset(root=test_data('my_dataset'))
```

Each `data.py` runs in its own directory, with source metadata copied alongside it. Use paths relative to that directory, or its absolute working directory for embedded local download URLs. The subprocess keeps random seeds and working-directory changes independent of the tests. Scripts must generate every binary fixture they use, including download archives, and should use the same optional dependencies as their datasets. Text metadata such as split lists and annotations can remain as generator inputs.

Task tests use `test_config` to resolve the `tests/data/` paths in `tests/conf/*.yaml` against these generated directories. To regenerate files manually, run `data.py` from its directory with the repository root on `PYTHONPATH`; do not add its outputs to Git.

Depending on the type of dataset, fake data can be created in multiple ways:

## GeoDataset

GeoDataset data can be created like so. We first open an existing data example and use it to copy the driver/CRS/transform to the fake data.

### Raster data

```python
import os

import numpy as np
import rasterio as rio

ROOT = 'data/landsat8'
FILENAME = 'LC08_L2SP_023032_20210622_20210629_02_T1_SR_B1.TIF'
SIZE = 64

with rio.open(os.path.join(ROOT, FILENAME), 'r') as src:
    dtype = src.profile['dtype']
    Z = np.random.randint(np.iinfo(dtype).max, size=(SIZE, SIZE), dtype=dtype)
    with rio.open(FILENAME, 'w', **src.profile) as dst:
        for i in dst.indexes:
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

ROOT = 'data/cbf'
FILENAME = 'Ontario.geojson'

rec = {
    'type': 'Feature',
    'id': '0',
    'properties': OrderedDict(),
    'geometry': {
        'type': 'Polygon',
        'coordinates': [[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]],
    },
}
with fiona.open(os.path.join(ROOT, FILENAME), 'r') as src:
    src.meta['schema']['properties'] = OrderedDict()
    with fiona.open(FILENAME, 'w', **src.meta) as dst:
        dst.write(rec)
```

## NonGeoDataset

NonGeoDataset data can be created like so.

### RGB images

```python
import numpy as np
from PIL import Image

DTYPE = np.uint8
SIZE = 64

arr = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE, 3), dtype=DTYPE)
img = Image.fromarray(arr)
img.save('01.png')
```

### Grayscale images

```python
import numpy as np
from PIL import Image

DTYPE = np.uint8
SIZE = 64

arr = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE), dtype=DTYPE)
img = Image.fromarray(arr)
img.save('02.jpg')
```

### Audio wav files

```python
import numpy as np
from scipy.io import wavfile

audio = np.random.randn(1).astype(np.float32)
wavfile.write('01.wav', rate=22050, data=audio)
```

### HDF5 datasets

```python
import h5py
import numpy as np

DTYPE = np.uint8
SIZE = 64
NUM_CLASSES = 10

images = np.random.randint(np.iinfo(DTYPE).max, size=(SIZE, SIZE, 3), dtype=DTYPE)
masks = np.random.randint(NUM_CLASSES, size=(SIZE, SIZE), dtype=DTYPE)
with h5py.File('data.hdf5', 'w') as f:
    f.create_dataset('images', data=images)
    f.create_dataset('masks', data=masks)
```

### LAS Point Cloud files

```python
import laspy

num_points = 4

las = laspy.read('0.las')
las.points = las.points[:num_points]

points = np.random.randint(low=0, high=100, size=(num_points,), dtype=las.x.dtype)
las.x = points
las.y = points
las.z = points

if hasattr(las, 'red'):
    colors = np.random.randint(low=0, high=10, size=(num_points,), dtype=las.red.dtype)
    las.red = colors
    las.green = colors
    las.blue = colors

las.write('0.las')
```
