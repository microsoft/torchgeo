#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import random
import shutil

import pandas as pd
from PIL import Image

SIZE = 32

random.seed(0)

for csv in glob.iglob('*.csv'):
    captions = pd.read_csv(csv)
    for jpg in captions['filepath']:
        os.makedirs(os.path.dirname(jpg), exist_ok=True)
        width = random.randrange(SIZE)
        height = random.randrange(SIZE)
        img = Image.new('RGB', (width, height))
        img.save(jpg)

for directory in [f'images{i}' for i in range(2, 8)]:
    shutil.make_archive(directory, 'zip', '.', directory)
