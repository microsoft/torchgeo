#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import bz2
import csv
import glob
import hashlib
import os
import random
import shutil

from PIL import Image

SIZE = 64  # image width/height
STOP = 2  # range of values for labels
PREFIX = 'Detection'
SUFFIX = 'detection'

random.seed(0)

sites = [
    'Toronto_ISPRS',
    'Selwyn_LINZ',
    'Potsdam_ISPRS',
    'Vaihingen_ISPRS',
    'Columbus_CSUAV_AFRL',
    'Utah_AGRC',
]

# Remove old data
for filename in glob.glob('COWC_*'):
    os.remove(filename)
for site in sites:
    if os.path.exists(site):
        shutil.rmtree(site)

i = 1
data_list = {'train': [], 'test': []}
image_md5s = []
for site in sites:
    # Create images
    for split in ['test', 'train', 'train']:
        directory = os.path.join(site, split)
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f'fake_{i:02}.png')

        img = Image.new('RGB', (SIZE, SIZE))
        img.save(filename)

        data_list[split].append((filename, random.randrange(STOP)))

        if split == 'train':
            i += 1

    # Compress images
    filename = f'COWC_{PREFIX}_{site}.tbz'
    bad_filename = shutil.make_archive(filename.replace('.tbz', ''), 'bztar', '.', site)
    os.rename(bad_filename, filename)

    # Compute checksums
    with open(filename, 'rb') as f:
        image_md5s.append(hashlib.md5(f.read()).hexdigest())

label_md5s = []
for split in ['train', 'test']:
    # Create labels
    filename = f'COWC_{split}_list_{SUFFIX}.txt'
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        csvwriter.writerows(data_list[split])

    # Compress labels
    with open(filename, 'rb') as src:
        with bz2.open(filename + '.bz2', 'wb') as dst:
            dst.write(src.read())

    # Compute checksums
    with open(filename + '.bz2', 'rb') as f:
        label_md5s.append(hashlib.md5(f.read()).hexdigest())

md5s = label_md5s + image_md5s
for md5 in md5s:
    print(repr(md5) + ',')
