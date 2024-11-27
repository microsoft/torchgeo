#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df1 = pd.read_csv('original-benchmark-results.csv')
df2 = pd.read_csv('warped-benchmark-results.csv')

mean1 = df1.groupby('sampler').mean()
mean2 = df2.groupby('sampler').mean()

cached1 = (
    df1[(df1['cached']) & (df1['sampler'] != 'resnet18')].groupby('sampler').mean()
)
cached2 = (
    df2[(df2['cached']) & (df2['sampler'] != 'resnet18')].groupby('sampler').mean()
)
not_cached1 = (
    df1[(~df1['cached']) & (df1['sampler'] != 'resnet18')].groupby('sampler').mean()
)
not_cached2 = (
    df2[(~df2['cached']) & (df2['sampler'] != 'resnet18')].groupby('sampler').mean()
)

print('cached, original\n', cached1)
print('cached, warped\n', cached2)
print('not cached, original\n', not_cached1)
print('not cached, warped\n', not_cached2)

cmap = sns.color_palette()

labels = ['GridGeoSampler', 'RandomBatchGeoSampler', 'RandomGeoSampler']

fig, ax = plt.subplots()
x = np.arange(3)
width = 0.2

rects1 = ax.bar(
    x - width * 3 / 2,
    not_cached1['rate'],
    width,
    label='Raw Data, Not Cached',
    color=cmap[0],
)
rects2 = ax.bar(
    x - width * 1 / 2,
    not_cached2['rate'],
    width,
    label='Preprocessed, Not Cached',
    color=cmap[1],
)
rects2 = ax.bar(
    x + width * 1 / 2, cached1['rate'], width, label='Raw Data, Cached', color=cmap[2]
)
rects3 = ax.bar(
    x + width * 3 / 2,
    cached2['rate'],
    width,
    label='Preprocessed, Cached',
    color=cmap[3],
)

ax.set_ylabel('sampling rate (patches/sec)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.tick_params(axis='x', labelrotation=10)
ax.legend(fontsize='large')

plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.tight_layout()
plt.show()
