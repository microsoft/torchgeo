#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('warped-benchmark-results.csv')

random_cached = df[(df['sampler'] == 'RandomGeoSampler') & (df['cached'])]
random_batch_cached = df[(df['sampler'] == 'RandomBatchGeoSampler') & (df['cached'])]
grid_cached = df[(df['sampler'] == 'GridGeoSampler') & (df['cached'])]
other = [
    ('RandomGeoSampler', random_cached),
    ('RandomBatchGeoSampler', random_batch_cached),
    ('GridGeoSampler', grid_cached),
]

cmap = sns.color_palette()

ax = plt.gca()

for i, (label, df) in enumerate(other):
    df = df.groupby('batch_size')
    ax.plot(df.mean().index, df.mean()['rate'], color=cmap[i], label=label)
    ax.fill_between(
        df.mean().index, df.min()['rate'], df.max()['rate'], color=cmap[i], alpha=0.2
    )


ax.set_xscale('log')
ax.set_xticks([16, 32, 64, 128, 256])
ax.set_xticklabels(['16', '32', '64', '128', '256'], fontsize=12)
ax.set_xlabel('batch size', fontsize=12)
ax.set_ylabel('sampling rate (patches/sec)', fontsize=12)
ax.legend(loc='center right', fontsize='large')

plt.gca().spines.right.set_visible(False)
plt.gca().spines.top.set_visible(False)
plt.tight_layout()
plt.show()
