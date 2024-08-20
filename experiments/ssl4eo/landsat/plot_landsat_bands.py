#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Plot Landsat band wavelengths and resolutions."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Match NeurIPS template
plt.rcParams.update(
    {
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'text.usetex': True,
    }
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
)
parser.add_argument('skip', nargs='*', help='sensors to skip', metavar='SENSOR')
parser.add_argument(
    '--fig-height', default=5, type=float, help='height of figure in inches'
)
parser.add_argument('--bar-start', default=1, type=float, help='height of first bar')
parser.add_argument('--bar-height', default=3, type=float, help='height of each bar')
parser.add_argument(
    '--bar-sep', default=3.5, type=float, help='separation between bars'
)
parser.add_argument(
    '--bar-jump', default=2.6, type=float, help='additional height for narrow bars'
)
parser.add_argument(
    '--sensor-sep', default=2, type=float, help='separation between sensors'
)
args = parser.parse_args()

# https://www.usgs.gov/landsat-missions/landsat-satellite-missions
df = pd.read_csv('band_data.csv', skip_blank_lines=True)
df = df.iloc[::-1]

fig, ax = plt.subplots(figsize=(5.5, args.fig_height))
ax1, ax2 = fig.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

sensor_names: list[str] = []
sensor_ylocs: list[float] = []
res_names: list[int] = []
res_ylocs: list[float] = []
bar_min = args.bar_start

# For each satellite/sensor
for (satellite, sensor), group1 in df.groupby(['Satellite', 'Sensor'], sort=False):
    if sensor in args.skip:
        continue

    sensor_names.append(f'{satellite}\n({sensor})')
    sensor_yloc = 0.0
    res_count = 0

    # For each resolution
    for res, group2 in group1.groupby('Resolution (m)'):
        res_names.append(res)
        res_ylocs.append(bar_min)

        if len(group2) > res_count:
            sensor_yloc = bar_min
            res_count = len(group2)

        # For each band
        for i in range(group2.shape[0]):
            row = group2.iloc[i]
            wavelength_start = row['Wavelength Start (μm)']
            wavelength_width = row['Wavelength Width']
            color = row['Color']
            band = row['Band']

            # We've split the plot into two parts as the thermal bands are > 10μm
            # while the other bands are < 3μm
            y = bar_min + args.bar_height / 2
            if wavelength_width < 0.05:
                y += args.bar_jump

            if wavelength_start < 10:
                ax1.broken_barh(
                    [[wavelength_start, wavelength_width]],
                    [bar_min, args.bar_height],
                    edgecolor='k',
                    facecolors=color,
                    linewidth=0.5,
                    alpha=0.8,
                )
                ax1.text(
                    wavelength_start + (wavelength_width / 2),
                    y,
                    band,
                    horizontalalignment='center',
                    verticalalignment='center_baseline',
                )
            else:
                ax2.broken_barh(
                    [[wavelength_start, wavelength_width]],
                    [bar_min, args.bar_height],
                    edgecolor='k',
                    facecolors=color,
                    linewidth=0.5,
                    alpha=0.8,
                )
                ax2.text(
                    wavelength_start + (wavelength_width / 2),
                    y,
                    band,
                    horizontalalignment='center',
                    verticalalignment='center_baseline',
                )
        bar_min += args.bar_sep
    bar_min += args.sensor_sep

    sensor_ylocs.append(sensor_yloc)

# Labels
ax.set_xlabel(r'Wavelength (\textmu m)')
ax.set_xticks([0], labels=['0'])
ax.set_yticks([0], labels=['0'])
ax.tick_params(colors='w')
ax.spines[['bottom', 'left', 'top', 'right']].set_visible(False)

ax1.set_yticks(np.array(sensor_ylocs) + args.bar_height / 2)
ax1.set_yticklabels(sensor_names)
ax1.set_ylim(0, max(res_ylocs) + args.bar_height + args.bar_start)
ax1.spines[['left', 'top', 'right']].set_visible(False)
ax1.tick_params(axis='both', which='both', left=False)

ax2.set_xlim(10.1, 12.8)
ax2.set_ylim(0, max(res_ylocs) + args.bar_height + args.bar_start)
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()
ax2.set_ylabel('Resolution (m)')
ax2.set_yticks(np.array(res_ylocs) + args.bar_height / 2)
ax2.set_yticklabels(res_names)
ax2.spines[['left', 'top']].set_visible(False)

# Draw axis break symbol
d = 1.5
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle='none',
    color='k',
    mec='k',
    mew=0.75,
    clip_on=False,
)
ax1.plot(1, 0, transform=ax1.transAxes, **kwargs)
ax2.plot(0, 0, transform=ax2.transAxes, **kwargs)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.show()
