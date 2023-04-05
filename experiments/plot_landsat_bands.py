#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json

import matplotlib.pyplot as plt

# https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
with open("landsat_non_thermal.json") as f:
    non_thermal = json.load(f)
with open("landsat_thermal.json") as f:
    thermal = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [4, 1]})
fig.subplots_adjust(wspace=0.05)

# Plot bands
for sensor, value1 in non_thermal.items():
    for res, value2 in value1.items():
        ax1.broken_barh(
            value2["xranges"],
            value2["yrange"],
            edgecolor="k",
            facecolors=value2["facecolors"],
            linewidth=0.5,
        )

for sensor, value1 in thermal.items():
    for res, value2 in value1.items():
        ax2.broken_barh(
            value2["xranges"],
            value2["yrange"],
            edgecolor="k",
            facecolors=value2["facecolors"],
            linewidth=0.5,
        )

# Labels
fig.supxlabel("Wavelength (μm)")

ax1.set_ylabel("Satellite (Sensor)")
ax1.set_yticks([62.5, 46.5, 30.5, 8.5])
ax1.set_yticklabels(
    [
        "Landsat 1–5\n(MSS)",
        "Landsat 4–5\n(TM)",
        "Landsat 7\n(ETM+)",
        "Landsat 8–9\n(OLI+TIRS)",
    ]
)
ax1.set_ylim(-5, 70)
ax1.spines.right.set_visible(False)
ax1.spines.top.set_visible(False)

ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_ylabel("Resolution (m)")
ax2.set_yticks([62.5, 52.5, 46.5, 36.5, 30.5, 24.5, 14.5, 8.5, 2.5])
ax2.set_yticklabels([60, 120, 30, 60, 30, 15, 100, 30, 15])
ax2.set_ylim(-5, 70)
ax2.spines.left.set_visible(False)
ax2.spines.top.set_visible(False)

# Draw axis break symbol
d = 2
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot(1, 0, transform=ax1.transAxes, **kwargs)
ax2.plot(0, 0, transform=ax2.transAxes, **kwargs)

plt.tight_layout()
plt.show()
