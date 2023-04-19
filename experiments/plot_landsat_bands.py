#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import pandas as pd

# https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
df = pd.read_csv("band_data.csv")

bar_height = 5
# This dictionary maps a sensor and resolution to a y location on the plot
bar_to_height_map = {
    ("tm", 120): 50,
    ("etm", 60): 34,
    ("tirs", 100): 12,
    ("mss", 60): 60,
    ("tm", 30): 44,
    ("etm", 30): 28,
    ("etm", 15): 22,
    ("oli", 30): 6,
    ("oli", 15): 0,
}

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [4, 1]})
fig.subplots_adjust(wspace=0.05)

for sensor_name, group1 in df.groupby("Sensor name"):
    for resolution, group2 in group1.groupby("Resolution (m)"):
        # Plot one row of data
        y_position = bar_to_height_map[(sensor_name, resolution)]

        for i in range(group2.shape[0]):
            row = group2.iloc[i]
            wavelength_start = row["Wavelength start (μm)"]
            wavelength_width = row["Wavelength width"]
            color = row["Color"]
            band_number = row["Band number"]

            # We've split the plot into two parts as the thermal bands are > 10μm
            # while the other bands are < 3μm
            if wavelength_start < 10:
                ax1.broken_barh(
                    [[wavelength_start, wavelength_width]],
                    [y_position, bar_height],
                    edgecolor="k",
                    facecolors=color,
                    linewidth=0.5,
                    alpha=0.8,
                )
                if wavelength_width < 0.05:
                    y = y_position + 6.5
                else:
                    y = y_position + 2.25
                ax1.text(
                    wavelength_start + (wavelength_width / 2),
                    y,
                    str(band_number),
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                ax2.broken_barh(
                    [[wavelength_start, wavelength_width]],
                    [y_position, bar_height],
                    edgecolor="k",
                    facecolors=color,
                    linewidth=0.5,
                    alpha=0.8,
                )
                ax2.text(
                    wavelength_start + (wavelength_width / 2),
                    y_position + 2.25,
                    str(band_number),
                    horizontalalignment="center",
                    verticalalignment="center",
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
