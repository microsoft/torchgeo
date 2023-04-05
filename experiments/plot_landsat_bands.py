#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt

# https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
non_thermal = {
    "mss": {
        60: {
            "xranges": [(0.5, 0.1), (0.6, 0.1), (0.7, 0.1), (0.8, 0.3)],
            "yrange": [60, 5],
            "facecolors": ["tab:green", "tab:red", "tab:pink", "tab:orange"],
            "labels": [1, 2, 3, 4],
        }
    },
    "tm": {
        30: {
            "xranges": [
                (0.45, 0.07),
                (0.52, 0.08),
                (0.63, 0.06),
                (0.76, 0.14),
                (1.55, 0.2),
                (2.08, 0.27),
            ],
            "yrange": [44, 5],
            "facecolors": [
                "tab:blue",
                "tab:green",
                "tab:red",
                "tab:pink",
                "tab:orange",
                "tab:gray",
            ],
            "labels": [1, 2, 3, 4, 5, 7],
        }
    },
    "etm": {
        30: {
            "xranges": [
                (0.45, 0.07),
                (0.52, 0.08),
                (0.63, 0.06),
                (0.77, 0.13),
                (1.55, 0.2),
                (2.09, 0.26),
            ],
            "yrange": [28, 5],
            "facecolors": [
                "tab:blue",
                "tab:green",
                "tab:red",
                "tab:pink",
                "tab:orange",
                "tab:gray",
            ],
            "labels": [1, 2, 3, 4, 5, 7],
        },
        15: {
            "xranges": [(0.52, 0.38)],
            "yrange": [22, 5],
            "facecolors": ["tab:olive"],
            "labels": [8],
        },
    },
    "oli": {
        30: {
            "xranges": [
                (0.43, 0.02),
                (0.45, 0.06),
                (0.53, 0.06),
                (0.64, 0.03),
                (0.85, 0.03),
                (1.57, 0.08),
                (2.11, 0.18),
                (1.36, 0.02),
            ],
            "yrange": [6, 5],
            "facecolors": [
                "tab:cyan",
                "tab:blue",
                "tab:green",
                "tab:red",
                "tab:pink",
                "tab:orange",
                "tab:gray",
                "tab:purple",
            ],
            "labels": [1, 2, 3, 4, 5, 6, 7, 9],
        },
        15: {
            "xranges": [(0.50, 0.18)],
            "yrange": [0, 5],
            "facecolors": ["tab:olive"],
            "labels": [8],
        },
    },
}

thermal = {
    "tm": {
        120: {
            "xranges": [(10.40, 2.1)],
            "yrange": [50, 5],
            "facecolors": ["tab:brown"],
            "labels": [6],
        }
    },
    "etm": {
        60: {
            "xranges": [(10.40, 2.1)],
            "yrange": [34, 5],
            "facecolors": ["tab:brown"],
            "labels": [6],
        }
    },
    "tirs": {
        100: {
            "xranges": [(10.60, 0.59), (11.50, 1.01)],
            "yrange": [12, 5],
            "facecolors": ["tab:brown", "tab:brown"],
            "labels": [10, 11],
        }
    },
}


fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [4, 1]})
fig.subplots_adjust(wspace=0.05)

# Plot bands
for sensor, value in non_thermal.items():
    for res, value in value.items():
        ax1.broken_barh(
            value["xranges"],
            value["yrange"],
            edgecolor="k",
            facecolors=value["facecolors"],
            linewidth=0.5,
        )

for sensor, value in thermal.items():
    for res, value in value.items():
        ax2.broken_barh(
            value["xranges"],
            value["yrange"],
            edgecolor="k",
            facecolors=value["facecolors"],
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
