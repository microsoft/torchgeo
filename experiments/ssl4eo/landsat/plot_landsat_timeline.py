#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from datetime import date, timedelta

import matplotlib.pyplot as plt

# Match NeurIPS template
plt.rcParams.update(
    {
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'text.usetex': True,
        'hatch.linewidth': 0.5,
    }
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
)
parser.add_argument('--bar-start', default=1, type=float, help='height of first bar')
parser.add_argument('--bar-height', default=3, type=float, help='height of each bar')
parser.add_argument('--bar-sep', default=2, type=float, help='separation between bars')
args = parser.parse_args()

working: dict[int, list[tuple[date, date]]] = {
    1: [(date(1972, 7, 23), date(1972, 8, 5))],
    2: [(date(1975, 1, 22), date(1982, 2, 25))],
    3: [(date(1978, 3, 5), date(1983, 3, 31))],
    4: [  # approximate dates
        (date(1982, 7, 16), date(1983, 2, 15)),
        (date(1983, 4, 4), date(1986, 1, 15)),
        (date(1987, 7, 1), date(1993, 7, 1)),
    ],
    5: [  # approximate dates
        (date(1984, 3, 1), date(2006, 1, 1)),
        (date(2006, 1, 31), date(2009, 12, 18)),
        (date(2010, 1, 7), date(2011, 11, 18)),
        (date(2012, 2, 16), date(2013, 1, 6)),
    ],
    6: [],  # failed to reach orbit
    7: [(date(1999, 4, 15), date(2003, 5, 31))],
    8: [(date(2013, 2, 11), date.today())],
    9: [(date(2021, 9, 27), date.today())],
}

failing: dict[int, list[tuple[date, date]]] = {
    1: [(date(1972, 8, 5), date(1978, 1, 6))],  # RBV electrical issue
    2: [(date(1982, 2, 25), date(1983, 3, 31))],  # faulty yaw control thruster
    3: [(date(1983, 3, 31), date(1983, 9, 7))],  # placed on standby
    4: [  # approximate dates
        (date(1983, 2, 15), date(1983, 4, 4)),  # lost half its solar power
        (date(1986, 1, 15), date(1987, 7, 1)),  # standby
        (date(1993, 7, 1), date(2001, 6, 15)),  # lost its TDRS link
    ],
    5: [  # approximate dates
        (date(2006, 1, 15), date(2006, 1, 31)),  # primary solar array drive failed
        (date(2009, 12, 18), date(2010, 1, 7)),  # transmitter technical difficulties
        (date(2011, 11, 18), date(2012, 2, 16)),  # amplifier performance fluctuations
        (date(2013, 1, 6), date(2013, 6, 5)),  # gyroscope failure
    ],
    6: [(date(1993, 10, 5), date(1993, 10, 6))],  # explosion in liquid fuel system
    7: [(date(2003, 5, 31), date.today())],  # SLC-off
    8: [],
    9: [],
}

global_xmin = date(1972, 7, 23) - timedelta(weeks=52 * 4)
global_xmax = date.today()

fig, ax = plt.subplots(figsize=(5.5, 3))

cmap = iter(plt.cm.tab10(range(9, 0, -1)))  # type: ignore[attr-defined]
ymin = args.bar_start
yticks = []
for satellite in range(9, 0, -1):
    # Bar plot
    kwargs = {
        'yrange': (ymin, args.bar_height),
        'alpha': 0.8,
        'color': next(cmap),
        'edgecolor': (0, 0, 0, 0.8),
        'linewidth': 0.5,
    }

    xranges = [(start, end - start) for start, end in working[satellite]]
    ax.broken_barh(xranges, hatch=None, **kwargs)

    xranges = [(start, end - start) for start, end in failing[satellite]]
    ax.broken_barh(xranges, hatch='////', **kwargs)

    # Label
    xmin = global_xmax
    xmax = global_xmin
    if working[satellite]:
        xmin = min(xmin, working[satellite][0][0])
        xmax = max(xmax, working[satellite][-1][1])
    if failing[satellite]:
        xmin = min(xmin, failing[satellite][0][0])
        xmax = max(xmax, failing[satellite][-1][1])

    if (xmin - global_xmin) > (global_xmax - xmax):
        # Left side label
        x = xmin - timedelta(weeks=52)
        horizontalalignment = 'right'
    else:
        # Right side label
        x = xmax + timedelta(weeks=52)
        horizontalalignment = 'left'

    start = f'{xmin:%b %Y}'
    end = f'{xmax:%b %Y}'
    if xmax == date.today():
        end = 'Present'
    if start == end:
        s = start
    else:
        s = f'{start}--{end}'

    kwargs = {
        'y': ymin + args.bar_height / 2,
        's': s,
        'verticalalignment': 'center_baseline',
    }

    ax.text(x, horizontalalignment=horizontalalignment, **kwargs)

    yticks.append(ymin + args.bar_height / 2)
    ymin += args.bar_height + args.bar_sep

ax.xaxis_date()
ax.set_xlim(global_xmin, global_xmax)
ax.set_ylabel('Landsat Mission')
ax.set_yticks(yticks)
ax.set_yticklabels(map(str, range(9, 0, -1)))
ax.tick_params(axis='both', which='both', top=False, right=False)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()
