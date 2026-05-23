#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import argparse

import pandas as pd
import requests
from common import name_to_github
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from update_github import page_count

columns = ['Start', 'Stop']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Not strictly required, but increases rate limit
    parser.add_argument('--api-key', help='GitHub API token')
    args = parser.parse_args()

    # GitHub
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'

    df = pd.DataFrame(columns=columns, index=name_to_github.keys())
    for name, (owner, repo) in name_to_github.items():
        url = f'https://api.github.com/repos/{owner}/{repo}/commits'
        params = {'per_page': 1}
        response = requests.get(url, params=params, headers=headers)
        timestamp = response.json()[0]['commit']['author']['date']
        df.loc[name, 'Stop'] = pd.Timestamp(timestamp)

        params['page'] = page_count(response)
        response = requests.get(url, params=params, headers=headers)
        timestamp = response.json()[0]['commit']['author']['date']
        df.loc[name, 'Start'] = pd.Timestamp(timestamp)

    df.sort_values(by='Start', inplace=True)
    print(df)

    # Plotting
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif'})

    today = pd.Timestamp.today(tz='UTC')
    last_year = pd.Timestamp(today.year - 1, today.month, today.day, tz='UTC')
    fig, ax = plt.subplots(figsize=(7, 7))
    yheight = 0.8
    ymin = -yheight / 2
    for name, (start, stop) in df.iterrows():
        xranges = [(start, stop - start)]
        yrange = (ymin, yheight)

        if stop > last_year:
            color = 'black'
        else:
            color = 'red'

        ax.broken_barh(xranges, yrange, color=color)

        ymin += 1

    # Axes and labels
    left = pd.Timestamp(2015, 6, 1)
    right = today
    ax.set_xlim(left, right)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_yticks(range(len(df.index)), labels=df.index, fontsize=12)
    ax.invert_yaxis()
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_ylim(len(df.index), -1)
    ax.set_axisbelow(True)
    ax.grid(axis='both', linestyle='dashed', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Legend
    handles = [Patch(facecolor='black'), Patch(facecolor='red')]
    labels = ['Active', 'Inactive']
    ax.legend(handles, labels, fontsize=15, loc='lower left', framealpha=1)

    fig.tight_layout()
    fig.savefig('timeline.pdf')
