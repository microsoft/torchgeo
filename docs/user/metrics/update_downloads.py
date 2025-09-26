#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import argparse
import re
import time

import pandas as pd
import requests
from common import index, name_to_conda, name_to_cran, name_to_pypi

columns = [
    'PyPI/CRAN Last Week',
    'PyPI/CRAN Last Month',
    'PyPI/CRAN All Time',
    'Conda All Time',
    'Total All Time',
]


def pypistats(package: str) -> tuple[int, int]:
    """Retrieve download statistics from pypistats.org.

    See https://pypistats.org/api/ for documentation.

    Args:
        package: Name of the PyPI library.

    Returns:
        Tuple of total number of downloads in the last (week, month).
    """
    response = requests.get(f'https://pypistats.org/api/packages/{package}/recent')
    data = response.json()['data']
    return data['last_week'], data['last_month']


def pepytech(package: str, api_key: str) -> int:
    """Retrieve download statistics from pepy.tech.

    See https://pepy.tech/pepy-api for documentation.

    Args:
        package: Name of the PyPI library.
        api_key: pepy.tech API key.

    Returns:
        Total number of downloads.
    """
    # API limit is 10 requests per minute
    time.sleep(6)

    headers = {'X-API-Key': api_key}
    response = requests.get(
        f'https://api.pepy.tech/api/v2/projects/{package}', headers=headers
    )
    data = response.json()
    return data['total_downloads']


def cranlogs(package: str) -> tuple[int, int, int]:
    """Retrieve download statistics from cranlogs.r-pkg.org.

    See https://cranlogs.r-pkg.org/#jsonapi for documentation.

    Args:
        package: Name of the PyPI library.

    Returns:
        Tuple of total number of downloads in the (last-week, last-month, grand-total).
    """
    response1 = requests.get(
        f'https://cranlogs.r-pkg.org/downloads/total/last-week/{package}'
    )
    data1 = response1.json()[0]
    response2 = requests.get(
        f'https://cranlogs.r-pkg.org/downloads/total/last-month/{package}'
    )
    data2 = response2.json()[0]
    # https://github.com/r-hub/cranlogs.app/issues/49
    response3 = requests.get(
        f'https://cranlogs.r-pkg.org/downloads/total/1970-01-01:2100-01-01/{package}'
    )
    data3 = response3.json()[0]
    return data1['downloads'], data2['downloads'], data3['downloads']


def condaforge(package: str) -> int:
    """Retrieve download statistics from anaconda.org/conda-forge.

    Args:
        package: Name of the PyPI library.

    Returns:
        Total number of downloads.
    """
    # TODO: should really be using one of the following instead:
    # https://github.com/conda-incubator/condastats
    # https://github.com/anaconda/anaconda-package-data
    response = requests.get(f'https://anaconda.org/conda-forge/{package}')
    for line in response.iter_lines():
        if match := re.search(r'<span>(\d+)</span> total downloads', str(line)):
            return int(match.group(1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True, help='pepy.tech API key')
    args = parser.parse_args()

    df = pd.DataFrame(0.0, columns=columns, index=index)
    for name, package in name_to_pypi.items():
        df.loc[name, 'PyPI/CRAN Last Week':'PyPI/CRAN Last Month'] += pypistats(package)
        df.loc[name, 'PyPI/CRAN All Time'] += pepytech(package, api_key=args.api_key)

    for name, package in name_to_cran.items():
        df.loc[name, 'PyPI/CRAN Last Week':'PyPI/CRAN All Time'] += cranlogs(package)

    for name, package in name_to_conda.items():
        df.loc[name, 'Conda All Time'] += condaforge(package)

    df['Total All Time'] = df['PyPI/CRAN All Time'] + df['Conda All Time']

    df.rename('`{}`_'.format, inplace=True)
    df.to_csv('downloads.csv', float_format='{:,.0f}'.format, index_label='Library')
