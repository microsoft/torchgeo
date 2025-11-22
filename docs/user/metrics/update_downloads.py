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
    url = f'https://pypistats.org/api/packages/{package}/recent'
    while True:
        response = requests.get(url)
        match response.status_code:
            case 200:
                # Success
                data = response.json()['data']
                return data['last_week'], data['last_month']
            case 429:
                # Rate Limit Exceeded
                time.sleep(10)
            case _:
                # Other
                print(response.status_code)
                print(response.text)
                raise


def pepytech(package: str, api_key: str) -> int:
    """Retrieve download statistics from pepy.tech.

    See https://pepy.tech/pepy-api for documentation.

    Args:
        package: Name of the PyPI library.
        api_key: pepy.tech API key.

    Returns:
        Total number of downloads.
    """
    url = f'https://api.pepy.tech/api/v2/projects/{package}'
    headers = {'X-API-Key': api_key}
    while True:
        response = requests.get(url, headers=headers)
        match response.status_code:
            case 200:
                # Success
                data = response.json()
                return int(data['total_downloads'])
            case 429:
                # Rate Limit Exceeded
                time.sleep(10)
            case _:
                # Other
                print(response.status_code)
                print(response.text)
                raise


def cranlogs(package: str) -> tuple[int, int, int]:
    """Retrieve download statistics from cranlogs.r-pkg.org.

    See https://cranlogs.r-pkg.org/#jsonapi for documentation.

    Args:
        package: Name of the PyPI library.

    Returns:
        Tuple of total number of downloads in the (last-week, last-month, grand-total).
    """
    url = f'https://cranlogs.r-pkg.org/downloads/total/last-week/{package}'
    response = requests.get(url)
    week = response.json()[0]['downloads']

    url = f'https://cranlogs.r-pkg.org/downloads/total/last-month/{package}'
    response = requests.get(url)
    month = response.json()[0]['downloads']

    # https://github.com/r-hub/cranlogs.app/issues/49
    url = f'https://cranlogs.r-pkg.org/downloads/total/1970-01-01:2100-01-01/{package}'
    response = requests.get(url)
    total = response.json()[0]['downloads']

    return week, month, total


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
    url = f'https://anaconda.org/conda-forge/{package}'
    pattern = r'<span>(\d+)</span> total downloads'
    response = requests.get(url)
    for line in response.iter_lines():
        if match := re.search(pattern, str(line)):
            return int(match.group(1))
    else:
        print(response.status_code)
        print(response.text)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True, help='pepy.tech API key')
    args = parser.parse_args()

    df = pd.DataFrame(0.0, columns=columns, index=index)

    print('\nPyPI')
    for name, package in name_to_pypi.items():
        print(f'* {package}')
        df.loc[name, 'PyPI/CRAN Last Week':'PyPI/CRAN Last Month'] += pypistats(package)  # type: ignore[misc]
        df.loc[name, 'PyPI/CRAN All Time'] += pepytech(package, api_key=args.api_key)

    print('\nCRAN')
    for name, package in name_to_cran.items():
        print(f'* {package}')
        df.loc[name, 'PyPI/CRAN Last Week':'PyPI/CRAN All Time'] += cranlogs(package)  # type: ignore[misc]

    print('\nConda')
    for name, package in name_to_conda.items():
        print(f'* {package}')
        df.loc[name, 'Conda All Time'] += condaforge(package)

    df['Total All Time'] = df['PyPI/CRAN All Time'] + df['Conda All Time']

    print(df)

    df.rename('`{}`_'.format, inplace=True)
    df.to_csv('downloads.csv', float_format='{:,.0f}'.format, index_label='Library')
