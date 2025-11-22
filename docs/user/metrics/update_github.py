#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import argparse
import re

import pandas as pd
import requests
from common import hardcoded_coverage, index, name_to_codecov, name_to_github
from requests import Response

columns = [
    'Contributors',
    'Forks',
    'Watchers',
    'Stars',
    'Issues',
    'PRs',
    'Releases',
    'Commits',
    'Test Coverage',
    'License',
]


def page_count(response: Response) -> int:
    """Retrieve page count from paginated response.

    Args:
        response: HTTP response.

    Returns:
        Total number of pages.
    """
    if 'Link' in response.headers:
        if match := re.search(r'&page=(\d+)>; rel="last"', response.headers['Link']):
            return int(match.group(1))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Not strictly required, but increases rate limit
    parser.add_argument('--api-key', help='GitHub API token')
    args = parser.parse_args()

    df = pd.DataFrame(columns=columns, index=index, dtype=float)
    df = df.astype({'License': str})

    print('\nGitHub')
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }
    if args.api_key:
        headers['Authorization'] = f'Bearer {args.api_key}'

    for name, (owner, repo) in name_to_github.items():
        if name not in index:
            continue

        print(f'* {name}')
        url = f'https://api.github.com/repos/{owner}/{repo}'
        response = requests.get(url, headers=headers)
        data = response.json()
        df.loc[name, 'Forks'] = data['forks_count']
        df.loc[name, 'Watchers'] = data['subscribers_count']
        df.loc[name, 'Stars'] = data['stargazers_count']
        df.loc[name, 'Issues'] = data['open_issues_count']
        df.loc[name, 'License'] = data['license']['spdx_id']

        url = f'https://api.github.com/repos/{owner}/{repo}/commits'
        params: dict[str, int | str] = {'per_page': 1}
        response = requests.get(url, params=params, headers=headers)
        df.loc[name, 'Commits'] = page_count(response)

        url = f'https://api.github.com/repos/{owner}/{repo}/contributors'
        params = {'per_page': 1}
        response = requests.get(url, params=params, headers=headers)
        df.loc[name, 'Contributors'] = page_count(response)

        url = f'https://api.github.com/repos/{owner}/{repo}/pulls'
        params = {'per_page': 1, 'state': 'all'}
        response = requests.get(url, params=params, headers=headers)
        df.loc[name, 'PRs'] = page_count(response)

        url = f'https://api.github.com/repos/{owner}/{repo}/tags'
        params = {'per_page': 1}
        response = requests.get(url, params=params, headers=headers)
        df.loc[name, 'Releases'] = page_count(response)

    print('\nCodecov')
    headers = {'accept': 'application/json'}
    for name, (service, owner, repo) in name_to_codecov.items():
        print(f'* {name}')
        url = (
            f'https://api.codecov.io/api/v2/{service}/{owner}/repos/{repo}/report/tree'
        )
        response = requests.get(url, headers=headers)
        df.loc[name, 'Test Coverage'] = response.json()[0]['coverage'] / 100

    for name, coverage in hardcoded_coverage.items():
        df.loc[name, 'Test Coverage'] = coverage / 100

    df.sort_values(by=['Contributors', 'Forks'], ascending=False, inplace=True)

    print(df)

    df.rename('`{}`_'.format, inplace=True)
    df['Test Coverage'] = df['Test Coverage'].map('{:.0%}'.format)
    df.to_csv('github.csv', float_format='{:,.0f}'.format, index_label='Library')
