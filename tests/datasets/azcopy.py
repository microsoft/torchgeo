#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Basic mock-up of the azcopy CLI."""

import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    copy = subparsers.add_parser('copy')
    copy.add_argument('source')
    copy.add_argument('destination')
    copy.add_argument('--recursive', default='false')
    sync = subparsers.add_parser('sync')
    sync.add_argument('source')
    sync.add_argument('destination')
    sync.add_argument('--recursive', default='true')
    args, _ = parser.parse_known_args()

    if args.recursive == 'true':
        shutil.copytree(args.source, args.destination, dirs_exist_ok=True)
    else:
        shutil.copy(args.source, args.destination)
