#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Basic mock-up of the AWS CLI."""

import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    s3 = subparsers.add_parser('s3')
    subsubparsers = s3.add_subparsers()
    cp = subsubparsers.add_parser('cp')
    cp.add_argument('source')
    cp.add_argument('destination')
    args, _ = parser.parse_known_args()
    shutil.copy(args.source, args.destination)
