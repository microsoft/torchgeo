#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import io
import json
import os
import tarfile

import torch

tar_path = os.path.join('example-000000.tar')

metadata = {
    's1_grd': [
        [
            '0772176_-96.00_44.00/0865831_-95.92_43.96/20221121',
            '0772176_-96.00_44.00/0865831_-95.92_43.96/20220322',
        ],
        [
            '0772176_-96.00_44.00/0865832_-95.95_43.99/20221121',
            '0772176_-96.00_44.00/0865832_-95.95_43.99/20220322',
        ],
    ],  # simplify to 2 local patches and 2 timestamps
    's2_toa': [
        [
            '0772176_-96.00_44.00/0865831_-95.92_43.96/20221121',
            '0772176_-96.00_44.00/0865831_-95.92_43.96/20220322',
        ],
        [
            '0772176_-96.00_44.00/0865832_-95.95_43.99/20221121',
            '0772176_-96.00_44.00/0865832_-95.95_43.99/20220322',
        ],
    ],  # simplify to 2 timestamps
    's3_olci': [
        '0772176_-96.00_44.00/20210127',
        '0772176_-96.00_44.00/20210227',
    ],  # simplify to 2 timestamps
    's5p_co': [
        '0772176_-96.00_44.00/20211201',
        '0772176_-96.00_44.00/20211101',
    ],  # simplify to 2 timestamps
    's5p_no2': [
        '0772176_-96.00_44.00/20211201',
        '0772176_-96.00_44.00/20211101',
    ],  # simplify to 2 timestamps
    's5p_o3': [
        '0772176_-96.00_44.00/20211201',
        '0772176_-96.00_44.00/20211101',
    ],  # simplify to 2 timestamps
    's5p_so2': [
        '0772176_-96.00_44.00/20211201',
        '0772176_-96.00_44.00/20211101',
    ],  # simplify to 2 timestamps
    'dem': ['0772176_-96.00_44.00'],
}

grid_id = '0772176'

with tarfile.open(tar_path, 'w') as tar:
    # Generate random tensors for each modality
    data = {
        f'{grid_id}.s1_grd.pth': torch.rand(2, 2, 2, 264, 264, dtype=torch.float32),
        f'{grid_id}.s2_toa.pth': torch.rand(2, 2, 13, 264, 264, dtype=torch.float32).to(
            torch.int16
        ),
        f'{grid_id}.s3_olci.pth': torch.rand(2, 21, 96, 96, dtype=torch.float32),
        f'{grid_id}.s5p_co.pth': torch.rand(2, 1, 28, 28, dtype=torch.float32),
        f'{grid_id}.s5p_no2.pth': torch.rand(2, 1, 28, 28, dtype=torch.float32),
        f'{grid_id}.s5p_o3.pth': torch.rand(2, 1, 28, 28, dtype=torch.float32),
        f'{grid_id}.s5p_so2.pth': torch.rand(2, 1, 28, 28, dtype=torch.float32),
        f'{grid_id}.dem.pth': torch.rand(1, 960, 960, dtype=torch.float32),
    }

    # Save tensors to tarfile
    for filename, tensor in data.items():
        tensor_bytes = io.BytesIO()
        torch.save(tensor, tensor_bytes)
        tensor_bytes.seek(0)
        tarinfo = tarfile.TarInfo(name=filename)
        tarinfo.size = len(tensor_bytes.getvalue())
        tar.addfile(tarinfo, tensor_bytes)

    # metadata
    json_bytes = json.dumps(metadata).encode('utf-8')
    json_bytes_io = io.BytesIO(json_bytes)
    tarinfo = tarfile.TarInfo(name=f'{grid_id}.json')
    tarinfo.size = len(json_bytes)
    tar.addfile(tarinfo, json_bytes_io)
