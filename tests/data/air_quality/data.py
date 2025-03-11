#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

columns = [
    'Date',
    'Time',
    'CO(GT)',
    'PT08.S1(CO)',
    'NMHC(GT)',
    'C6H6(GT)',
    'PT08.S2(NMHC)',
    'NOx(GT)',
    'PT08.S3(NOx)',
    'NO2(GT)',
    'PT08.S4(NO2)',
    'PT08.S5(O3)',
    'T',
    'RH',
    'AH',
]

nrows = 50
data = np.random.rand(nrows, len(columns))

df = pd.DataFrame(data, columns=columns)


df.to_csv('data.csv', index=False)
