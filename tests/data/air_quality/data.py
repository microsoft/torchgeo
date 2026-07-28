#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

nrows = 20

timestamps = pd.date_range(start='2004-03-10 18:00:00', periods=nrows, freq='h')
date_col = timestamps.strftime('%m/%d/%Y')
time_col = timestamps.strftime('%H:%M:%S')

t = np.linspace(0, 3 * np.pi, nrows)
T = 12 + 5 * np.sin(t) + np.random.normal(0, 0.5, nrows)
RH = 50 + 10 * np.sin(t + 1) + np.random.normal(0, 2, nrows)
AH = 0.6 + 0.02 * RH / 10 + np.random.normal(0, 0.02, nrows)


CO = np.abs(np.random.normal(2, 0.7, nrows))  # ~0-5
NMHC = np.abs(np.random.normal(100, 50, nrows))  # ~0-300
NOx = np.abs(np.random.normal(150, 60, nrows))  # ~0-300
NO2 = np.abs(np.random.normal(100, 40, nrows))  # ~0-200
C6H6 = np.abs(np.random.normal(8, 3, nrows))  # ~0-20

PT08_S1 = 1000 + CO * 200 + np.random.normal(0, 50, nrows)
PT08_S2 = 800 + C6H6 * 20 + np.random.normal(0, 40, nrows)
PT08_S3 = 1200 - NOx * 2 + np.random.normal(0, 50, nrows)
PT08_S4 = 1300 + NO2 * 3 + np.random.normal(0, 50, nrows)
PT08_S5 = 900 + NOx * 1.5 + np.random.normal(0, 50, nrows)


df = pd.DataFrame(
    {
        'Date': date_col,
        'Time': time_col,
        'CO(GT)': np.round(CO, 1),
        'PT08.S1(CO)': np.round(PT08_S1).astype(int),
        'NMHC(GT)': np.round(NMHC).astype(int),
        'C6H6(GT)': np.round(C6H6, 1),
        'PT08.S2(NMHC)': np.round(PT08_S2).astype(int),
        'NOx(GT)': np.round(NOx).astype(int),
        'PT08.S3(NOx)': np.round(PT08_S3).astype(int),
        'NO2(GT)': np.round(NO2).astype(int),
        'PT08.S4(NO2)': np.round(PT08_S4).astype(int),
        'PT08.S5(O3)': np.round(PT08_S5).astype(int),
        'T': np.round(T, 1),
        'RH': np.round(RH, 1),
        'AH': np.round(AH, 4),
    }
)

for col in df.columns[2:]:
    mask = np.random.rand(nrows) < 0.025
    df.loc[mask, col] = -200

df.to_csv('data.csv', index=False)
