# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:09:54 2025

@author: malco
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import csv

# %% define function for reading in file, and read

def parse_tests_with_headers(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    project = None
    tests = {}
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].strip().startswith('Project:'):
            project = lines[i].strip().split(',', 1)[1].strip()
            i += 1
            break
        i += 1

    if project is None:
        raise ValueError("Project line not found")

    while i < n:
        line = lines[i].strip()
        if line.startswith('Test:'):
            test_name = line.split(',', 1)[1].strip()

            i += 1
            while i < n and not lines[i].strip().startswith('Interval data:'):
                i += 1

            if i >= n:
                print(f"Reached EOF while looking for Interval data for test '{test_name}'")
                break

            header_line = lines[i].strip()
            print(f"\nParsing Test: {test_name}")
            print(f"Raw header line: {header_line!r}")

            header = [h.strip() for h in header_line.split(',') if h.strip() != '']

            print(f"Processed header columns ({len(header)}): {header}")

            if len(header) == 0:
                raise ValueError(f"Header line empty for test '{test_name}' at line {i+1}")

            data_start = i + 3

            i = data_start
            data_lines = []

            while i < n:
                curr_line = lines[i].strip()
                if curr_line.startswith('Test:') or curr_line.startswith('Project:') or curr_line == '':
                    break
                data_lines.append(lines[i])
                i += 1

            if not data_lines:
                print(f"No data lines found for test '{test_name}'")
                continue

            data_str = ''.join(data_lines)
            df = pd.read_csv(StringIO(data_str), header=None)

            print(f"Dataframe shape: {df.shape}")
            print(f"Assigning columns...")

            if df.shape[1] != len(header):
                raise ValueError(f"Data columns ({df.shape[1]}) and header columns ({len(header)}) count mismatch for test '{test_name}'")

            df.columns = header

            key = f"{project} | {test_name}"
            tests[key] = df

        else:
            i += 1

    return tests

filepath = 'Rheology\\2025-06-20_compiled.csv'

dfs = parse_tests_with_headers(filepath)

rheo_names = [k for k in dfs.keys()]

rheo_tests = [t for t in dfs.values()]

# %% plotting variables

mark = ['o', '^', 's', 'd', 'x']

col = ['k', 'r', 'b', 'm', 'c', 'g']

# %% plot frequency sweeps

fig, axs = plt.subplots()

i = 0

for idx, r in enumerate(rheo_names):
    
    if 'frequency' not in r:
        continue
    elif i == 3:
        i += 1
        continue
    
    label = ' '.join(r.split(' ')[2].strip(',').split('_')[1:-2])
    
    rheo_test = rheo_tests[idx]
    
    plt.plot(rheo_test['Angular Frequency'],
             rheo_test['Storage Modulus'],
             c=col[i], marker=mark[i],
             label=label)

    plt.plot(rheo_test['Angular Frequency'],
             rheo_test['Loss Modulus'],
             c=col[i], mfc='w', marker=mark[i])
    
    i += 1


plt.yscale('log')

plt.xscale('log')

plt.ylabel('Modulus [Pa]')

plt.xlabel(r'$\omega$ [1/s]')

plt.legend()

plt.show()

fig.savefig(f'Rheology\\{today}_frequency_sweeep', dpi=300)
fig.savefig(f'Rheology\\{today}_frequency_sweeep.svg')

# %% plot stress sweeps

fig, axs = plt.subplots()

i = 0

for idx, r in enumerate(rheo_names):
    
    if 'stress' not in r:
        continue
    
    label = ' '.join(r.split(' ')[2].strip(',').split('_')[1:-2])
    
    rheo_test = rheo_tests[idx]
    
    plt.plot(rheo_test['Shear Rate'],
             rheo_test['Shear Stress'],
             c=col[i], marker=mark[i],
             label=label)

    i += 1

plt.yscale('log')

plt.xscale('log')

plt.ylabel(r'$\sigma$ [Pa]')

plt.xlabel(r'$\omega$ [1/s]')

plt.legend()

plt.show()

fig.savefig(f'Rheology\\{today}_stress_sweeep', dpi=300)
fig.savefig(f'Rheology\\{today}_stress_sweeep.svg')

# %% plot frequency sweeps

fig, axs = plt.subplots()

i = 0

for idx, r in enumerate(rheo_names):
    
    if 'frequency' not in r:
        continue
    elif i == 3:
        i += 1
        continue
    
    label = ' '.join(r.split(' ')[2].strip(',').split('_')[1:-2])
    
    rheo_test = rheo_tests[idx]
    
    plt.plot(rheo_test['Angular Frequency'],
             rheo_test['Shear Stress'],
             c=col[i], marker=mark[i],
             label=label)
    
    i += 1


plt.yscale('log')

plt.xscale('log')

plt.ylabel(r'$\sigma$ [Pa]')

plt.xlabel(r'$\omega$ [1/s]')

plt.legend()

plt.show()

fig.savefig(f'Rheology\\{today}_frequency_sweeep_stress', dpi=300)
fig.savefig(f'Rheology\\{today}_frequency_sweeep_stress.svg')

# %% plot strain sweeps

fig, axs = plt.subplots()

i = 0

for idx, r in enumerate(rheo_names):
    
    if 'strain' not in r:
        continue
    elif i == 0:
        i += 1
        continue
    
    label = ' '.join(r.split(' ')[2].strip(',').split('_')[1:-2])
    
    rheo_test = rheo_tests[idx]
    
    plt.plot(rheo_test['Shear Strain'],
             rheo_test['Storage Modulus'],
             c=col[i], marker=mark[i],
             label=label)

    plt.plot(rheo_test['Shear Strain'],
             rheo_test['Loss Modulus'],
             c=col[i], mfc='w', marker=mark[i])
    
    i += 1


plt.yscale('log')

plt.xscale('log')

plt.ylabel('Modulus [Pa]')

plt.xlabel(r'$\gamma$ [%]')

plt.legend()

plt.show()

fig.savefig(f'Rheology\\{today}_strain_sweeep', dpi=300)
fig.savefig(f'Rheology\\{today}_strain_sweeep.svg')