# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np
import scipy.integrate as integrate
import shutil
from scipy.signal import argrelextrema
import pickle

today = datetime.today().strftime('%Y-%m-%d')

# %% defining functions

# %%% functions for general setup, reading and saving files

# convert force to stress
def flat_impactor(x, r, adj):
    if not adj: # raw force measurements
        return x.force / (np.pi * (r * 1e-3)**2)
    else: # force measurements with air tests subtracted
        return x.force_adj / (np.pi * (r * 1e-3)**2)
        
# save current dataframe
def save_test_info():
    global info_files, info_folder
    
    # list of all current files in test_info folder
    info_files = [i for i in os.listdir(info_folder) if '.csv' in i]
    
    # move all current files to OLD folder
    for file in info_files:
        shutil.move(os.path.join(info_folder, file),
                    os.path.join(info_folder,'OLD',file))
    
    # save new file in their place
    test_info.to_csv(os.path.join(info_folder, f'{today}_test_info.csv'))

    #added
    with open('saved_data/tests.pkl', 'wb') as f:
        pickle.dump(tests, f)
    with open('saved_data/air_tests.pkl', 'wb') as f:
        pickle.dump(air_tests, f)
        
# import info about all tests
def setup_program():
    # Get the directory where this script is located (STG/STG-python)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # read in file with basic information about each trial
    test_codes = pd.read_csv(code_file)
                                    
    # define new test_info file
    globals()['test_info'] = test_codes
        
    # list of all file names
    all_files = [f for f in os.listdir(file_folder) if f.endswith('.TXT')]
    file_map = {}

    for f in all_files:
        try:
            date_str, trial_str = f.replace('.TXT', '').split('_')
            trial = int(trial_str)
            file_map[(date_str, trial)] = f
        except:
            print(f"Skipping malformed file: {f}")

    print(list(file_map))
    #print(list(test_info.iterrows()))
    
    # get matching filenames in correct order
    test_names = []
    for _, row in test_info.iterrows():
        key = (row['date'], int(row['trial']))
        if key in file_map:
            test_names.append(file_map[key])
    
    globals()['test_names'] = test_names
    print(test_names)
    # order test_info, making sure indices are aligned with test_names
    print(test_names)
    
    # check if all .csv files are already converted
    if len([i for i in os.listdir(csv_folder) if '.csv' in i]) == len(test_names):
        # if so, fill tests list with dataframes corresponding to each trial's data
        globals()['tests'] = [pd.read_csv(os.path.join(csv_folder,i.replace('TXT', 'csv')), index_col=0)
                 for i in test_names]
    else:
        # if some .csv files are missing, read in each .txt file
        globals()['tests'] = [pd.read_csv(os.path.join(file_folder,i),
                             names=['time', 'travel', 'position', 'force'],
                             skiprows=1,sep=',')
                 for i in test_names]
        
        # make sure all .txt files are converted correctly and save as .csv files
        for df_idx, df in enumerate(tests):
            if np.isnan(df.loc[0, 'force']):
                tests[df_idx].drop('force', axis=1, inplace=True)
                tests[df_idx].rename(columns={'position': 'force'}, inplace=True)
            
            # add stress column to each dataframe
            tests[df_idx]['stress'] = tests[df_idx].apply(flat_impactor, axis=1, args=(test_info.iloc[df_idx].width,0))
                
            # save all dataframes to .csv files in correct folder
            tests[df_idx].to_csv(os.path.join(csv_folder,test_names[df_idx].replace('TXT', 'csv')))
    # take in force due to acceleration of impactor at various speeds
    globals()['air_names'] = [i for i in os.listdir(os.path.join(file_folder,'AIR')) if '.TXT' in i]
    
    # define new dataframe for air test info
    globals()['air_info'] = pd.read_csv(os.path.join(script_dir, 'air_test_codes.csv'))
    
    # check if all air .csv files are already converted
    if len([i for i in os.listdir(os.path.join(file_folder,'AIR','CSV')) if '.csv' in i]) == len(air_names):
        # if so, fill air_tests list with dataframes corresponding to each trial's data
        globals()['air_tests'] = [pd.read_csv(os.path.join(os.path.join(file_folder,'AIR','CSV'),i.replace('TXT', 'csv')), index_col=0)
                 for i in air_names]
    else:
        # if some air .csv files are missing, read in each .txt file
        globals()['air_tests'] = [pd.read_csv(os.path.join(file_folder,'AIR',i),
                                              names=['time', 'travel', 'position', 'force'],
                                              skiprows=1,sep=',')
                                  for i in air_names]
        
        # make sure all air .txt files are converted correctly and save as .csv files
        for df_idx, df in enumerate(air_tests):
            if np.isnan(df.loc[0, 'force']):
                air_tests[df_idx].drop('force', axis=1, inplace=True)
                air_tests[df_idx].rename(columns={'position': 'force'}, inplace=True)
    
            # save all air dataframes to .csv files in correct folder
            air_tests[df_idx].to_csv(os.path.join(file_folder,'AIR','CSV',air_names[df_idx].replace('TXT', 'csv')))

    
# make sure all dataframes are ordered correctly
def order_dataframes():
    global test_info
    
    # Ensure 'trial' column is numeric
    test_info['trial'] = pd.to_numeric(test_info['trial'], errors='coerce')

    # Sort test_info by date, and within date by trial (numerically!)
    df_sorted = test_info.groupby('date', group_keys=False).apply(
        lambda x: x.sort_values(by='trial')
    )

    # Reset index for a cleaner dataframe
    test_info = df_sorted.reset_index(drop=True)

# clean up air_test dataframes, identifying regions of acceleration
def get_keep_around_extrema_pair(signal):
    signal = signal.values # work with raw array
    
    global_max_idx = np.argmax(signal) # index of global maximum (acceleration)
    
    global_min_idx = np.argmin(signal) # index of global minimum (deceleration)

    keep_indices = set() # set to hold all relevant indices

    local_minima = argrelextrema(signal, np.less)[0] # indices of local minima
    
    local_maxima = argrelextrema(signal, np.greater)[0] # indices of local maxima

    # for global MAXIMUM
    # find all local minima moving forward in impact
    closest_local_min = local_minima[local_minima > global_max_idx]

    if len(closest_local_min) > 0:
        local_min_idx = closest_local_min[0] # index of closest local minimum

        sign = np.sign(signal[global_max_idx]) # sign of global maximum

        # iterate through signal, keeping all elements of same sign as global
        # maximum, in local vicinity
        for i in range(global_max_idx, len(signal)):
            if np.sign(signal[i]) == sign and signal[i] != 0:
                keep_indices.add(i)
            else:
                break # leave at sign change
        
        sign = np.sign(signal[local_min_idx]) # sign of local minimum
        
        # iterate through signal, keeping all elements of same sign as local
        # minimum, in local vicinity
        for i in range(local_min_idx, -1, -1):
            if np.sign(signal[i]) == sign and signal[i] != 0:
                keep_indices.add(i)
            else:
                break # leave at sign change

    # for global MINIMUM
    # find all local maxima moving backward in impact
    closest_local_max = local_maxima[local_maxima < global_min_idx]

    if len(closest_local_max) > 0:
        local_max_idx = closest_local_max[-1] # index of closest local maximum

        sign = np.sign(signal[global_min_idx]) # sign of global minimum

        # iterate through signal, keeping all elements of same sign as global
        # minimum, in local vicinity
        for i in range(global_min_idx, -1, -1):
            if np.sign(signal[i]) == sign and signal[i] != 0:
                keep_indices.add(i)
            else:
                break # leave at sign change

        sign = np.sign(signal[local_max_idx]) # sign of local maximum

        # iterate through signal, keeping all elements of same sign as local
        # maximum, in local vicinity
        for i in range(local_max_idx, len(signal)):
            if np.sign(signal[i]) == sign and signal[i] != 0:
                keep_indices.add(i)
            else:
                break # leave at sign change

    return keep_indices # return set of relevant indices

# subtract force due to acceleration/deceleration of impactor
def subtract_acceleration():
    global test_info, air_tests, air_info
    
    # iterate through all test information
    for idx, info in test_info.iterrows():
        
        test = tests[idx] # this test
                
        # identify which air_test to use, by velocity
        which = (np.abs(air_info.velocity - info.velocity)).idxmin()
        
        air_test = air_tests[which]
        
        # get relevant indices (acceleration, deceleration)
        keep_indices = get_keep_around_extrema_pair(air_test.force)

        # add column to dataframe for subtracted forces, initially all zero
        air_test['force_cut'] = 0
        
        # set relevant indices to actual values for acceleration
        air_test.loc[list(keep_indices), 'force_cut'] = air_test.loc[list(keep_indices), 'force']
        
        # update air_tests list with new dataframe
        air_tests[which] = air_test
        
        # acceleration portion of dataframe
        acc = air_test[(air_test.travel > air_test.travel.shift(5)) & (air_test.force_cut > 0)]
        
        # deceleration portion of dataframe
        dec = air_test[(air_test.travel > air_test.travel.shift(5)) & (air_test.force_cut < 0)]
        
        # add column for adjusted force, subtracting acceleration
        tests[idx]['force_adj'] = test.force.subtract(acc.force_cut, fill_value=0)
        
        # align ends of force column and deceleration        
        offset = info.stop_idx - dec.index[-1]
        
        # shifted deceleration dataframe
        shifted_dec = dec.copy()
        
        shifted_dec.index = dec.index + offset
        
        # align test.force and shifted_dec.force
        aligned_test_force, aligned_dec_force = test.force_adj.align(shifted_dec.force_cut, join='left', fill_value=0)
        
        tests[idx]['force_adj'] = tests[idx].force - tests[idx].force[0]
        
        # subtract deceleration from adjusted force
        tests[idx]['force_adj'] = aligned_test_force - aligned_dec_force
        
        # clean up indices of dataframe
        tests[idx]['force_adj'] = tests[idx]['force_adj'].reindex(test.index, fill_value=np.nan)
        
        # calculate adjusted stress measurements from adjusted force
        tests[idx]['stress_adj'] = tests[idx].apply(flat_impactor, axis=1, args=(test_info.iloc[idx].width,1))
        
        

# fill columns for tracked trials
def fill(track_rows):
    global test_info, tests
    print(track_rows)
    for test_idx, test in enumerate(tests):
        if track and (test_idx not in track_rows):
            continue

        info = test_info.loc[test_idx]
        
        mask = np.abs(test.travel - info.depth) < 1e-3
        matches = test[mask]
        if not matches.empty:
            print("hello1", test.travel.max(), test.travel.min(), info.depth)
            test_info.loc[test_idx, 'stop_idx'] = matches.index[0]
            test_info.loc[test_idx, 'return_idx'] = matches.index[-1]
        else:
            print("hello", test.travel.max(), test.travel.min(), info.depth)
            #print("hello", test.travel, info.depth, mask)
            print(f"Warning: No matching depth found for test index {test_idx} with depth {info.depth}")
            test_info.loc[test_idx, 'stop_idx'] = np.nan
            test_info.loc[test_idx, 'return_idx'] = np.nan
    subtract_acceleration()

    for test_idx, test in enumerate(tests):
        if track and (test_idx not in track_rows):
            continue

        info = test_info.loc[test_idx]

        forward = test[test.travel > test.travel.shift(5)]
        backward = test[test.travel < test.travel.shift(5)]

        test_info.loc[test_idx, 'e_tot'] = integrate.trapezoid(forward.force_adj, forward.travel)
        test_info.loc[test_idx, 'e_tot/t'] = test_info.loc[test_idx, 'e_tot'] / (info.depth * 1e-3)

        try:
            test_info.loc[test_idx, 'relax_10'] = test.loc[forward.index[-1], 'stress_adj']
        except (KeyError, IndexError):
            test_info.loc[test_idx, 'relax_10'] = np.nan

    # Convert to integer only where valid
    test_info['stop_idx'] = pd.to_numeric(test_info['stop_idx'], errors='coerce').astype('Int64')
    test_info['return_idx'] = pd.to_numeric(test_info['return_idx'], errors='coerce').astype('Int64')

    test_info['max_stress'] = [test.get('stress_adj', pd.Series([np.nan])).max() for test in tests]

                    
# %% define main

def main():
    global code_file, info_folder, file_folder, track, track_rows, \
        csv_folder, info_files
    
    # Path to the file with basic trial information (inside STG/STG-python/test_info)
    code_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_codes (2).csv')

    # Folder with test_info files (inside STG/STG-python/test_info)
    info_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_info')

    # Folder with .TXT files (go up to STG/Zwick)
    file_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Zwick'))

    # Folder with .CSV files inside Zwick
    csv_folder = os.path.join(file_folder, 'CSV')
    
    setup_program()

    for i, row in test_info.iterrows():
        print(f"Row {i}: info trial={row['trial']}, file={test_names[i]}")
                
    track = 1 # track specific rows
    track_rows = [] # list to hold tracked rows

    # get tracked rows systematically
    if track:
        for df_idx, df_row in test_info.iterrows():
            track_rows.append(df_idx)

            continue
    fill(track_rows)
        
    save_test_info()
            
            
# %% run main

if __name__ == '__main__':
    main()