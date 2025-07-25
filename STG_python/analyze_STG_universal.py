# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:48:26 2025

@author: malco
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

today = datetime.today().strftime('%Y-%m-%d') # today's date!

# %% take in most recent test_info file

# --- CHANGED: Use absolute paths for all file and folder operations ---
script_dir = os.path.dirname(os.path.abspath(__file__))

date = '2025-07-24'

test_info_path = os.path.join(script_dir, 'test_info', f'{date}_test_info.csv')
saved_data_dir = os.path.join(script_dir, 'saved_data')

# Zwick-related folders: go one level up from STG-python to Zwick
file_folder = os.path.abspath(os.path.join(script_dir, '..', 'Zwick'))
csv_folder = os.path.join(file_folder, 'CSV')
air_folder = os.path.join(file_folder, 'AIR', 'CSV')

# Read test info
test_info = pd.read_csv(test_info_path)
# --- CHANGED: Expect air_test_info.csv in the same folder as the script ---
# --- CHANGED: Use air_test_codes.csv instead of air_test_info.csv ---
air_info = pd.read_csv(os.path.join(script_dir, 'air_test_codes.csv'))
# --- END CHANGED ---
# --- END CHANGED ---
# Load pickle files
with open(os.path.join(saved_data_dir, 'tests.pkl'), 'rb') as f:
    tests = pickle.load(f)

with open(os.path.join(saved_data_dir, 'air_tests.pkl'), 'rb') as f:
    air_tests = pickle.load(f)

# Create output folders if they don’t exist
os.makedirs(os.path.join(script_dir, 'Figures', 'Raw'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'Figures'), exist_ok=True)
# --- END CHANGED ---

# --- CHANGED: Utility function to create labeled subfolders and save figures ---
def save_labeled_figure(fig, base_folder, date, label, test_type=None, trial=None, ext='png'):
    """
    Save a matplotlib figure in a structured, labeled way.
    - base_folder: root folder (e.g., Figures/Raw)
    - date: experiment date (string)
    - label: what the plot shows (e.g., 'stress_strain', 'air', etc.)
    - test_type: 'air' or 'sample' (optional)
    - trial: trial number (optional)
    - ext: file extension (default 'png')
    """
    import os
    folder = os.path.join(base_folder, date)
    if test_type:
        folder = os.path.join(folder, test_type)
    os.makedirs(folder, exist_ok=True)
    if trial is not None:
        fname = f'{date}_trial{trial}_{label}.{ext}'
    else:
        fname = f'{date}_{label}.{ext}'
    fig.savefig(os.path.join(folder, fname), dpi=300)
    fig.savefig(os.path.join(folder, fname.replace('.png', '.svg')))
# --- END CHANGED ---





# %% useful plotting variables

mark = ['o', '^', 's', 'd']

col = ['k', 'r', 'b', 'g', 'm', 'c', 'tab:blue', 'tab:orange','purple']

# %% plot raw data from each test and save

for idx, test in enumerate(tests):

    info = test_info.loc[idx]

    fig, axs = plt.subplots()

    plt.plot(test.travel / info.height, test.stress, 'r', label='Raw')

    plt.plot(test.travel / info.height, test.stress_adj, 'k', label='Adjusted')
        
    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Strain')
    
    plt.title(f'{info.date} trial {info.trial}')
    
    plt.legend()
    plt.tight_layout()
    # --- CHANGED: Save in labeled subfolders ---
    save_labeled_figure(fig, os.path.join(script_dir, 'Figures', 'Raw'), str(info.date), 'stress_strain', test_type='sample', trial=info.trial)
    # --- END CHANGED ---
    plt.close(fig)
    
# %% plot raw air data from each test and save

for idx, test in enumerate(air_tests):

    info = air_info.loc[idx]

    fig, axs = plt.subplots()
    
    plt.plot(test.travel, test.force_cut)
    
    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Depth [mm]')
    
    plt.title(f'{info.date} trial {info.trial} (air)')
    
    plt.tight_layout()
    # --- CHANGED: Save in labeled subfolders ---
    save_labeled_figure(fig, os.path.join(script_dir, 'Figures', 'Raw'), str(info.date), 'air_force_cut', test_type='air', trial=info.trial)
    # --- END CHANGED ---
    plt.close(fig)

# %% plot raw data on a plot together for each date

for date, info in test_info.groupby('date'):
    
    fig, axs = plt.subplots()
    
    for idx, ind_info in info.iterrows():
                
        plt.plot(tests[idx].travel / ind_info.height, tests[idx].stress_adj)

    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Strain [mm]')
    
    plt.title(f'{date} all trials')
    
    fig.savefig(f'Figures\\{info.date.unique()[0]}_all', dpi=300)
    fig.savefig(f'Figures\\{info.date.unique()[0]}_all.svg')

# %% plot raw relaxation data on a plot together for each velocity, as a function of time

for date, info_date in test_info.groupby('date'):

    i = 0

    fig, axs = plt.subplots()
    
    for v, info in info_date.groupby('velocity'):
        j = 0
        
        for idx, ind_info in info.iterrows():
            
            after_impact = tests[idx].loc[ind_info.stop_idx:ind_info.return_idx]
            
            if not j:
                plt.plot(after_impact.time[1:] - after_impact.iloc[1].time,
                         after_impact.stress_adj[1:] / after_impact.iloc[1].stress_adj,
                         label=f'{v} mm/s', c=col[i])
            else:
                plt.plot(after_impact.time[1:] - after_impact.iloc[1].time,
                         after_impact.stress_adj[1:] / after_impact.iloc[1].stress_adj,
                         c=col[i])
                
            j += 1
        
        i += 1
    
    plt.ylabel('% Max stress [Pa]')
    
    plt.xlabel('Time [s]')
    
    plt.title(f'{date} all trials')
    
    plt.legend()
    
    plt.show()
        
    fig.savefig(f'Figures\\{date}_all_time', dpi=300)
    fig.savefig(f'Figures\\{date}_all_time.svg')
    
# %% plot raw data on a plot together for each date, colored by velocity

for date, info in test_info.groupby('date'):
    i = 0

    fig, axs = plt.subplots()
    
    for v, info_v in info.groupby('velocity'):
        j = 0
        
        for idx, ind_info in info_v.iterrows():
            test = tests[idx]
            
            forward = test[test.travel > test.travel.shift(3)]
            
            if not j:
                plt.plot(forward.travel / ind_info.height, forward.stress_adj,
                         c=col[i], label=f'{v} mm/s')
            else:
                plt.plot(forward.travel / ind_info.height, forward.stress_adj,
                         c=col[i])
            
            j += 1

        i += 1

    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Strain [mm]')
    
    plt.title(f'{date} all trials')
    
    plt.legend()
    
    fig.savefig(f'Figures\\{date}_all', dpi=300)
    fig.savefig(f'Figures\\{date}_all.svg')

# %% plot max stress as a function of strain, separating by velocity

for date, info_date in test_info.groupby('date'):
    fig, axs = plt.subplots()

    i = 0
    
    for v, info_v in info_date.groupby('velocity'):
            plt.scatter(info_v.depth / info_v.height, info_v.max_stress,
                         c=col[i], ec='k', lw=0.5, label=f'{v} mm/s')
                    
            i += 1
    
    plt.legend()
    
    plt.ylabel('Max stress [Pa]')
    
    plt.xlabel('Strain [mm]')
    
    plt.show()
    
    fig.savefig(f'Figures\\{date}_max_stress-vs-v', dpi=300)
    fig.savefig(f'Figures\\{date}_max_stress-vs-v.svg')


# %% plot max stress as a function of velocity, for each strain

for date, info_date in test_info.groupby('date'):

    fig, axs = plt.subplots()
    
    for d, info in info_date.groupby('depth'):
                
        for v, info_v in info.groupby('velocity'):
                
            plt.scatter(info.velocity, info.max_stress, ec='k', lw=0.5)
    
        plt.ylabel('Max stress [Pa]')
        
        plt.xlabel('Velocity [mm/s]')
        
        plt.title(f'{d / info.height.unique()[0]:.2f} strain')
    
    plt.show()
        
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain', dpi=300)
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain.svg')

# %% plot total energy as a function of velocity

for date, info_date in test_info.groupby('date'):

    fig, axs = plt.subplots()
    
    plt.scatter(info_date.depth / info_date.height, info_date.e_tot * 1e-3)
    
    plt.ylabel('Total Energy [J]')
    
    plt.xlabel('Strain')
    
    plt.title(f'{date} all trials')
    
    plt.show()
    
    fig.savefig(f'Figures\\{date}_e_tot-by_strain', dpi=300)
    fig.savefig(f'Figures\\{date}_e_tot-by_strain.svg')

# %% plot total energy as a function of velocity

for date, info_date in test_info.groupby('date'):

    fig, axs = plt.subplots()
    
    i = 0
    
    for d, info in info_date.groupby('depth'):
                
        for v, info_v in info.groupby('velocity'):
            
            plt.errorbar(v, info_v.max_stress.mean(),
                         yerr=info_v.max_stress.std(),
                         mec='k', lw=0.5, c=col[i], marker='o',
                         elinewidth=1, capsize=5)
            
        i += 1
    
        plt.ylabel('Max stress [Pa]')
        
        plt.xlabel('Velocity [mm/s]')
        
        plt.title(f'{d / info.height.unique()[0]:.2f} strain, {date}')
    
    plt.show()
        
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain', dpi=300)
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain.svg')


# %% plot stress after wait time as percent of max stress

for date, info_date in test_info.groupby('date'):

    fig, axs = plt.subplots()
    
    i = 0
    
    for v, info_v in info_date.groupby('velocity'):
        
        j = 0
        
        for d, info_d in info_v.groupby('depth'):
            if not j:
                plt.errorbar(d / info_d.height.mean(),
                             (info_d.relax_10 / info_d.max_stress).mean(),
                             yerr=(info_d.relax_10 / info_d.max_stress).std(),
                             c=col[i], marker='o', capsize=5, lw=0, elinewidth=1,
                             label=f'{v} mm/s')
            else:
                plt.errorbar(d / info_d.height.mean(),
                             (info_d.relax_10 / info_d.max_stress).mean(),
                             yerr=(info_d.relax_10 / info_d.max_stress).std(),
                             c=col[i], marker='o', capsize=5, lw=0, elinewidth=1)
                
            j += 1
        i += 1
    
    plt.ylabel('% stress after relaxation')
    
    plt.xlabel('Strain [mm]')
    
    plt.title(f'{date} all trials')
    
    plt.legend()
    
    fig.savefig(f'Figures\\{date}_stress_relax', dpi=300)
    fig.savefig(f'Figures\\{date}_stress_relax.svg')
    
# %%

for (g_wt, p_wt), info in test_info.groupby(['gel_wt', 'particle_wt']):
    
    fig, axs = plt.subplots()
    
    i = 0
    
    for v, info_v in info.groupby('velocity'):
                
        j = 0
        
        for d, info_d in info_v.groupby('depth'):
            if not j:
                plt.errorbar(d / info_d.height.mean(),
                             (info_d.relax_10 / info_d.max_stress).mean(),
                             yerr=(info_d.relax_10 / info_d.max_stress).std(),
                             c=col[i], marker='o', capsize=5, lw=0, elinewidth=1,
                             label=f'{v} mm/s')
            else:
                plt.errorbar(d / info_d.height.mean(),
                             (info_d.relax_10 / info_d.max_stress).mean(),
                             yerr=(info_d.relax_10 / info_d.max_stress).std(),
                             c=col[i], marker='o', capsize=5, lw=0, elinewidth=1)
                
            j += 1
        i += 1
    
    plt.ylabel('% stress after relaxation')
    
    plt.xlabel('Strain [mm]')
    
    plt.title(f'{date} all trials')
    
    plt.legend()
    
    fig.savefig(f'Figures\\{g_wt}_bovine_{p_wt}_cs_stress_relax', dpi=300)
    fig.savefig(f'Figures\\{g_wt}_bovine_{p_wt}_cs_stress_relax.svg')

# --- CHANGED: Write a markdown file describing each graph type ---
with open(os.path.join(script_dir, 'Figures', 'Raw', 'README_RAW.md'), 'w') as f:
    f.write('# Figures/Raw Graphs: Hydrogels Compression Analysis\n\n')
    f.write('This folder contains raw and processed plots from hydrogel compression experiments.\n\n')
    f.write('## Folder Structure\n')
    f.write('- Each date has its own folder.\n')
    f.write('- Within each date, plots are separated by test type ("sample" for hydrogel, "air" for control/air tests).\n')
    f.write('- Filenames include the trial number and a label for the plot type.\n\n')
    f.write('## Plot Types\n')
    f.write('### stress_strain\n')
    f.write('- **What it shows:** Stress vs. strain for a single hydrogel sample.\n')
    f.write('- **Red curve:** Raw stress (from force sensor).\n')
    f.write('- **Black curve:** Adjusted stress (after subtracting air/control background).\n')
    f.write('- **Meaning:** Shows how the hydrogel resists compression, and how much of the force is due to the sample vs. the impactor/air.\n\n')
    f.write('### air_force_cut\n')
    f.write('- **What it shows:** Force (or stress) vs. depth for an air/control test (no sample).\n')
    f.write('- **Meaning:** Used to correct for the force due to the impactor itself, so that sample measurements are accurate.\n\n')
    f.write('## How to Use\n')
    f.write('- Compare "stress_strain" plots across trials and dates to see how different hydrogels behave.\n')
    f.write('- Use "air_force_cut" plots to check the background/impactor force profile.\n')
    f.write('- All plots are saved as both PNG and SVG for publication or further analysis.\n')
# --- END CHANGED ---