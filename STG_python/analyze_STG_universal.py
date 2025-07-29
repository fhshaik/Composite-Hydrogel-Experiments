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
import numpy as np

today = datetime.today().strftime('%Y-%m-%d') # today's date!

# %% take in most recent test_info file

# --- CHANGED: Use absolute paths for all file and folder operations ---
script_dir = os.path.dirname(os.path.abspath(__file__))

date = '2025-07-28'

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

    plt.plot(test.travel / info.height, test.stress, 'r', label='Raw Force', linewidth=2)

    plt.plot(test.travel / info.height, test.stress_adj, 'k', label='Adjusted Force', linewidth=2)
        
    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Strain [-]')
    
    plt.title(f'{info.date} Trial {info.trial} - Stress vs Strain')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # --- CHANGED: Save in labeled subfolders ---
    save_labeled_figure(fig, os.path.join(script_dir, 'Figures', 'Raw'), str(info.date), 'stress_strain', test_type='sample', trial=info.trial)
    # --- END CHANGED ---
    plt.close(fig)

# %% plot combined stress-strain curves with error bands for each velocity group

for date, info_date in test_info.groupby('date'):
    for v, info_v in info_date.groupby('velocity'):
        if len(info_v) > 1:  # Only create combined plots if multiple trials
            fig, axs = plt.subplots()
            
            # Collect all trials with this velocity
            all_strains_raw = []
            all_stresses_raw = []
            all_strains_adj = []
            all_stresses_adj = []
            
            for idx in info_v.index:
                test = tests[idx]
                strain = test.travel / info_v.loc[idx].height
                
                all_strains_raw.append(strain.values)
                all_stresses_raw.append(test.stress.values)
                all_strains_adj.append(strain.values)
                all_stresses_adj.append(test.stress_adj.values)
            
            # Find common strain range
            min_strain = max([s.min() for s in all_strains_raw])
            max_strain = min([s.max() for s in all_strains_raw])
            strain_points = np.linspace(min_strain, max_strain, 100)
            
            # Process raw data
            interpolated_raw = []
            for strain, stress in zip(all_strains_raw, all_stresses_raw):
                stress_interp = np.interp(strain_points, strain, stress)
                interpolated_raw.append(stress_interp)
            
            mean_raw = np.mean(interpolated_raw, axis=0)
            std_raw = np.std(interpolated_raw, axis=0)
            
            # Process adjusted data
            interpolated_adj = []
            for strain, stress in zip(all_strains_adj, all_stresses_adj):
                stress_interp = np.interp(strain_points, strain, stress)
                interpolated_adj.append(stress_interp)
            
            mean_adj = np.mean(interpolated_adj, axis=0)
            std_adj = np.std(interpolated_adj, axis=0)
            
            # Plot with error bands
            plt.plot(strain_points, mean_raw, 'r', label='Raw Force', linewidth=2)
            plt.fill_between(strain_points, mean_raw - std_raw, mean_raw + std_raw, 
                           color='red', alpha=0.3)
            
            plt.plot(strain_points, mean_adj, 'k', label='Adjusted Force', linewidth=2)
            plt.fill_between(strain_points, mean_adj - std_adj, mean_adj + std_adj, 
                           color='black', alpha=0.3)
            
            plt.ylabel('Stress [Pa]')
            plt.xlabel('Strain [-]')
            plt.title(f'{date} Velocity {v} mm/s - Combined Trials (n={len(info_v)})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_labeled_figure(fig, os.path.join(script_dir, 'Figures', 'Raw'), str(date), 
                              f'stress_strain_combined_v{v}', test_type='sample')
            plt.close(fig)
    
# %% plot raw air data from each test and save

for idx, test in enumerate(air_tests):

    info = air_info.loc[idx]

    fig, axs = plt.subplots()
    
    plt.plot(test.travel, test.force_cut, 'b', linewidth=2, label='Air Force')
    
    plt.ylabel('Force [N]')
    
    plt.xlabel('Depth [mm]')
    
    plt.title(f'{info.date} Trial {info.trial} - Air Test (Control)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # --- CHANGED: Save in labeled subfolders ---
    save_labeled_figure(fig, os.path.join(script_dir, 'Figures', 'Raw'), str(info.date), 'air_force_cut', test_type='air', trial=info.trial)
    # --- END CHANGED ---
    plt.close(fig)

# %% plot raw data on a plot together for each date

for date, info in test_info.groupby('date'):
    
    fig, axs = plt.subplots()
    
    for idx, ind_info in info.iterrows():
                
        plt.plot(tests[idx].travel / ind_info.height, tests[idx].stress_adj, 
                label=f'Trial {ind_info.trial}', alpha=0.7, linewidth=1.5)

    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Strain [-]')
    
    plt.title(f'{date} - All Trials Combined')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(f'Figures\\{info.date.unique()[0]}_all.png', dpi=300)
    fig.savefig(f'Figures\\{info.date.unique()[0]}_all.svg')

# %% plot raw relaxation data on a plot together for each velocity, as a function of time

for date, info_date in test_info.groupby('date'):

    i = 0

    fig, axs = plt.subplots()
    
    for v, info in info_date.groupby('velocity'):
        j = 0
        
        for idx, ind_info in info.iterrows():
            
            after_impact = tests[idx].loc[ind_info.stop_idx:ind_info.return_idx]
            
            # Check if after_impact has enough data points
            if len(after_impact) < 2:
                print(f"Warning: Not enough data points for test {idx}, skipping...")
                continue
            
            if not j:
                plt.plot(after_impact.time[1:] - after_impact.iloc[1].time,
                         after_impact.stress_adj[1:] / after_impact.iloc[1].stress_adj,
                         label=f'{v} mm/s', c=col[i], linewidth=2)
            else:
                plt.plot(after_impact.time[1:] - after_impact.iloc[1].time,
                         after_impact.stress_adj[1:] / after_impact.iloc[1].stress_adj,
                         c=col[i], linewidth=2, alpha=0.7)
                
            j += 1
        
        i += 1
    
    plt.ylabel('Normalized Stress [-]')
    
    plt.xlabel('Time [s]')
    
    plt.title(f'{date} - Stress Relaxation Over Time')
    
    plt.legend(title='Impact Velocity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(f'Figures\\{date}_all_time.png', dpi=300)
    fig.savefig(f'Figures\\{date}_all_time.svg')
    
# %% plot raw data on a plot together for each date, colored by velocity with error bands

for date, info in test_info.groupby('date'):
    i = 0

    fig, axs = plt.subplots()
    
    for v, info_v in info.groupby('velocity'):
        # Get all trials with this velocity
        trial_indices = info_v.index.tolist()
        
        if len(trial_indices) == 1:
            # Single trial - plot individual line
            idx = trial_indices[0]
            test = tests[idx]
            forward = test[test.travel > test.travel.shift(3)]
            
            plt.plot(forward.travel / info_v.iloc[0].height, forward.stress_adj,
                     c=col[i], label=f'{v} mm/s', linewidth=2)
        else:
            # Multiple trials - create error bands
            all_strains = []
            all_stresses = []
            
            # Collect data from all trials
            for idx in trial_indices:
                test = tests[idx]
                forward = test[test.travel > test.travel.shift(3)]
                
                strain = forward.travel / info_v.loc[idx].height
                stress = forward.stress_adj
                
                all_strains.append(strain.values)
                all_stresses.append(stress.values)
            
            # Find common strain range
            min_strain = max([s.min() for s in all_strains])
            max_strain = min([s.max() for s in all_strains])
            
            # Create common strain points
            strain_points = np.linspace(min_strain, max_strain, 100)
            
            # Interpolate each trial to common strain points
            interpolated_stresses = []
            for strain, stress in zip(all_strains, all_stresses):
                stress_interp = np.interp(strain_points, strain, stress)
                interpolated_stresses.append(stress_interp)
            
            # Calculate mean and std
            mean_stress = np.mean(interpolated_stresses, axis=0)
            std_stress = np.std(interpolated_stresses, axis=0)
            
            # Plot mean curve with error band
            plt.plot(strain_points, mean_stress, 
                     c=col[i], label=f'{v} mm/s (n={len(trial_indices)})', 
                     linewidth=2)
            plt.fill_between(strain_points, 
                           mean_stress - std_stress, 
                           mean_stress + std_stress, 
                           color=col[i], alpha=0.3)

        i += 1

    plt.ylabel('Stress [Pa]')
    
    plt.xlabel('Strain [-]')
    
    plt.title(f'{date} - Stress vs Strain by Impact Velocity (with Error Bands)')
    
    plt.legend(title='Impact Velocity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(f'Figures\\{date}_all.png', dpi=300)
    fig.savefig(f'Figures\\{date}_all.svg')

# %% plot max stress as a function of strain, separating by velocity

for date, info_date in test_info.groupby('date'):
    fig, axs = plt.subplots()

    i = 0
    
    for v, info_v in info_date.groupby('velocity'):
            plt.scatter(info_v.depth / info_v.height, info_v.max_stress,
                         c=col[i], ec='k', lw=0.5, label=f'{v} mm/s', s=100)
                    
            i += 1
    
    plt.legend(title='Impact Velocity')
    
    plt.ylabel('Maximum Stress [Pa]')
    
    plt.xlabel('Strain [-]')
    
    plt.title(f'{date} - Maximum Stress vs Strain')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(f'Figures\\{date}_max_stress-vs-v.png', dpi=300)
    fig.savefig(f'Figures\\{date}_max_stress-vs-v.svg')


# %% plot max stress as a function of velocity, for each strain

for date, info_date in test_info.groupby('date'):

    fig, axs = plt.subplots()
    
    for d, info in info_date.groupby('depth'):
                
        for v, info_v in info.groupby('velocity'):
                
            plt.scatter(info.velocity, info.max_stress, ec='k', lw=0.5, s=100)
    
        plt.ylabel('Maximum Stress [Pa]')
        
        plt.xlabel('Impact Velocity [mm/s]')
        
        plt.title(f'{date} - Maximum Stress vs Velocity (Strain = {d / info.height.unique()[0]:.2f})')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
        
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain.png', dpi=300)
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain.svg')

# %% plot total energy as a function of velocity

for date, info_date in test_info.groupby('date'):

    fig, axs = plt.subplots()
    
    plt.scatter(info_date.depth / info_date.height, info_date.e_tot * 1e-3, s=100, c='blue', alpha=0.7)
    
    plt.ylabel('Total Energy [J]')
    
    plt.xlabel('Strain [-]')
    
    plt.title(f'{date} - Total Energy Absorption vs Strain')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    fig.savefig(f'Figures\\{date}_e_tot-by_strain.png', dpi=300)
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
                         elinewidth=1, capsize=5, markersize=8)
            
        i += 1
    
        plt.ylabel('Maximum Stress [Pa]')
        
        plt.xlabel('Impact Velocity [mm/s]')
        
        plt.title(f'{date} - Maximum Stress vs Velocity (Strain = {d / info.height.unique()[0]:.2f})')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
        
    fig.savefig(f'Figures\\{date}_max_stress-vs-v-by_strain.png', dpi=300)
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
                             label=f'{v} mm/s', markersize=8)
            else:
                plt.errorbar(d / info_d.height.mean(),
                             (info_d.relax_10 / info_d.max_stress).mean(),
                             yerr=(info_d.relax_10 / info_d.max_stress).std(),
                             c=col[i], marker='o', capsize=5, lw=0, elinewidth=1, markersize=8)
                
            j += 1
        i += 1
    
    plt.ylabel('Stress Retention [%]')
    
    plt.xlabel('Strain [-]')
    
    plt.title(f'{date} - Stress Retention After Relaxation')
    
    plt.legend(title='Impact Velocity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(f'Figures\\{date}_stress_relax.png', dpi=300)
    fig.savefig(f'Figures\\{date}_stress_relax.svg')
    
# %% plot stress retention by composition (COMMENTED OUT - CAUSING ERRORS)

# for (g_wt, p_wt), info in test_info.groupby(['gel_wt', 'particle_wt']):
    
#     fig, axs = plt.subplots()
    
#     i = 0
    
#     for v, info_v in info.groupby('velocity'):
                
#         j = 0
        
#         for d, info_d in info_v.groupby('depth'):
#             if not j:
#                 plt.errorbar(d / info_d.height.mean(),
#                              (info_d.relax_10 / info_d.max_stress).mean(),
#                              yerr=(info_d.relax_10 / info_d.max_stress).std(),
#                              c=col[i], marker='o', capsize=5, lw=0, elinewidth=1,
#                              label=f'{v} mm/s', markersize=8)
#             else:
#                 plt.errorbar(d / info_d.height.mean(),
#                              (info_d.relax_10 / info_d.max_stress).mean(),
#                              yerr=(info_d.relax_10 / info_d.max_stress).std(),
#                              c=col[i], marker='o', capsize=5, lw=0, elinewidth=1, markersize=8)
                
#             j += 1
#         i += 1
    
#     plt.ylabel('Stress Retention [%]')
    
#     plt.xlabel('Strain [-]')
    
#     plt.title(f'Gel: {g_wt}g, Particles: {p_wt}g - Stress Retention by Composition')
    
#     plt.legend(title='Impact Velocity')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     fig.savefig(f'Figures\\{g_wt}_bovine_{p_wt}_cs_stress_relax.png', dpi=300)
#     fig.savefig(f'Figures\\{g_wt}_bovine_{p_wt}_cs_stress_relax.svg')

# %% plot stress-strain curves combined across all days for same compositions (SINGLE LAYER ONLY)

print("\n" + "="*50)
print("=== COMPOSITION-BASED PLOTS (SINGLE LAYER) ===")
print("="*50)
print(f"Total number of trials in test_info: {len(test_info)}")
print(f"Available columns: {list(test_info.columns)}")

# Filter out bilayer hydrogels (those with gel2 and particle2 columns filled)
single_layer_info = test_info.copy()
if 'gel2' in test_info.columns and 'particle2' in test_info.columns:
    # Remove rows where gel2 or particle2 are not empty (bilayer hydrogels)
    single_layer_info = test_info[
        (test_info['gel2'].isna() | (test_info['gel2'] == '')) & 
        (test_info['particle2'].isna() | (test_info['particle2'] == ''))
    ]
    print(f"Filtered to single layer hydrogels: {len(single_layer_info)} trials")

# Group by composition (gelatin type, gelatin weight, particle type, particle weight)
for (gel, gel_wt, particle, particle_wt), info in single_layer_info.groupby(['gel', 'gel_wt', 'particle', 'particle_wt']):
    
    print(f"Found composition: {gel} {gel_wt}%, {particle} {particle_wt}% - {len(info)} trials")
    
    # Only proceed if we have multiple trials across different dates
    if len(info) > 1:
        fig, axs = plt.subplots()
        
        # Group by velocity for this composition
        i = 0
        for v, info_v in info.groupby('velocity'):
            trial_indices = info_v.index.tolist()
            
            if len(trial_indices) == 1:
                # Single trial - plot individual line
                idx = trial_indices[0]
                test = tests[idx]
                forward = test[test.travel > test.travel.shift(3)]
                
                plt.plot(forward.travel / info_v.iloc[0].height, forward.stress_adj,
                         c=col[i], label=f'{v} mm/s', linewidth=2)
            else:
                # Multiple trials - create error bands for this velocity
                all_strains = []
                all_stresses = []
                
                # Collect data from all trials with this velocity
                for idx in trial_indices:
                    test = tests[idx]
                    forward = test[test.travel > test.travel.shift(3)]
                    
                    strain = forward.travel / info_v.loc[idx].height
                    stress = forward.stress_adj
                    
                    all_strains.append(strain.values)
                    all_stresses.append(stress.values)
                
                # Find common strain range for this velocity
                min_strain = max([s.min() for s in all_strains])
                max_strain = min([s.max() for s in all_strains])
                
                # Create common strain points
                strain_points = np.linspace(min_strain, max_strain, 100)
                
                # Interpolate each trial to common strain points
                interpolated_stresses = []
                for strain, stress in zip(all_strains, all_stresses):
                    stress_interp = np.interp(strain_points, strain, stress)
                    interpolated_stresses.append(stress_interp)
                
                # Calculate mean and std for this velocity
                mean_stress = np.mean(interpolated_stresses, axis=0)
                std_stress = np.std(interpolated_stresses, axis=0)
                
                # Plot mean curve with error band for this velocity
                plt.plot(strain_points, mean_stress, 
                         c=col[i], label=f'{v} mm/s (n={len(trial_indices)})', 
                         linewidth=2)
                plt.fill_between(strain_points, 
                               mean_stress - std_stress, 
                               mean_stress + std_stress, 
                               color=col[i], alpha=0.3)
            
            i += 1
        
        plt.ylabel('Stress [Pa]')
        plt.xlabel('Strain [-]')
        plt.title(f'Mixture: {gel.title()} {gel_wt}%, {particle.upper()} {particle_wt}%\nAll Trials Across All Dates')
        plt.legend(title='Impact Velocity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with descriptive filename
        filename = f'{gel}_{gel_wt}_{particle}_{particle_wt}_all_trials_combined'
        fig.savefig(f'Figures\\{filename}.png', dpi=300)
        fig.savefig(f'Figures\\{filename}.svg')
        plt.show()  # Display the plot
        plt.close(fig)

# %% plot stress-strain curves for bilayer hydrogels (separate analysis)

print("\n" + "="*50)
print("=== BILAYER HYDROGEL PLOTS ===")
print("="*50)

# Filter for bilayer hydrogels (those with gel2 and particle2 columns filled)
bilayer_info = test_info.copy()
if 'gel2' in test_info.columns and 'particle2' in test_info.columns:
    # Keep rows where gel2 or particle2 are not empty (bilayer hydrogels)
    bilayer_info = test_info[
        ~((test_info['gel2'].isna() | (test_info['gel2'] == '')) & 
          (test_info['particle2'].isna() | (test_info['particle2'] == '')))
    ]
    print(f"Found bilayer hydrogels: {len(bilayer_info)} trials")
    
    if len(bilayer_info) > 0:
        # Group by bilayer composition
        for (gel1, gel_wt1, particle1, particle_wt1, gel2, gel_wt2, particle2, particle_wt2), info in bilayer_info.groupby(['gel', 'gel_wt', 'particle', 'particle_wt', 'gel2', 'gel_wt2', 'particle2', 'particle_wt2']):
            
            print(f"Found bilayer: Layer1: {gel1} {gel_wt1}%, {particle1} {particle_wt1}% | Layer2: {gel2} {gel_wt2}%, {particle2} {particle_wt2}% - {len(info)} trials")
            
            if len(info) > 1:
                fig, axs = plt.subplots()
                
                # Group by velocity for this bilayer composition
                i = 0
                for v, info_v in info.groupby('velocity'):
                    trial_indices = info_v.index.tolist()
                    
                    if len(trial_indices) == 1:
                        # Single trial - plot individual line
                        idx = trial_indices[0]
                        test = tests[idx]
                        forward = test[test.travel > test.travel.shift(3)]
                        
                        plt.plot(forward.travel / info_v.iloc[0].height, forward.stress_adj,
                                 c=col[i], label=f'{v} mm/s', linewidth=2)
                    else:
                        # Multiple trials - create error bands for this velocity
                        all_strains = []
                        all_stresses = []
                        
                        # Collect data from all trials with this velocity
                        for idx in trial_indices:
                            test = tests[idx]
                            forward = test[test.travel > test.travel.shift(3)]
                            
                            strain = forward.travel / info_v.loc[idx].height
                            stress = forward.stress_adj
                            
                            all_strains.append(strain.values)
                            all_stresses.append(stress.values)
                        
                        # Find common strain range for this velocity
                        min_strain = max([s.min() for s in all_strains])
                        max_strain = min([s.max() for s in all_strains])
                        
                        # Create common strain points
                        strain_points = np.linspace(min_strain, max_strain, 100)
                        
                        # Interpolate each trial to common strain points
                        interpolated_stresses = []
                        for strain, stress in zip(all_strains, all_stresses):
                            stress_interp = np.interp(strain_points, strain, stress)
                            interpolated_stresses.append(stress_interp)
                        
                        # Calculate mean and std for this velocity
                        mean_stress = np.mean(interpolated_stresses, axis=0)
                        std_stress = np.std(interpolated_stresses, axis=0)
                        
                        # Plot mean curve with error band for this velocity
                        plt.plot(strain_points, mean_stress, 
                                 c=col[i], label=f'{v} mm/s (n={len(trial_indices)})', 
                                 linewidth=2)
                        plt.fill_between(strain_points, 
                                       mean_stress - std_stress, 
                                       mean_stress + std_stress, 
                                       color=col[i], alpha=0.3)
                    
                    i += 1
                
                plt.ylabel('Stress [Pa]')
                plt.xlabel('Strain [-]')
                plt.title(f'Bilayer: L1-{gel1.title()} {gel_wt1}% {particle1.upper()} {particle_wt1}% | L2-{gel2.title()} {gel_wt2}% {particle2.upper()} {particle_wt2}%\nAll Trials Across All Dates')
                plt.legend(title='Impact Velocity')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with descriptive filename
                filename = f'bilayer_{gel1}_{gel_wt1}_{particle1}_{particle_wt1}_{gel2}_{gel_wt2}_{particle2}_{particle_wt2}_combined'
                fig.savefig(f'Figures\\{filename}.png', dpi=300)
                fig.savefig(f'Figures\\{filename}.svg')
                plt.show()  # Display the plot
                plt.close(fig)
else:
    print("No bilayer hydrogel data found (gel2/particle2 columns not present)")

# %% plot maximum stress vs strain for same compositions across all days

for (gel, gel_wt, particle, particle_wt), info in single_layer_info.groupby(['gel', 'gel_wt', 'particle', 'particle_wt']):
    
    if len(info) > 1:
        fig, axs = plt.subplots()
        
        # Group by velocity for this composition
        i = 0
        for v, info_v in info.groupby('velocity'):
            # Calculate mean and std for max stress at each strain level
            for d, info_d in info_v.groupby('depth'):
                if len(info_d) > 1:
                    # Multiple trials at same strain and velocity
                    mean_max_stress = info_d.max_stress.mean()
                    std_max_stress = info_d.max_stress.std()
                    strain = d / info_d.height.mean()
                    
                    plt.errorbar(strain, mean_max_stress,
                               yerr=std_max_stress,
                               c=col[i], marker='o', capsize=5, 
                               markersize=8, label=f'{v} mm/s (n={len(info_d)})')
                else:
                    # Single trial
                    strain = d / info_d.height.mean()
                    plt.scatter(strain, info_d.max_stress.iloc[0],
                               c=col[i], marker='o', s=100, label=f'{v} mm/s')
            
            i += 1
        
        plt.ylabel('Maximum Stress [Pa]')
        plt.xlabel('Strain [-]')
        plt.title(f'Mixture: {gel.title()} {gel_wt}%, {particle.upper()} {particle_wt}%\nMax Stress vs Strain - All Dates')
        plt.legend(title='Impact Velocity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{gel}_{gel_wt}_{particle}_{particle_wt}_max_stress_all_trials'
        fig.savefig(f'Figures\\{filename}.png', dpi=300)
        fig.savefig(f'Figures\\{filename}.svg')
        plt.show()  # Display the plot
        plt.close(fig)

# %% plot stress retention for same compositions across all days

for (gel, gel_wt, particle, particle_wt), info in single_layer_info.groupby(['gel', 'gel_wt', 'particle', 'particle_wt']):
    
    if len(info) > 1:
        fig, axs = plt.subplots()
        
        # Group by velocity for this composition
        i = 0
        for v, info_v in info.groupby('velocity'):
            j = 0
            
            for d, info_d in info_v.groupby('depth'):
                if len(info_d) > 1:
                    # Multiple trials - calculate mean and std of stress retention
                    stress_retention = info_d.relax_10 / info_d.max_stress
                    mean_retention = stress_retention.mean()
                    std_retention = stress_retention.std()
                    strain = d / info_d.height.mean()
                    
                    if j == 0:
                        plt.errorbar(strain, mean_retention,
                                   yerr=std_retention,
                                   c=col[i], marker='o', capsize=5, 
                                   markersize=8, label=f'{v} mm/s (n={len(info_d)})')
                    else:
                        plt.errorbar(strain, mean_retention,
                                   yerr=std_retention,
                                   c=col[i], marker='o', capsize=5, markersize=8)
                else:
                    # Single trial
                    strain = d / info_d.height.mean()
                    retention = info_d.relax_10.iloc[0] / info_d.max_stress.iloc[0]
                    
                    if j == 0:
                        plt.scatter(strain, retention,
                                  c=col[i], marker='o', s=100, label=f'{v} mm/s')
                    else:
                        plt.scatter(strain, retention,
                                  c=col[i], marker='o', s=100)
                
                j += 1
            
            i += 1
        
        plt.ylabel('Stress Retention [%]')
        plt.xlabel('Strain [-]')
        plt.title(f'Mixture: {gel.title()} {gel_wt}%, {particle.upper()} {particle_wt}%\nStress Retention - All Dates')
        plt.legend(title='Impact Velocity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'{gel}_{gel_wt}_{particle}_{particle_wt}_retention_all_trials'
        fig.savefig(f'Figures\\{filename}.png', dpi=300)
        fig.savefig(f'Figures\\{filename}.svg')
        plt.show()  # Display the plot
        plt.close(fig)

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