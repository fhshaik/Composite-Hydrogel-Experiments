# -*- coding: utf-8 -*-
"""
Bovine Hydrogel Comparison Script
Compares: Bovine 20% + 10% Corn Starch vs Bovine 20% + 0% Corn Starch
Graph Type: Max Stress vs Strain with Error Bars
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
test_info_path = os.path.join(script_dir, 'test_info', '2025-08-08_test_info.csv')
saved_data_dir = os.path.join(script_dir, 'saved_data')

# Load data
test_info = pd.read_csv(test_info_path)
with open(os.path.join(saved_data_dir, 'tests.pkl'), 'rb') as f:
    tests = pickle.load(f)

# Define compositions to compare
composition1 = {'gel': 'bovine', 'gel_wt': 20, 'particle': 'cs', 'particle_wt': '10'}
composition2 = {'gel': 'bovine', 'gel_wt': 20, 'particle': 'na', 'particle_wt': '0'}

# Filter data for each composition (single layer)
def filter_composition(data, comp):
    """Filter data for specific composition (single layer only)"""
    mask = (
        (data['gel'] == comp['gel']) &
        (data['gel_wt'] == comp['gel_wt']) &
        (data['particle'] == comp['particle']) &
        (data['particle_wt'] == comp['particle_wt']) &
        # Exclude bilayer samples
        ((data['gel2'].isna()) | (data['gel2'] == ''))
    )
    return data[mask]

comp1_data = filter_composition(test_info, composition1)
comp2_data = filter_composition(test_info, composition2)

print(f"Found {len(comp1_data)} trials for Bovine 20% + 10% CS")
print(f"Found {len(comp2_data)} trials for Bovine 20% + 0% CS")

# Create comparison plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Colors and markers for each composition
colors = ['#1f77b4', '#d62728']  # Blue for composite (10% CS), Red for non-composite (0% CS)
markers = ['o', 's']  # Circle, Square
labels = ['Bovine 20% + 10% CS', 'Bovine 20% + 0% CS']

# Process single-layer compositions for both plots
for i, (comp_data, color, marker, label) in enumerate(zip([comp1_data, comp2_data], colors, markers, labels)):
    
    if len(comp_data) == 0:
        print(f"No data found for {label}")
        continue
    
    # Filter velocities <= 10 mm/s
    comp_data = comp_data[comp_data['velocity'] <= 10]
    
    # Track if we've already labeled this composition
    first_point_for_composition = True
    
    # Calculate strain for each data point
    comp_data['strain'] = comp_data['depth'] / comp_data['height']
    
    # Filter by strain
    comp_data = comp_data[comp_data['strain'] <= 0.4]
    
    # Group by strain level
    for strain_level, strain_data in comp_data.groupby('strain'):
        # Only label the first point for each composition
        if first_point_for_composition:
            plot_label = label
            first_point_for_composition = False
        else:
            plot_label = None
        
        if len(strain_data) > 1:
            # Multiple trials - calculate mean and std
            mean_stress = strain_data['max_stress'].mean()
            std_stress = strain_data['max_stress'].std()
            
            ax1.errorbar(strain_level, mean_stress,
                       yerr=std_stress,
                       c=color, marker=marker, capsize=5,
                       markersize=8, label=plot_label, alpha=0.8)
        else:
            # Single trial
            ax1.scatter(strain_level, strain_data['max_stress'].iloc[0],
                       c=color, marker=marker, s=100, label=plot_label, alpha=0.8)
        
        # Energy absorption plot
        if len(strain_data) > 1:
            # Multiple trials - calculate mean and std for energy
            mean_energy = strain_data['e_tot'].mean()
            std_energy = strain_data['e_tot'].std()
            
            ax2.errorbar(strain_level, mean_energy,
                       yerr=std_energy,
                       c=color, marker=marker, capsize=5,
                       markersize=8, label=plot_label, alpha=0.8)
        else:
            # Single trial
            ax2.scatter(strain_level, strain_data['e_tot'].iloc[0],
                       c=color, marker=marker, s=100, label=plot_label, alpha=0.8)

# Customize plots
ax1.set_xlabel('Strain [-]', fontsize=14, fontweight='bold')
ax1.set_ylabel('Maximum Stress [Pa]', fontsize=14, fontweight='bold')
ax1.set_title('Bovine Hydrogel Comparison: Max Stress vs Strain\n(20% Bovine + 10% CS vs 20% Bovine + 0% CS)', 
              fontsize=16, fontweight='bold')
ax1.legend(title='Composition', loc='upper left', fontsize=12, title_fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=12)

ax2.set_xlabel('Strain [-]', fontsize=14, fontweight='bold')
ax2.set_ylabel('Energy Absorption [J/m³]', fontsize=14, fontweight='bold')
ax2.set_title('Bovine Hydrogel Comparison: Energy Absorption vs Strain\n(20% Bovine + 10% CS vs 20% Bovine + 0% CS)', 
              fontsize=16, fontweight='bold')
ax2.legend(title='Composition', loc='upper left', fontsize=12, title_fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save plots
output_filename = 'bovine_comparison_max_stress_and_energy_vs_strain'
plt.savefig(f'Figures/{output_filename}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'Figures/{output_filename}.svg', bbox_inches='tight')

# Show plot if desired
SHOW_PLOTS = 1  # Set to 1 to display plots
if SHOW_PLOTS:
    plt.show()

plt.close()

print(f"\nComparison plots saved as: {output_filename}.png and .svg")
print("\nSummary:")
print(f"- Bovine 20% + 10% CS: {len(comp1_data)} trials")
print(f"- Bovine 20% + 0% CS: {len(comp2_data)} trials")
print("- Filtered: velocities ≤ 10 mm/s, strains ≤ 0.4")
print("- Error bars show standard deviation for multiple trials") 