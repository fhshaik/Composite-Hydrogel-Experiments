import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

def calculate_volume_pixels_only():
    """
    Calculate volume using only pixel measurements, avoiding scaling issues.
    """
    
    print("=" * 80)
    print("VOLUME CALCULATION USING PIXEL MEASUREMENTS ONLY")
    print("=" * 80)
    
    # Path to volume data
    volume_data_path = "C2225_analysis/volume_analysis/combined_volume_results.csv"
    
    if not os.path.exists(volume_data_path):
        print(f"ERROR: Volume data file not found: {volume_data_path}")
        return None
    
    # Read volume data
    print(f"Reading data from: {volume_data_path}")
    df = pd.read_csv(volume_data_path)
    
    print(f"Found {len(df)} data points")
    
    # Get pixel-to-mm scale for reference (but we won't use it for calculations)
    pixel_to_mm_scale = df['pixel_to_mm_scale'].iloc[0]
    print(f"\nPixel-to-mm scale: {pixel_to_mm_scale:.6f} mm/pixel")
    print(f"Note: We will NOT use this scale for volume calculations")
    
    # Convert mm measurements back to pixels
    print(f"\nCONVERTING MEASUREMENTS TO PIXELS:")
    print(f"=" * 50)
    
    df['height_pixels'] = df['height_mm'] / pixel_to_mm_scale
    df['radius_pixels'] = df['avg_radius_mm'] / pixel_to_mm_scale
    
    print(f"Height range: {df['height_pixels'].min():.1f} to {df['height_pixels'].max():.1f} pixels")
    print(f"Radius range: {df['radius_pixels'].min():.1f} to {df['radius_pixels'].max():.1f} pixels")
    
    # Apply smoothing to pixel measurements
    window_length = min(101, len(df) // 2 * 2 + 1)
    if window_length < 5:
        window_length = 5
    
    print(f"\nApplying smoothing with window length: {window_length}")
    df['height_pixels_smoothed'] = savgol_filter(df['height_pixels'], window_length, 3)
    df['radius_pixels_smoothed'] = savgol_filter(df['radius_pixels'], window_length, 3)
    
    # Calculate volume in pixels³
    print(f"\nCALCULATING VOLUME IN PIXELS³:")
    print(f"=" * 50)
    
    # Volume = π × radius² × height (all in pixels)
    df['volume_pixels3'] = np.pi * df['radius_pixels_smoothed']**2 * df['height_pixels_smoothed']
    
    # Also calculate volume using raw (unsmoothed) pixel measurements for comparison
    df['volume_pixels3_raw'] = np.pi * df['radius_pixels']**2 * df['height_pixels']
    
    # Print statistics
    print(f"\nVOLUME STATISTICS (PIXELS³):")
    print(f"=" * 50)
    print(f"Volume from raw pixels:")
    print(f"  Range: {df['volume_pixels3_raw'].min():.0f} to {df['volume_pixels3_raw'].max():.0f} pixels³")
    print(f"  Mean: {df['volume_pixels3_raw'].mean():.0f} pixels³")
    print(f"  Std: {df['volume_pixels3_raw'].std():.0f} pixels³")
    
    print(f"\nVolume from smoothed pixels:")
    print(f"  Range: {df['volume_pixels3'].min():.0f} to {df['volume_pixels3'].max():.0f} pixels³")
    print(f"  Mean: {df['volume_pixels3'].mean():.0f} pixels³")
    print(f"  Std: {df['volume_pixels3'].std():.0f} pixels³")
    
    # Compare initial and final volumes
    initial_volume_raw = df['volume_pixels3_raw'].iloc[0]
    final_volume_raw = df['volume_pixels3_raw'].iloc[-1]
    initial_volume_smoothed = df['volume_pixels3'].iloc[0]
    final_volume_smoothed = df['volume_pixels3'].iloc[-1]
    
    print(f"\nVOLUME COMPARISON (PIXELS³):")
    print(f"=" * 50)
    print(f"Raw pixel volume:")
    print(f"  Initial: {initial_volume_raw:.0f} pixels³")
    print(f"  Final: {final_volume_raw:.0f} pixels³")
    print(f"  Change: {((final_volume_raw - initial_volume_raw) / initial_volume_raw * 100):.1f}%")
    
    print(f"\nSmoothed pixel volume:")
    print(f"  Initial: {initial_volume_smoothed:.0f} pixels³")
    print(f"  Final: {final_volume_smoothed:.0f} pixels³")
    print(f"  Change: {((final_volume_smoothed - initial_volume_smoothed) / initial_volume_smoothed * 100):.1f}%")
    
    # Define fps for time calculations
    fps = 120  # frames per second
    
    # Identify compression period using height over time
    print(f"\nIDENTIFYING COMPRESSION PERIOD:")
    print(f"=" * 50)
    
    # Find the compression period by looking for continuous height decrease
    height_diff = df['height_pixels_smoothed'].diff()
    
    # Find start of compression (first significant height decrease)
    compression_start = 0
    for i in range(1, len(height_diff)):
        if height_diff.iloc[i] < -0.5:  # Significant height decrease
            compression_start = i
            break
    
    # Find end of compression (height stabilizes or starts increasing)
    compression_end = len(df) - 1
    for i in range(compression_start + 10, len(height_diff)):  # Start looking after some compression
        if height_diff.iloc[i] > 0.1:  # Height starts increasing (end of compression)
            compression_end = i
            break
    
    # Ensure we have a reasonable compression period
    if compression_end - compression_start < 10:
        print(f"Warning: Compression period too short ({compression_end - compression_start} frames)")
        print(f"Using last 50% of data as compression period")
        compression_start = len(df) // 2
        compression_end = len(df) - 1
    
    print(f"Compression start frame: {compression_start}")
    print(f"Compression end frame: {compression_end}")
    print(f"Compression duration: {compression_end - compression_start} frames")
    print(f"Compression duration: {(compression_end - compression_start) / fps:.2f} seconds")
    
    # Extract compression period data
    df_compression = df.iloc[compression_start:compression_end+1].copy()
    print(f"Data points during compression: {len(df_compression)}")
    
    # Add time_seconds to compression data
    df_compression['time_seconds'] = df_compression['image_index'] / fps
    
    # Calculate Poisson ratio from pixel volume (COMPRESSION PERIOD ONLY)
    print(f"\nCALCULATING POISSON RATIO FROM PIXEL VOLUME (COMPRESSION PERIOD):")
    print(f"=" * 50)
    print(f"Using CORRECTED formula: ν = (ε - ΔV/V)/(2ε)")
    print(f"Instead of the incorrect: ν = (1 - ΔV/V)/2")
    
    # Use compression start values as initial values
    initial_height_pixels = df_compression['height_pixels_smoothed'].iloc[0]
    initial_volume_pixels = df_compression['volume_pixels3'].iloc[0]
    
    # Calculate strain and volume change (all in pixels) for compression period only
    df_compression['engineering_strain_pixels'] = (df_compression['height_pixels_smoothed'] - initial_height_pixels) / initial_height_pixels
    df_compression['volume_change_ratio_pixels'] = (df_compression['volume_pixels3'] - initial_volume_pixels) / initial_volume_pixels
    
    # Calculate Poisson ratio with and without filtering (COMPRESSION PERIOD ONLY)
    poisson_ratios_filtered = []
    poisson_ratios_unfiltered = []
    valid_indices = []
    all_indices = []
    
    for i, (vol_change, strain) in enumerate(zip(df_compression['volume_change_ratio_pixels'], df_compression['engineering_strain_pixels'])):
        if abs(strain) > 1e-6:  # Avoid division by zero
            # CORRECTED formula: ν = (ε - ΔV/V)/(2ε)
            # Instead of the incorrect: ν = (1 - ΔV/V)/2
            nu = (strain - vol_change) / (2 * strain)
            poisson_ratios_unfiltered.append(nu)
            all_indices.append(i)
            
            # Ensure physically reasonable values (0 ≤ ν ≤ 0.5 for most materials)
            if 0 <= nu <= 0.5:
                poisson_ratios_filtered.append(nu)
                valid_indices.append(i)
    
    print(f"  Total data points during compression: {len(df_compression)}")
    print(f"  Unfiltered Poisson ratios: {len(poisson_ratios_unfiltered)} ({len(poisson_ratios_unfiltered)/len(df_compression)*100:.1f}%)")
    print(f"  Filtered Poisson ratios: {len(poisson_ratios_filtered)} ({len(poisson_ratios_filtered)/len(df_compression)*100:.1f}%)")
    
    if len(poisson_ratios_unfiltered) > 0:
        poisson_mean_unfiltered = np.mean(poisson_ratios_unfiltered)
        poisson_std_unfiltered = np.std(poisson_ratios_unfiltered)
        print(f"  Unfiltered - Mean: {poisson_mean_unfiltered:.3f} ± {poisson_std_unfiltered:.3f}")
        print(f"  Unfiltered - Range: {np.min(poisson_ratios_unfiltered):.3f} to {np.max(poisson_ratios_unfiltered):.3f}")
    
    if len(poisson_ratios_filtered) > 0:
        poisson_mean_filtered = np.mean(poisson_ratios_filtered)
        poisson_std_filtered = np.std(poisson_ratios_filtered)
        print(f"  Filtered - Mean: {poisson_mean_filtered:.3f} ± {poisson_std_filtered:.3f}")
        print(f"  Filtered - Range: {np.min(poisson_ratios_filtered):.3f} to {np.max(poisson_ratios_filtered):.3f}")
        
        # Apply additional smoothing to Poisson ratio to reduce jumpiness
        poisson_window_length = min(21, len(poisson_ratios_filtered) // 2 * 2 + 1)
        if poisson_window_length < 5:
            poisson_window_length = 5
        
        poisson_ratios_filtered_smoothed = savgol_filter(poisson_ratios_filtered, poisson_window_length, 3)
        print(f"  Applied additional smoothing with window length: {poisson_window_length}")
        print(f"  Smoothed - Mean: {np.mean(poisson_ratios_filtered_smoothed):.3f} ± {np.std(poisson_ratios_filtered_smoothed):.3f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Volume Calculation Using Pixel Measurements Only\n(Compression Period Analysis)', fontsize=16, fontweight='bold')
    
    # Create separate figures for Poisson ratio plots
    fig2, ax_poisson = plt.subplots(1, 1, figsize=(16, 10))
    fig3, ax_strain = plt.subplots(1, 1, figsize=(20, 12))  # Large figure for strain vs Poisson ratio
    
    # Convert frames to seconds
    df['time_seconds'] = df['image_index'] / fps
    
    # Plot 1: Volume comparison (pixels³)
    axes[0, 0].plot(df['image_index'], df['volume_pixels3_raw'], 'red', linewidth=2, alpha=0.7, label='Raw Pixel Volume')
    axes[0, 0].plot(df['image_index'], df['volume_pixels3'], 'blue', linewidth=2, alpha=0.9, label='Smoothed Pixel Volume')
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Volume (pixels³)')
    axes[0, 0].set_title('Volume Comparison (Pixels³)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Volume change over time (pixels) - COMPRESSION PERIOD ONLY
    axes[0, 1].plot(df_compression['image_index'], df_compression['volume_change_ratio_pixels']*100, 'orange', linewidth=2, alpha=0.8)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Volume Change (%)')
    axes[0, 1].set_title('Volume Change Over Time (Compression Period)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Height and radius in pixels
    ax3 = axes[0, 2]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(df['image_index'], df['height_pixels_smoothed'], 'blue', linewidth=2, alpha=0.8, label='Height')
    line2 = ax3_twin.plot(df['image_index'], df['radius_pixels_smoothed'], 'green', linewidth=2, alpha=0.8, label='Radius')
    
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Height (pixels)', color='blue')
    ax3_twin.set_ylabel('Radius (pixels)', color='green')
    ax3.set_title('Smoothed Height and Radius (Pixels)')
    ax3.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    # Plot 4: Strain over time (pixels) - COMPRESSION PERIOD ONLY
    axes[1, 0].plot(df_compression['image_index'], df_compression['engineering_strain_pixels']*100, 'brown', linewidth=2, alpha=0.8)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Strain (%)')
    axes[1, 0].set_title('Engineering Strain Over Time (Compression Period)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Compression period identification
    axes[1, 1].plot(df['image_index'], df['height_pixels_smoothed'], 'blue', linewidth=2, alpha=0.8, label='Height')
    axes[1, 1].axvspan(compression_start, compression_end, alpha=0.3, color='red', label='Compression Period')
    axes[1, 1].axvline(x=compression_start, color='red', linestyle='--', alpha=0.7, label=f'Start: {compression_start}')
    axes[1, 1].axvline(x=compression_end, color='red', linestyle='--', alpha=0.7, label=f'End: {compression_end}')
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].set_title('Compression Period Identification')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Poisson ratio comparison (COMPRESSION PERIOD ONLY)
    if len(poisson_ratios_unfiltered) > 0:
        strain_values_unfiltered = [df_compression['engineering_strain_pixels'].iloc[i] for i in all_indices]
        strain_values_filtered = [df_compression['engineering_strain_pixels'].iloc[i] for i in valid_indices]
        
        # Make strains positive for plotting
        strain_values_unfiltered_positive = [-s for s in strain_values_unfiltered]  # Negative to make positive
        strain_values_filtered_positive = [-s for s in strain_values_filtered]  # Negative to make positive
        
        # Plot unfiltered data
        axes[1, 2].plot(strain_values_unfiltered_positive, poisson_ratios_unfiltered, 'red', linewidth=1, alpha=0.6, label='Unfiltered')
        if len(poisson_ratios_filtered) > 0:
            axes[1, 2].plot(strain_values_filtered_positive, poisson_ratios_filtered, 'purple', linewidth=1, alpha=0.6, label='Filtered (0≤ν≤0.5)')
            axes[1, 2].axhline(y=poisson_mean_filtered, color='purple', linestyle='--', alpha=0.7, 
                              label=f'Filtered Mean = {poisson_mean_filtered:.3f}')
            
            # Plot smoothed Poisson ratio
            axes[1, 2].plot(strain_values_filtered_positive, poisson_ratios_filtered_smoothed, 'blue', linewidth=3, alpha=0.9, label='Smoothed Filtered')
            axes[1, 2].axhline(y=np.mean(poisson_ratios_filtered_smoothed), color='blue', linestyle='--', alpha=0.7, 
                              label=f'Smoothed Mean = {np.mean(poisson_ratios_filtered_smoothed):.3f}')
        
        axes[1, 2].axhline(y=poisson_mean_unfiltered, color='red', linestyle='--', alpha=0.7, 
                          label=f'Unfiltered Mean = {poisson_mean_unfiltered:.3f}')
        axes[1, 2].set_xlabel('Strain (|ε|)')
        axes[1, 2].set_ylabel('Poisson Ratio (ν)')
        axes[1, 2].set_title('Poisson Ratio: Filtered vs Unfiltered\n(Compression Period Only, Corrected Formula)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        # Create large dedicated Poisson ratio vs strain plot
        ax_strain.plot(strain_values_unfiltered_positive, poisson_ratios_unfiltered, 'red', linewidth=2, alpha=0.7, label='Unfiltered Poisson Ratio')
        ax_strain.axhline(y=poisson_mean_unfiltered, color='red', linestyle='--', alpha=0.7, 
                         label=f'Unfiltered Mean = {poisson_mean_unfiltered:.3f}')
        
        if len(poisson_ratios_filtered) > 0:
            ax_strain.plot(strain_values_filtered_positive, poisson_ratios_filtered, 'purple', linewidth=2, alpha=0.7, label='Filtered Poisson Ratio (0≤ν≤0.5)')
            ax_strain.axhline(y=poisson_mean_filtered, color='purple', linestyle='--', alpha=0.7, 
                            label=f'Filtered Mean = {poisson_mean_filtered:.3f}')
            
            # Plot smoothed Poisson ratio with thicker line
            ax_strain.plot(strain_values_filtered_positive, poisson_ratios_filtered_smoothed, 'blue', linewidth=4, alpha=0.9, label='Smoothed Filtered Poisson Ratio')
            smoothed_mean = np.mean(poisson_ratios_filtered_smoothed)
            smoothed_std = np.std(poisson_ratios_filtered_smoothed)
            ax_strain.axhline(y=smoothed_mean, color='blue', linestyle='--', alpha=0.7, 
                            label=f'Smoothed Mean = {smoothed_mean:.3f}')
            
            # Add shaded region for standard deviation
            ax_strain.fill_between(strain_values_filtered_positive, 
                                   [smoothed_mean - smoothed_std] * len(poisson_ratios_filtered),
                                   [smoothed_mean + smoothed_std] * len(poisson_ratios_filtered),
                                   alpha=0.2, color='blue', label=f'Smoothed ±1σ = {smoothed_std:.3f}')
        
        ax_strain.set_xlabel('Strain (|ε|)', fontsize=16)
        ax_strain.set_ylabel('Poisson Ratio (ν)', fontsize=16)
        ax_strain.set_title('Poisson Ratio vs Strain\n(Compression Period Only, Corrected Formula)', fontsize=20, fontweight='bold')
        ax_strain.grid(True, alpha=0.3)
        ax_strain.legend(fontsize=14)
        ax_strain.tick_params(axis='both', which='major', labelsize=14)
        
        # Set y-limits to focus on reasonable Poisson ratio range
        all_poisson_values = poisson_ratios_unfiltered + poisson_ratios_filtered
        min_poisson = max(0.0, np.min(all_poisson_values) - 0.05)
        max_poisson = min(0.6, np.max(all_poisson_values) + 0.05)
        ax_strain.set_ylim(min_poisson, max_poisson)
        
        # Create Poisson ratio over time plot (COMPRESSION PERIOD ONLY)
        time_indices_unfiltered = [df_compression.iloc[i]['time_seconds'] for i in all_indices]
        time_indices_filtered = [df_compression.iloc[i]['time_seconds'] for i in valid_indices]
        
        # Plot unfiltered data
        ax_poisson.plot(time_indices_unfiltered, poisson_ratios_unfiltered, 'red', linewidth=2, alpha=0.6, label='Unfiltered')
        ax_poisson.axhline(y=poisson_mean_unfiltered, color='red', linestyle='--', alpha=0.7, 
                          label=f'Unfiltered Mean = {poisson_mean_unfiltered:.3f}')
        
        # Plot filtered data
        if len(poisson_ratios_filtered) > 0:
            ax_poisson.plot(time_indices_filtered, poisson_ratios_filtered, 'purple', linewidth=1, alpha=0.6, label='Filtered (0≤ν≤0.5)')
            ax_poisson.axhline(y=poisson_mean_filtered, color='purple', linestyle='--', alpha=0.7, 
                              label=f'Filtered Mean = {poisson_mean_filtered:.3f}')
            
            # Plot smoothed Poisson ratio
            ax_poisson.plot(time_indices_filtered, poisson_ratios_filtered_smoothed, 'blue', linewidth=4, alpha=0.9, label='Smoothed Filtered')
            smoothed_mean = np.mean(poisson_ratios_filtered_smoothed)
            smoothed_std = np.std(poisson_ratios_filtered_smoothed)
            ax_poisson.axhline(y=smoothed_mean, color='blue', linestyle='--', alpha=0.7, 
                              label=f'Smoothed Mean = {smoothed_mean:.3f}')
            ax_poisson.fill_between(time_indices_filtered, 
                                   [smoothed_mean - smoothed_std] * len(poisson_ratios_filtered),
                                   [smoothed_mean + smoothed_std] * len(poisson_ratios_filtered),
                                   alpha=0.2, color='blue', label=f'Smoothed ±1σ = {smoothed_std:.3f}')
        
        ax_poisson.set_xlabel('Time (seconds)')
        ax_poisson.set_ylabel('Poisson Ratio (ν)')
        ax_poisson.set_title('Poisson Ratio Over Time: Filtered vs Unfiltered\n(Compression Period Only, Corrected Formula)', fontsize=16, fontweight='bold')
        ax_poisson.grid(True, alpha=0.3)
        ax_poisson.legend(fontsize=12)
        
        # Set y-limits to show both filtered and unfiltered data
        all_poisson_values = poisson_ratios_unfiltered + poisson_ratios_filtered
        min_poisson = max(-0.1, np.min(all_poisson_values) - 0.05)
        max_poisson = min(1.0, np.max(all_poisson_values) + 0.05)
        ax_poisson.set_ylim(min_poisson, max_poisson)
        
    else:
        axes[1, 2].text(0.5, 0.5, 'No valid Poisson ratios\n(volume change too large)', 
                       ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Poisson Ratio: Filtered vs Unfiltered\n(Compression Period Only, Corrected Formula)')
        
        # Show message in separate plot too
        ax_poisson.text(0.5, 0.5, 'No valid Poisson ratios\n(volume change too large)', 
                       ha='center', va='center', transform=ax_poisson.transAxes, fontsize=14)
        ax_poisson.set_title('Poisson Ratio Over Time: Filtered vs Unfiltered\n(Compression Period Only, Corrected Formula)')
    
    plt.tight_layout()
    
    # Save the plots
    output_path = "C2225_analysis/volume_analysis/volume_pixels_only_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMain plot saved to: {output_path}")
    
    # Save the Poisson ratio over time plot
    output_path2 = "C2225_analysis/volume_analysis/poisson_ratio_over_time_pixels.png"
    plt.figure(fig2.number)
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Poisson ratio over time plot saved to: {output_path2}")
    
    # Save the large Poisson ratio vs strain plot
    output_path3 = "C2225_analysis/volume_analysis/poisson_ratio_vs_strain_large.png"
    plt.figure(fig3.number)
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Large Poisson ratio vs strain plot saved to: {output_path3}")
    
    # Save the compression period data
    output_csv = "C2225_analysis/volume_analysis/volume_pixels_only_compression_data.csv"
    columns_to_save = [
        'image_index', 'height_pixels', 'height_pixels_smoothed', 
        'radius_pixels', 'radius_pixels_smoothed',
        'volume_pixels3_raw', 'volume_pixels3',
        'engineering_strain_pixels', 'volume_change_ratio_pixels'
    ]
    df_compression[columns_to_save].to_csv(output_csv, index=False)
    print(f"\nCompression period data saved to: {output_csv}")
    
    # Save Poisson ratio data if available (COMPRESSION PERIOD ONLY)
    if len(poisson_ratios_unfiltered) > 0:
        poisson_csv = "C2225_analysis/volume_analysis/poisson_ratio_pixels_only_compression_data.csv"
        poisson_data = pd.DataFrame({
            'image_index': [df_compression.iloc[i]['image_index'] for i in all_indices],
            'time_seconds': [df_compression.iloc[i]['time_seconds'] for i in all_indices],
            'strain': [df_compression['engineering_strain_pixels'].iloc[i] for i in all_indices],
            'poisson_ratio_unfiltered': poisson_ratios_unfiltered,
            'poisson_ratio_filtered': [poisson_ratios_filtered[valid_indices.index(i)] if i in valid_indices else np.nan for i in all_indices],
            'poisson_ratio_filtered_smoothed': [poisson_ratios_filtered_smoothed[valid_indices.index(i)] if i in valid_indices else np.nan for i in all_indices],
            'volume_change_ratio': [df_compression['volume_change_ratio_pixels'].iloc[i] for i in all_indices]
        })
        poisson_data.to_csv(poisson_csv, index=False)
        print(f"Poisson ratio data saved to: {poisson_csv}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"=" * 50)
    print(f"  ✓ Volume calculated using pixel measurements only")
    print(f"  ✓ No scaling conversions used")
    print(f"  ✓ Volume change during compression: {((df_compression['volume_pixels3'].iloc[-1] - df_compression['volume_pixels3'].iloc[0]) / df_compression['volume_pixels3'].iloc[0] * 100):.1f}%")
    print(f"  ✓ Height change during compression: {((df_compression['height_pixels_smoothed'].iloc[-1] - df_compression['height_pixels_smoothed'].iloc[0]) / df_compression['height_pixels_smoothed'].iloc[0] * 100):.1f}%")
    print(f"  ✓ Radius change during compression: {((df_compression['radius_pixels_smoothed'].iloc[-1] - df_compression['radius_pixels_smoothed'].iloc[0]) / df_compression['radius_pixels_smoothed'].iloc[0] * 100):.1f}%")
    
    if len(poisson_ratios_unfiltered) > 0:
        print(f"  ✓ Unfiltered Poisson ratios: {len(poisson_ratios_unfiltered)} ({len(poisson_ratios_unfiltered)/len(df_compression)*100:.1f}%)")
        print(f"  ✓ Unfiltered mean: {poisson_mean_unfiltered:.3f} ± {poisson_std_unfiltered:.3f}")
        
        if len(poisson_ratios_filtered) > 0:
            print(f"  ✓ Filtered Poisson ratios: {len(poisson_ratios_filtered)} ({len(poisson_ratios_filtered)/len(df_compression)*100:.1f}%)")
            print(f"  ✓ Filtered mean: {poisson_mean_filtered:.3f} ± {poisson_std_filtered:.3f}")
            print(f"  ✓ Smoothed mean: {np.mean(poisson_ratios_filtered_smoothed):.3f} ± {np.std(poisson_ratios_filtered_smoothed):.3f}")
        else:
            print(f"  ⚠️  No filtered Poisson ratios (all values outside 0≤ν≤0.5)")
    else:
        print(f"  ⚠️  No valid Poisson ratios (volume change too large)")
    
    # Additional analysis: Check if volume conservation is better in pixels (COMPRESSION PERIOD)
    print(f"\nVOLUME CONSERVATION ANALYSIS (COMPRESSION PERIOD):")
    print(f"=" * 50)
    volume_conservation_error = abs(df_compression['volume_change_ratio_pixels'].iloc[-1]) * 100
    print(f"  Volume conservation error during compression: {volume_conservation_error:.1f}%")
    
    if volume_conservation_error < 5:
        print(f"  ✓ Good volume conservation during compression (< 5%)")
    elif volume_conservation_error < 10:
        print(f"  ⚠️  Moderate volume conservation during compression (5-10%)")
    else:
        print(f"  ❌ Poor volume conservation during compression (> 10%)")
    
    plt.show()
    
    # Show the Poisson ratio over time plot separately
    plt.figure(fig2.number)
    plt.show()
    
    # Show the large Poisson ratio vs strain plot separately
    plt.figure(fig3.number)
    plt.show()
    
    return df

if __name__ == "__main__":
    results = calculate_volume_pixels_only()
    if results is not None:
        print(f"\nAnalysis complete!")
    else:
        print("\nAnalysis failed!")
