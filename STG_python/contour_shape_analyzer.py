# -*- coding: utf-8 -*-
"""
Script to analyze contour shapes from C2225 analysis folder
Extracts contour shape by tracing from middle vertical line to last white point
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import pandas as pd

def load_analysis_images(folder_path):
    """Load all processed images from the C2225 analysis folder"""
    if not os.path.exists(folder_path):
        print(f"Analysis folder not found: {folder_path}")
        return []
    
    image_files = []
    for file in os.listdir(folder_path):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, file))
    
    image_files.sort()
    print(f"Found {len(image_files)} images")
    return image_files

def detect_ground_and_press(image_path):
    """Detect ground and press positions using edge detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None
    
    height, width = img.shape
    
    # Crop out the top quarter to eliminate press top section
    crop_start = height // 4
    cropped_img = img[crop_start:, :]
    cropped_height = cropped_img.shape[0]
    
    # Apply Gaussian blur and edge detection
    blurred = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Connect horizontal edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find horizontal lines
    lines = cv2.HoughLinesP(horizontal_edges, 1, np.pi/180, threshold=30, 
                           minLineLength=width//4, maxLineGap=50)
    
    if lines is None:
        print("No horizontal lines detected")
        return None, None
    
    # Extract and filter horizontal line positions
    horizontal_y_positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 15:  # Nearly horizontal
            y_avg = (y1 + y2) // 2
            if y_avg > cropped_height * 0.1 and y_avg < cropped_height * 0.9:
                # Position-based filtering
                if y_avg < cropped_height * 0.3:  # Upper 30% - likely press
                    horizontal_y_positions.append(y_avg)
                elif y_avg > cropped_height * 0.7:  # Lower 30% - likely ground
                    horizontal_y_positions.append(y_avg)
    
    if len(horizontal_y_positions) < 2:
        print(f"Not enough horizontal lines found. Found: {len(horizontal_y_positions)}")
        return None, None
    
    # Remove duplicates
    unique_positions = []
    for pos in sorted(horizontal_y_positions):
        if not unique_positions or abs(pos - unique_positions[-1]) > 30:
            unique_positions.append(pos)
    
    horizontal_y_positions = unique_positions
    horizontal_y_positions.sort(reverse=True)  # Bottom to top
    
    # Ground is lowest, press is second lowest
    ground_y = horizontal_y_positions[0] if horizontal_y_positions else None
    press_y = horizontal_y_positions[1] if len(horizontal_y_positions) > 1 else None
    
    # Convert back to original image space and apply offsets
    if ground_y is not None:
        ground_y += crop_start
        ground_y -= 6  # Move ground line 6 pixels up
    if press_y is not None:
        press_y += crop_start
        press_y += 20  # Move press line 20 pixels down
    
    print(f"Ground position: {ground_y}, Press position: {press_y}")
    return ground_y, press_y

def extract_contour_from_image_both_sides(image_path, threshold=80, margin=100):
    """Extract contour shape from both left and right sides, scanning from center up and down"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None
    
    height, width = img.shape
    middle_x = width // 2
    
    # Detect ground and press to find the center region
    ground_y, press_y = detect_ground_and_press(image_path)
    
    if ground_y is None or press_y is None:
        print("Could not detect ground and press positions")
        return None, None
    
    # Calculate the center y-position (middle of puck)
    center_y = (ground_y + press_y) // 2
    print(f"Center y-position: {center_y} (ground: {ground_y}, press: {press_y})")
    
    # Initialize contour arrays
    right_contour_points = []
    left_contour_points = []
    
    # Track the last valid edge positions to avoid jumps
    last_left_x = middle_x
    last_right_x = middle_x
    
    # Scan UP from center to press (top of puck)
    print("Scanning UP from center to press...")
    for y in range(center_y, press_y - 1, -1):  # Go up (decreasing y)
        # Adaptive threshold based on region
        if y < center_y - 50:  # Top region (well-lit)
            adaptive_threshold = 128
        else:  # Center region
            adaptive_threshold = 100
        
        # Find left and right edges
        left_edge, right_edge = find_edges_in_row(img, y, margin, width, adaptive_threshold, last_left_x, last_right_x)
        
        # Apply jump prevention
        left_edge = prevent_jump_to_center(left_edge, last_left_x, middle_x)
        right_edge = prevent_jump_to_center(right_edge, last_right_x, middle_x)
        
        # Update last valid positions only if we found a real edge
        if left_edge != middle_x:
            last_left_x = left_edge
        if right_edge != middle_x:
            last_right_x = right_edge
        
        left_contour_points.append((y, left_edge))
        right_contour_points.append((y, right_edge))
    
    # Reset tracking for downward scan
    last_left_x = middle_x
    last_right_x = middle_x
    
    # Scan DOWN from center to ground (bottom of puck)
    print("Scanning DOWN from center to ground...")
    for y in range(center_y, ground_y + 1):  # Go down (increasing y)
        # Adaptive threshold based on region
        if y > center_y + 50:  # Bottom region (shadows)
            adaptive_threshold = 20
        else:  # Center region
            adaptive_threshold = 100
        
        # Find left and right edges
        left_edge, right_edge = find_edges_in_row(img, y, margin, width, adaptive_threshold, last_left_x, last_right_x)
        
        # Apply jump prevention
        left_edge = prevent_jump_to_center(left_edge, last_left_x, middle_x)
        right_edge = prevent_jump_to_center(right_edge, last_right_x, middle_x)
        
        # Update last valid positions only if we found a real edge
        if left_edge != middle_x:
            last_left_x = left_edge
        if right_edge != middle_x:
            last_right_x = right_edge
        
        left_contour_points.append((y, left_edge))
        right_contour_points.append((y, right_edge))
    
    # Sort by y-coordinate to ensure proper order
    left_contour_points.sort(key=lambda p: p[0])
    right_contour_points.sort(key=lambda p: p[0])
    
    return np.array(right_contour_points), np.array(left_contour_points)

def find_edges_in_row(img, y, margin, width, threshold, last_left_x, last_right_x):
    """Find left and right edges in a single row"""
    # Left contour: first white pixel when scanning left to right
    left_edge = None
    for x in range(margin, width - margin):
        if img[y, x] >= threshold:
            left_edge = x
            break
    
    # Right contour: last white pixel when scanning right to left
    right_edge = None
    for x in range(width - margin - 1, margin - 1, -1):  # Scan right to left
        if img[y, x] >= threshold:
            right_edge = x
            break
    
    # If no edge found, use the last known good position
    if left_edge is None:
        left_edge = last_left_x
    if right_edge is None:
        right_edge = last_right_x
    
    return left_edge, right_edge

def prevent_jump_to_center(current_x, last_x, middle_x, jump_threshold=20):
    """Prevent contour from jumping to center"""
    if abs(current_x - last_x) > jump_threshold:
        if abs(current_x - middle_x) < abs(last_x - middle_x):
            return last_x  # Keep the edge position
    return current_x

def smooth_contour(contour_points, window_length=21, polyorder=2):
    """Smooth contour using Savitzky-Golay filter"""
    if len(contour_points) < window_length:
        return contour_points
    
    x_coords = contour_points[:, 1]
    try:
        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_contour = np.column_stack((contour_points[:, 0], smoothed_x))
        return smoothed_contour
    except:
        print("Smoothing failed, returning original contour")
        return contour_points

def find_centerline(left_contour, right_contour, ground_y, press_y):
    """Find the average horizontal centerline between left and right contours"""
    if left_contour is None or right_contour is None:
        return None
    
    # Filter contours to only include points within ground/press region
    left_filtered = []
    right_filtered = []
    
    for point in left_contour:
        y, x = point
        if press_y <= y <= ground_y:
            left_filtered.append((y, x))
    
    for point in right_contour:
        y, x = point
        if press_y <= y <= ground_y:
            right_filtered.append((y, x))
    
    if not left_filtered or not right_filtered:
        return None
    
    # Convert to numpy arrays
    left_array = np.array(left_filtered)
    right_array = np.array(right_filtered)
    
    # Calculate average horizontal position
    all_x_positions = np.concatenate([left_array[:, 1], right_array[:, 1]])
    center_x = np.mean(all_x_positions)
    
    # Create vertical centerline from press to ground
    centerline = []
    for y in range(int(press_y), int(ground_y) + 1):
        centerline.append((y, center_x))
    
    return np.array(centerline)

def analyze_contour_shape(contour_points):
    """Analyze the contour shape and extract features"""
    if contour_points is None or len(contour_points) == 0:
        return {}
    
    y_coords = contour_points[:, 0]
    x_coords = contour_points[:, 1]
    
    features = {
        'max_width': np.max(x_coords) - np.min(x_coords),
        'mean_width': np.mean(x_coords),
        'std_width': np.std(x_coords),
        'height': len(y_coords),
        'area': trapezoid(x_coords, y_coords),
        'max_x': np.max(x_coords),
        'min_x': np.min(x_coords)
    }
    
    return features

def plot_contour_analysis(image_path, left_contour=None, right_contour=None, ground_y=None, press_y=None, save_path=None):
    """Create a plot showing the contour analysis"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(img, cmap='gray')
    
    if left_contour is not None:
        plt.plot(left_contour[:, 1], left_contour[:, 0], 'g-', linewidth=2, label='Left Contour')
    if right_contour is not None:
        plt.plot(right_contour[:, 1], right_contour[:, 0], 'orange', linewidth=2, label='Right Contour')
    
    # Add ground and press lines
    if ground_y is not None:
        plt.axhline(y=ground_y, color='red', linestyle='--', linewidth=3, label='Ground')
    if press_y is not None:
        plt.axhline(y=press_y, color='blue', linestyle='--', linewidth=3, label='Press')
    
    plt.title('Contour Analysis with Ground/Press Lines')
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_contour_distances(left_contour, right_contour, ground_y, press_y, save_path=None):
    """Plot distance from centerline to contours as function of y-position"""
    if left_contour is None or right_contour is None:
        print("No contours to plot")
        return
    
    # Find centerline
    centerline = find_centerline(left_contour, right_contour, ground_y, press_y)
    
    if centerline is None:
        print("Could not find centerline")
        return
    
    # Filter contours to only include points within ground/press region
    left_filtered = []
    right_filtered = []
    
    for point in left_contour:
        y, x = point
        if press_y <= y <= ground_y:
            left_filtered.append((y, x))
    
    for point in right_contour:
        y, x = point
        if press_y <= y <= ground_y:
            right_filtered.append((y, x))
    
    if not left_filtered or not right_filtered:
        print("No contour points in ground/press region")
        return
    
    # Convert to numpy arrays and sort by y-coordinate (bottom to top)
    left_array = np.array(left_filtered)
    right_array = np.array(right_filtered)
    
    left_array = left_array[left_array[:, 0].argsort()]
    right_array = right_array[right_array[:, 0].argsort()]
    
    # Get centerline x-coordinate
    center_x = centerline[0, 1]  # All centerline points have same x-coordinate
    
    # Calculate distances from centerline to contours
    y_positions = np.arange(int(press_y), int(ground_y) + 1)  # Y-coordinates (bottom to top)
    
    # Interpolate left and right arrays to match y_positions
    left_distances = []
    right_distances = []
    
    for y_pos in y_positions:
        # Find closest left point
        left_idx = np.argmin(np.abs(left_array[:, 0] - y_pos))
        left_x = left_array[left_idx, 1]
        
        # Find closest right point
        right_idx = np.argmin(np.abs(right_array[:, 0] - y_pos))
        right_x = right_array[right_idx, 1]
        
        left_distances.append(left_x - center_x)
        right_distances.append(right_x - center_x)
    
    left_distances = np.array(left_distances)
    right_distances = np.array(right_distances)
    
    # Smooth the distance data
    left_distances_smoothed = savgol_filter(left_distances, 21, 2)
    right_distances_smoothed = savgol_filter(right_distances, 21, 2)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original distance data
    ax1.plot(y_positions, left_distances, 'g-', linewidth=2, label='Left Contour Distance')
    ax1.plot(y_positions, right_distances, 'orange', linewidth=2, label='Right Contour Distance')
    ax1.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Centerline')
    ax1.set_xlabel('Y Position (pixels)')
    ax1.set_ylabel('Distance from Centerline (pixels)')
    ax1.set_title('Original Distance from Centerline to Contours')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Smoothed distance data
    ax2.plot(y_positions, left_distances_smoothed, 'g-', linewidth=2, label='Left Contour Distance (Smoothed)')
    ax2.plot(y_positions, right_distances_smoothed, 'orange', linewidth=2, label='Right Contour Distance (Smoothed)')
    ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Centerline')
    ax2.set_xlabel('Y Position (pixels)')
    ax2.set_ylabel('Distance from Centerline (pixels)')
    ax2.set_title('Smoothed Distance from Centerline to Contours')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Contour distance plot saved to: {save_path}")
    
    plt.show()

def process_all_images(analysis_folder, output_folder=None, max_images=10):
    """Process all images in the analysis folder"""
    image_files = load_analysis_images(analysis_folder)
    
    if not image_files:
        print("No images found to process")
        return
    
    if len(image_files) > max_images:
        print(f"Limiting to first {max_images} images for testing")
        image_files = image_files[:max_images]
    
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = []
    
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Extract contours (ground/press detection is now done inside this function)
        right_contour, left_contour = extract_contour_from_image_both_sides(image_path)
        
        # Get ground and press positions for plotting
        ground_y, press_y = detect_ground_and_press(image_path)
        
        if left_contour is not None and right_contour is not None:
            # Combine contours for analysis
            combined_contour = np.vstack([left_contour, right_contour[::-1]])
            smoothed_contour = smooth_contour(combined_contour)
            features = analyze_contour_shape(smoothed_contour)
            
            results.append({
                'image_path': image_path,
                'left_contour': left_contour,
                'right_contour': right_contour,
                'smoothed_contour': smoothed_contour,
                'features': features,
                'ground_y': ground_y,
                'press_y': press_y
            })
            
            # Create plots
            if output_folder:
                # Original contour plot
                plot_filename = f"contour_analysis_{i+1:03d}.png"
                plot_path = os.path.join(output_folder, plot_filename)
                plot_contour_analysis(image_path, left_contour, right_contour, ground_y, press_y, plot_path)
                
                # Contour distance plot
                distance_filename = f"contour_distances_{i+1:03d}.png"
                distance_path = os.path.join(output_folder, distance_filename)
                plot_contour_distances(left_contour, right_contour, ground_y, press_y, distance_path)
            else:
                if i < 3:  # Only show first 3 plots
                    plot_contour_analysis(image_path, left_contour, right_contour, ground_y, press_y)
                    plot_contour_distances(left_contour, right_contour, ground_y, press_y)
    
    return results

def save_contour_data(results, output_file):
    """Save contour data to CSV"""
    if not results:
        print("No results to save")
        return
    
    data = []
    for result in results:
        features = result['features']
        features['image_path'] = os.path.basename(result['image_path'])
        features['ground_y'] = result.get('ground_y', None)
        features['press_y'] = result.get('press_y', None)
        if result.get('ground_y') is not None and result.get('press_y') is not None:
            features['compression_height'] = result['ground_y'] - result['press_y']
        data.append(features)
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Contour data saved to: {output_file}")

def main():
    # Configuration
    analysis_folder = os.path.join(os.path.dirname(__file__), 'C2225_analysis')
    output_folder = os.path.join(os.path.dirname(__file__), 'contour_analysis_results')
    output_csv = os.path.join(os.path.dirname(__file__), 'contour_features.csv')
    
    print("=== Contour Shape Analyzer ===\n")
    print(f"Analysis folder: {analysis_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Output CSV: {output_csv}")
    print()
    
    # Process all images
    results = process_all_images(analysis_folder, output_folder)
    
    if results:
        save_contour_data(results, output_csv)
        print(f"\n=== Analysis Complete ===")
        print(f"Processed {len(results)} images")
        print(f"Results saved to: {output_csv}")
        print(f"Plots saved to: {output_folder}")
    else:
        print("No results generated")

if __name__ == "__main__":
    main() 