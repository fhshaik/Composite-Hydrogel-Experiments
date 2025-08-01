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
    
    # Convert back to original image space
    if ground_y is not None:
        ground_y += crop_start
    if press_y is not None:
        press_y += crop_start
    
    print(f"Ground position: {ground_y}, Press position: {press_y}")
    return ground_y, press_y

def extract_contour_from_image_both_sides(image_path, threshold=128, margin=100):
    """Extract contour shape from both left and right sides"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None
    
    height, width = img.shape
    middle_x = width // 2
    right_contour_points = []
    left_contour_points = []
    
    # Track the last valid edge positions to avoid jumps
    last_left_x = middle_x
    last_right_x = middle_x
    
    for y in range(height):
        # Left contour: first white pixel when scanning left to right
        first_white_x_left = middle_x
        for x in range(margin, width - margin):
            if img[y, x] >= threshold:
                first_white_x_left = x
                break
        
        # Right contour: last white pixel when scanning left to right
        last_white_x_right = middle_x
        for x in range(margin, width - margin):
            if img[y, x] >= threshold:
                last_white_x_right = x
        
        # Check for jumps to center and fix them (more conservative)
        if abs(first_white_x_left - last_left_x) > 100:  # Only fix very large jumps
            # Only fix if the new position is closer to center than the last position
            if abs(first_white_x_left - middle_x) < abs(last_left_x - middle_x):
                first_white_x_left = last_left_x  # Keep the edge position
        
        if abs(last_white_x_right - last_right_x) > 100:  # Only fix very large jumps
            # Only fix if the new position is closer to center than the last position
            if abs(last_white_x_right - middle_x) < abs(last_right_x - middle_x):
                last_white_x_right = last_right_x  # Keep the edge position
        
        # Update last valid positions only if we found a real edge
        if first_white_x_left != middle_x:
            last_left_x = first_white_x_left
        if last_white_x_right != middle_x:
            last_right_x = last_white_x_right
        
        left_contour_points.append((y, first_white_x_left))
        right_contour_points.append((y, last_white_x_right))
    
    return np.array(right_contour_points), np.array(left_contour_points)

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
        
        # Detect ground and press positions
        ground_y, press_y = detect_ground_and_press(image_path)
        
        # Extract contours
        right_contour, left_contour = extract_contour_from_image_both_sides(image_path)
        
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
            
            # Create plot
            if output_folder:
                plot_filename = f"contour_analysis_{i+1:03d}.png"
                plot_path = os.path.join(output_folder, plot_filename)
                plot_contour_analysis(image_path, left_contour, right_contour, ground_y, press_y, plot_path)
            else:
                if i < 3:  # Only show first 3 plots
                    plot_contour_analysis(image_path, left_contour, right_contour, ground_y, press_y)
    
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