# -*- coding: utf-8 -*-
"""
Script to analyze contour shapes from C2225 analysis folder
Extracts contour shape by tracing from middle vertical line to last white point
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent PIL errors
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

def detect_ground_and_press(image_path, fixed_ground_y=None, prev_press_y=None, search_range=15):
    """Detect ground and press positions using temporal consistency"""
    
    # Detect ground position (use first frame as reference if available)
    if fixed_ground_y is not None:
        ground_y = fixed_ground_y
    else:
        ground_y = detect_ground_position_first_frame(image_path)
    
    # Detect press position with temporal consistency
    press_y = detect_press_position_simple_edge(image_path, prev_press_y, search_range)
    
    if press_y is None:
        print("Could not detect press position")
        return ground_y, None
    
    return ground_y, press_y

def detect_ground_position_first_frame(image_path):
    """Detect ground position in the first frame"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
        print("No horizontal lines detected in first frame")
        return None
    
    # Extract and filter horizontal line positions for ground
    ground_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 15:  # Nearly horizontal
            y_avg = (y1 + y2) // 2
            if y_avg > cropped_height * 0.7:  # Lower 30% - likely ground
                ground_candidates.append(y_avg)
    
    if not ground_candidates:
        print("No ground candidates found in first frame")
        return None
    
    # Take the lowest position as ground
    ground_y = max(ground_candidates)  # Lowest in cropped image
    ground_y += crop_start  # Convert to original image space
    ground_y -= 6  # Move ground line 6 pixels up
    
    return int(ground_y)  # Ensure integer return

def detect_press_position_raw(image_path, prev_press_y=None, search_range=15):
    """Legacy function - now replaced by detect_press_position_simple_edge"""
    return detect_press_position_simple_edge(image_path, prev_press_y, search_range)

def count_white_pixels_around_line(img, y_position, margin=10, threshold=128):
    """Legacy function - no longer used"""
    return 0, 0

def visualize_adaptive_refinement(img, y_position, margin=10, threshold=128):
    """Legacy function - no longer used"""
    return img

def refine_press_position_adaptive(img, initial_y, prev_press_y=None, max_iterations=10):
    """Legacy function - no longer used"""
    return initial_y

def detect_press_position_simple_edge(image_path, prev_press_y=None, search_range=15):
    """
    Simple press position detection using edge maximization
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
        
        height, width = img.shape
        
        # Crop out the top quarter to eliminate press top section
        crop_start = height // 4
        cropped_img = img[crop_start:, :]
        cropped_height = cropped_img.shape[0]
        
        # Apply edge detection
        edges = cv2.Canny(cropped_img, 50, 150)
        
        # Find horizontal edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # If we have previous position, search around it
        if prev_press_y is not None:
            # Convert to cropped image coordinates
            prev_y_cropped = prev_press_y - crop_start
            
            # Search around previous position
            search_start = max(0, prev_y_cropped - search_range)
            search_end = min(cropped_height, prev_y_cropped + search_range)
            
            best_y = None
            max_white_pixels = 0
            
            for y in range(search_start, search_end):
                # Count white pixels (edges) in horizontal line
                white_pixels = np.sum(horizontal_edges[y, :])
                
                if white_pixels > max_white_pixels:
                    max_white_pixels = white_pixels
                    best_y = y
            
            if best_y is not None and max_white_pixels > width * 0.05:  # At least 5% of width
                return int(best_y + crop_start)
            else:
                # Fall back to previous position
                return prev_press_y
        
        else:
            # First frame: search in upper half
            search_start = 0
            search_end = cropped_height // 2
            
            best_y = None
            max_white_pixels = 0
            
            for y in range(search_start, search_end):
                # Count white pixels (edges) in horizontal line
                white_pixels = np.sum(horizontal_edges[y, :])
                
                if white_pixels > max_white_pixels:
                    max_white_pixels = white_pixels
                    best_y = y
            
            if best_y is not None and max_white_pixels > width * 0.05:
                return int(best_y + crop_start)
            else:
                return None
        
    except Exception as e:
        print(f"Error in simple edge press detection: {e}")
        return prev_press_y if prev_press_y is not None else None



def extract_contour_from_image_both_sides(image_path, threshold=80, margin=100, fixed_ground_y=None, prev_press_y=None):
    """Extract contour shape from both left and right sides, scanning from center up and down"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None
    
    height, width = img.shape
    middle_x = width // 2
    
    # Detect ground and press to find the center region
    ground_y, press_y = detect_ground_and_press(image_path, fixed_ground_y, prev_press_y, search_range=15)
    
    if ground_y is None or press_y is None:
        print("Could not detect ground and press positions")
        return None, None
    
    # Convert to integers for range operations
    ground_y = int(ground_y)
    press_y = int(press_y)
    
    # Calculate the center y-position (middle of puck)
    center_y = (ground_y + press_y) // 2
    print(f"Center y-position: {center_y} (ground: {ground_y}, press: {press_y})")
    
    # Validate that we have a reasonable range
    if abs(ground_y - press_y) < 10:
        print("Warning: Ground and press positions are too close together")
        return None, None
    
    # Ensure press_y is less than ground_y (press is above ground)
    if press_y >= ground_y:
        print("Warning: Press position is not above ground position")
        return None, None
    
    # Ensure we have valid y-coordinates for scanning
    if press_y < 0 or ground_y >= height:
        print("Warning: Press or ground position outside image bounds")
        return None, None
    
    # Initialize contour arrays
    right_contour_points = []
    left_contour_points = []
    
    # Track the last valid edge positions to avoid jumps
    # Start with None to indicate no previous good positions
    last_left_x = None
    last_right_x = None
    
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
        
        # Apply jump prevention with side-specific logic
        left_edge = prevent_jump_to_center(left_edge, last_left_x, middle_x, contour_side='left')
        right_edge = prevent_jump_to_center(right_edge, last_right_x, middle_x, contour_side='right')
        
        # Update last valid positions only if we found a real edge (not None)
        if left_edge is not None and left_edge != middle_x:
            last_left_x = left_edge
        if right_edge is not None and right_edge != middle_x:
            last_right_x = right_edge
        
        left_contour_points.append((y, left_edge))
        right_contour_points.append((y, right_edge))
    
    # Reset tracking for downward scan
    last_left_x = None
    last_right_x = None
    
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
        
        # Apply jump prevention with side-specific logic
        left_edge = prevent_jump_to_center(left_edge, last_left_x, middle_x, contour_side='left')
        right_edge = prevent_jump_to_center(right_edge, last_right_x, middle_x, contour_side='right')
        
        # Update last valid positions only if we found a real edge (not None)
        if left_edge is not None and left_edge != middle_x:
            last_left_x = left_edge
        if right_edge is not None and right_edge != middle_x:
            last_right_x = right_edge
        
        left_contour_points.append((y, left_edge))
        right_contour_points.append((y, right_edge))
    
    # Sort by y-coordinate to ensure proper order
    left_contour_points.sort(key=lambda p: p[0])
    right_contour_points.sort(key=lambda p: p[0])
    
    return np.array(right_contour_points), np.array(left_contour_points)

def find_edges_in_row(img, y, margin, width, threshold, last_left_x, last_right_x):
    """Find left and right edges in a single row with improved asymmetric handling"""
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
    
    # Validate that left and right edges are reasonable
    if left_edge is not None and right_edge is not None:
        # Ensure left edge is actually to the left of right edge
        if left_edge >= right_edge:
            # This shouldn't happen, but if it does, try to fix
            print(f"Warning: Left edge ({left_edge}) >= Right edge ({right_edge}) at y={y}")
            # Try to find a better right edge by looking further left
            for x in range(left_edge + 1, width - margin):
                if img[y, x] >= threshold:
                    right_edge = x
                    break
    
    # If no edge found, use the last known good position (only if not None)
    if left_edge is None and last_left_x is not None:
        left_edge = last_left_x
    if right_edge is None and last_right_x is not None:
        right_edge = last_right_x
    
    # Additional validation: ensure edges are not too close together
    if left_edge is not None and right_edge is not None:
        min_separation = 20  # Minimum pixels between left and right edges
        if right_edge - left_edge < min_separation:
            print(f"Warning: Edges too close at y={y}: left={left_edge}, right={right_edge}")
            # Use last known good positions if current ones are too close and last positions are valid
            if (last_left_x is not None and last_right_x is not None and 
                abs(last_left_x - last_right_x) >= min_separation):
                left_edge = last_left_x
                right_edge = last_right_x
    
    return left_edge, right_edge

def prevent_jump_to_center(current_x, last_x, middle_x, jump_threshold=20, contour_side='left'):
    """Prevent contour from jumping to center with side-specific logic"""
    # If last_x is None, we have no previous position to compare against
    if last_x is None:
        return current_x
    
    # If current_x is None, we have no current position to work with
    if current_x is None:
        return last_x
    
    if abs(current_x - last_x) > jump_threshold:
        # For left contour: should stay on the left side
        if contour_side == 'left' and current_x > middle_x:
            return last_x  # Keep the left position
        # For right contour: should stay on the right side  
        elif contour_side == 'right' and current_x < middle_x:
            return last_x  # Keep the right position
        # If jump is reasonable (not crossing center), allow it
        elif abs(current_x - middle_x) < abs(last_x - middle_x):
            return last_x  # Keep the edge position
    return current_x

def smooth_contour(contour_points, window_length=21, polyorder=2):
    """Smooth contour using Savitzky-Golay filter"""
    if len(contour_points) < window_length:
        return contour_points
    
    x_coords = contour_points[:, 1]
    try:
        # Ensure window_length is odd and not larger than data
        if window_length % 2 == 0:
            window_length += 1
        if window_length > len(x_coords):
            window_length = len(x_coords) - 1 if len(x_coords) > 1 else 1
        
        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_contour = np.column_stack((contour_points[:, 0], smoothed_x))
        return smoothed_contour
    except Exception as e:
        print(f"Smoothing failed: {e}, returning original contour")
        return contour_points

def find_centerline(left_contour, right_contour, ground_y, press_y):
    """Find the average horizontal centerline between left and right contours"""
    if left_contour is None or right_contour is None:
        return None
    
    # Check if ground_y and press_y are valid
    if ground_y is None or press_y is None:
        print("Warning: ground_y or press_y is None, cannot find centerline")
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
    
    plt.close()

def plot_contour_distances(left_contour, right_contour, ground_y, press_y, save_path=None):
    """Plot distance from centerline to contours as function of y-position"""
    if left_contour is None or right_contour is None:
        print("No contours to plot")
        return
    
    # Check if ground_y and press_y are valid
    if ground_y is None or press_y is None:
        print("Warning: ground_y or press_y is None, cannot plot distances")
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
    if len(centerline) == 0:
        print("Centerline is empty")
        return
    center_x = centerline[0, 1]  # All centerline points have same x-coordinate
    
    # Calculate distances from centerline to contours
    y_positions = np.arange(int(press_y), int(ground_y) + 1)  # Y-coordinates (bottom to top)
    
    # Interpolate left and right arrays to match y_positions
    left_distances = []
    right_distances = []
    
    for y_pos in y_positions:
        # Find closest left point
        if len(left_array) > 0:
            left_idx = np.argmin(np.abs(left_array[:, 0] - y_pos))
            left_x = left_array[left_idx, 1]
        else:
            left_x = center_x
        
        # Find closest right point
        if len(right_array) > 0:
            right_idx = np.argmin(np.abs(right_array[:, 0] - y_pos))
            right_x = right_array[right_idx, 1]
        else:
            right_x = center_x
        
        left_distances.append(left_x - center_x)
        right_distances.append(right_x - center_x)
    
    left_distances = np.array(left_distances)
    right_distances = np.array(right_distances)
    
    # Smooth the distance data
    try:
        left_distances_smoothed = savgol_filter(left_distances, 21, 2)
        right_distances_smoothed = savgol_filter(right_distances, 21, 2)
    except Exception as e:
        print(f"Smoothing failed: {e}, using original data")
        left_distances_smoothed = left_distances
        right_distances_smoothed = right_distances
    
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
    
    plt.close()

def visualize_press_height_tracking(image_path, press_y, ground_y, save_path=None):
    """
    Visualize press height tracking for debugging
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Draw press line (blue)
    if press_y is not None:
        cv2.line(img, (0, press_y), (img.shape[1], press_y), (255, 0, 0), 2)
        cv2.putText(img, f'Press: {press_y}', (10, press_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw ground line (red)
    if ground_y is not None:
        cv2.line(img, (0, ground_y), (img.shape[1], ground_y), (0, 0, 255), 2)
        cv2.putText(img, f'Ground: {ground_y}', (10, ground_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw compression height
    if press_y is not None and ground_y is not None:
        compression_height = ground_y - press_y
        cv2.putText(img, f'Compression: {compression_height}px', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, img)
    else:
        cv2.imshow('Press Height Tracking', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_all_images(analysis_folder, output_folder=None, max_images=1000):
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
    
    # Detect fixed ground position from first frame
    print("Detecting fixed ground position from first frame...")
    fixed_ground_y = detect_ground_position_first_frame(image_files[0])
    if fixed_ground_y is None:
        print("Could not detect ground position in first frame, using default")
        fixed_ground_y = 450  # Default ground position
    
    print(f"Fixed ground position: {fixed_ground_y}")
    
    # Track previous press position and history for temporal consistency
    prev_press_y = None
    press_history = []  # Keep last 5 press positions for smoothing
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Get ground and press positions with fixed ground and temporal consistency
            ground_y, press_y = detect_ground_and_press(image_path, fixed_ground_y, prev_press_y, search_range=15)
            
            # Extract contours (ground/press detection is now done inside this function)
            right_contour, left_contour = extract_contour_from_image_both_sides(image_path, fixed_ground_y=fixed_ground_y, prev_press_y=prev_press_y)
            
            # Update previous press position and history for next frame
            if press_y is not None:
                prev_press_y = press_y
                press_history.append(press_y)
                # Keep only last 5 positions
                if len(press_history) > 5:
                    press_history.pop(0)
            
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
                
                # Create plots only if ground and press detection was successful
                if ground_y is not None and press_y is not None:
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
                else:
                    print(f"Warning: Ground/press detection failed for {os.path.basename(image_path)}, skipping plots")
            else:
                print(f"Warning: Contour extraction failed for {os.path.basename(image_path)}")
                
        except Exception as e:
            print(f"Error processing image {os.path.basename(image_path)}: {e}")
            continue  # Continue with next image
    
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