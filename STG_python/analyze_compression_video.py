# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:01:06 2024

@author: malco
"""

import cv2
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import interpolate
from scipy import signal
from collections import OrderedDict
from datetime import datetime
import circle_fit
import multiprocessing as mp

# --- ADDED: Parameters for motion detection ---
BLUR_KERNEL = (7, 7)  # Gaussian blur kernel size
ROLLING_WINDOW = 7    # Rolling mean window size for MSE smoothing
THRESHOLD_FACTOR = 3  # How many std devs above noise to trigger motion
CONSECUTIVE_FRAMES = 5  # Require threshold to be exceeded for this many frames
# --- END ADDED ---

today = datetime.today().strftime('%Y-%m-%d')

# %% get distance between two points
def get_dist(a, b):
    
    a = np.array(a)
    b = np.array(b)
    
    return np.sqrt(np.sum(np.square(a - b)))

# %% calculate rmse
def calc_rmse(y_data, y_fit):
    return np.sqrt(np.mean((y_data - y_fit) ** 2))

# %% find left side of puck using contours

def find_puck(img):
    
    # crop image close to puck, with right side of image the center of the impactor
    global crop_top, crop_bottom, crop_left, crop_right
    
    crop_top = 250
    crop_bottom = 70
    crop_left = 100
    crop_right = impactor_center + impactor_left
    
    img = img[crop_top:-crop_bottom, crop_left:crop_right]
    
    # threshold image
    _, thresh = cv2.threshold(img, 150, 200, cv2.THRESH_BINARY)

    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # convert grayscale to BGR for color drawing
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # select largest contour (ideally contour around the puck)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # new height and length of image
    height2, length2 = img.shape[:2]

    # make a copy of the contour
    modified_contour = largest_contour.copy()
    
    # replace all points on the right half of the image with right boundary of image
    for i in range(modified_contour.shape[0]):
        if modified_contour[i, 0, 0] > length2 // 2:  # if x > halfway
            modified_contour[i, 0, 0] = length2 - 1  # set x to right edge
    
    # draw the contour
    cv2.drawContours(img_contours, [modified_contour], -1, (0, 255, 0), 2)

    # --- CHANGED: Remove GUI code for headless execution ---
    # finish = 0
    # while not finish:
    #     cv2.imshow('scaling', img_contours)   
    #     key = cv2.waitKey(0)
    #     
    #     # press escape to exit
    #     if key == 27:
    #         finish = True
    # cv2.destroyAllWindows()

# %% find first frame of moving impactor

# --- ADDED: Blurring in calculate_mse ---
def calculate_mse(img1, img2):
    # Apply Gaussian blur to both images before MSE
    img1_blur = cv2.GaussianBlur(img1, BLUR_KERNEL, 0)
    img2_blur = cv2.GaussianBlur(img2, BLUR_KERNEL, 0)
    return np.mean((img1_blur - img2_blur) ** 2)
# --- END ADDED ---

# --- UPDATED: Robust motion detection with smoothing and consecutive frames ---
def detect_motion_start(mse_scores, threshold_factor=THRESHOLD_FACTOR, consecutive_frames=CONSECUTIVE_FRAMES, rolling_window=ROLLING_WINDOW):
    """
    Detects the start of motion by finding the first frame where the smoothed MSE
    rises above a baseline threshold for several consecutive frames.
    """
    mse_smooth = pd.Series(mse_scores).rolling(rolling_window, min_periods=1, center=True).mean().values
    baseline_window = max(50, len(mse_smooth)//20)
    baseline_mean = np.mean(mse_smooth[:baseline_window])
    baseline_std = np.std(mse_smooth[:baseline_window])
    threshold = baseline_mean + threshold_factor * baseline_std
    above = mse_smooth > threshold
    # --- ADDED: Debug prints for new approach ---
    print(f"Baseline mean: {baseline_mean}, std: {baseline_std}, threshold: {threshold}")
    print(f"First 30 smoothed MSE: {mse_smooth[:30]}")
    print(f"First 30 'above' values: {above[:30]}")
    # --- END ADDED ---
    for i in range(len(above) - consecutive_frames + 1):
        if np.all(above[i:i+consecutive_frames]):
            return i
    return 0  # fallback
# --- END UPDATED ---

def find_compression(image_folder, test_info):
    # --- ADDED: Debug prints ---
    print("Starting find_compression function...")
    # --- END ADDED ---
    
    # --- ADDED: Convert test_info to dict for multiprocessing ---
    test_info_dict = test_info.to_dict('records')[0] if hasattr(test_info, 'to_dict') else dict(test_info)
    # --- END ADDED ---
    
    # initial image
    img_init = cv2.imread(os.path.join(image_folder, images[0]))
    # --- ADDED: Debug print ---
    print(f"Initial image shape: {img_init.shape if img_init is not None else 'None'}")
    # --- END ADDED ---
    
    # crop image
    img_init = img_init[crop_top:-crop_bottom, crop_left:crop_right]
    # --- ADDED: Debug print ---
    print(f"Cropped initial image shape: {img_init.shape}")
    # --- END ADDED ---
    
    # black and white image
    gray = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)
    # --- ADDED: Debug print ---
    print(f"Grayscale image shape: {gray.shape}")
    # --- END ADDED ---

    # number of images to analyze
    num_frames = len(images)
    # --- ADDED: Debug print ---
    print(f"Number of frames to process: {num_frames}")
    # --- END ADDED ---

    # array to score mse of every image from initial frame
    mse_scores = np.ndarray(num_frames)
    
    # number of processes, determined by cpu
    num = mp.cpu_count() - 1
    # --- ADDED: Debug print ---
    print(f"Using {num} processes for multiprocessing")
    # --- END ADDED ---
    
    # split images into num sections of equal length (last section might be short)
    len_sublist, remainder = divmod(num_frames, num) # number of frames in each sublist (and remainder)
    
    # indices of start frames
    start = [None] * num
    
    # indices of end frames
    end = [None] * num
    
    s = 0
    
    for i in range(num):
        start[i] = s
        end[i] = start[i] + len_sublist + (1 if i < remainder else 0)
        
        s = end[i]

    # --- ADDED: Debug print ---
    print("About to start multiprocessing...")
    # --- END ADDED ---

    with mp.Manager() as manager:
        mse_scores_manager = manager.list(mse_scores)
        # --- CHANGED: Pass test_info_dict instead of test_info ---
        tasks = list(zip([images[s_idx:e_idx] for (s_idx, e_idx) in (zip(start, end))],
                         start,
                         [img_init] * num,
                         [image_folder] * num,
                         [mse_scores_manager] * num,
                         [test_info_dict] * num,
                         [crop_top] * num,
                         [crop_bottom] * num,
                         [crop_left] * num,
                         [crop_right] * num))
        # --- END CHANGED ---
        # make pool
        with mp.Pool(processes=num) as pool:
            # send all processes to analyze video
            pool.starmap(worker, tasks)
        # save all mse scores
        mse_scores = np.array(list(mse_scores_manager))

    # --- CHANGED: Use robust motion detection ---
    start_compression = detect_motion_start(mse_scores)
    # --- ADDED: Print detected start index for motion ---
    print(f"Detected start index for motion (MSE): {start_compression}")
    # --- END ADDED ---
    # --- END CHANGED ---

    fig, axs = plt.subplots()
    plt.scatter(range(len(images)), mse_scores)
    plt.vlines(start_compression, mse_scores.min(), mse_scores.max(), 'k', label='Start of motion')
    plt.ylabel('MSE')
    plt.xlabel('Frame index')
    plt.legend()
    plt.show()
    
    mse_folder = os.path.join(f"E:\\STG\\Videos\\MSE")
    if not os.path.exists(mse_folder):
        os.mkdir(mse_folder)
    fig.savefig(os.path.join(mse_folder, f'C{video}_MSE'), dpi=300)
    mse_save_folder = 'MSE'
    mse_df = pd.DataFrame(
        {'frames': [int(i.split('_')[-1].strip('.jpg')) for i in images],
         'mse': mse_scores})
    mse_df.to_csv(os.path.join(mse_save_folder, f'C{video}_MSE.csv'))
    print(f'{test_info.date} trial {test_info.trial}, compression occurs at by frame {start_compression}')
    # --- ADDED: Print detected start of motion ---
    print(f"Detected start of motion at frame: {start_compression}")
    # --- END ADDED ---
    return start_compression

def worker(captures, start, control, frame_folder, mse_scores_manager, test_info_dict, crop_top, crop_bottom, crop_left, crop_right):
    for idx, i in enumerate(captures, start=start):
        img = cv2.imread(os.path.join(frame_folder, i))
        # --- CHANGED: Use passed crop variables ---
        img = img[crop_top:-crop_bottom, crop_left:crop_right]
        # --- ADDED: Apply Gaussian blur before MSE (handled in calculate_mse now) ---
        # --- END ADDED ---
        mse_scores_manager[idx] = calculate_mse(control, img)
        # --- CHANGED: Use test_info_dict instead of test_info ---
        print(f'Processed video {test_info_dict.get("video", "unknown")}, frame {idx}', flush=True)
        # --- END CHANGED ---


# %% set up program

# designate folder to get images from
date = 'C2225_frames'  # Updated to match the extracted frames folder
video = 'C2225'

# Use the absolute path to the frames folder in the current directory
image_folder = os.path.join(os.path.dirname(__file__), date)

# array of image names
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# %% get test info

# --- CHANGED: Use absolute path for test info file ---
test_info_path = os.path.join(os.path.dirname(__file__), '2025-07-18_test_info.csv')
test_info = pd.read_csv(test_info_path)
# --- END CHANGED ---

# this trial
#data = test_info[test_info.video == float(video.strip('C'))]

fps = 120 # frames per second of video
dt = 1.0 / fps # time between frames (seconds)

# --- CHANGED: Wrap main execution in if __name__ == '__main__' for Windows multiprocessing safety ---
if __name__ == '__main__':
    # --- ADDED: Debug print ---
    print("Starting main execution...")
    # --- END ADDED ---
    
    # %% make folder to store analyzed images
    output_folder = os.path.join(os.path.dirname(__file__), f"{video}_analysis")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # --- ADDED: Debug print ---
    print("Created output folder")
    # --- END ADDED ---

    # %% crop image
    example_img = cv2.imread(os.path.join(image_folder, images[100]))
    if example_img is None:
        raise FileNotFoundError(f"Could not read example image: {os.path.join(image_folder, images[100])}")
    # --- ADDED: Debug print ---
    print("Read example image successfully")
    # --- END ADDED ---

    height, length, _ = example_img.shape # height and length of image
    # --- ADDED: Debug print ---
    print(f"Example image shape: {example_img.shape}")
    # --- END ADDED ---

    # coordinates for line to draw on image (scale bar)
    top = 1000
    bottom = top
    left = 1550
    right = 2530

    # show image to verify line is drawn correctly, adjust as necessary
    control_col = example_img
    scale = 0.3

    # convert control image to grayscale
    control_gray = cv2.cvtColor(control_col, cv2.COLOR_BGR2GRAY)
    # --- ADDED: Debug print ---
    print("Converted control image to grayscale")
    # --- END ADDED ---

    # --- ADDED: Save sample grayscale images for verification ---
    test_images_folder = os.path.join(os.path.dirname(__file__), 'test_images')
    if not os.path.exists(test_images_folder):
        os.makedirs(test_images_folder)
    # Save initial, mid, and last grayscale frames
    sample_indices = [0, len(images)//2, len(images)-1]
    for idx in sample_indices:
        img_path = os.path.join(image_folder, images[idx])
        img = cv2.imread(img_path)
        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out_path = os.path.join(test_images_folder, f'gray_frame_{idx}.png')
            cv2.imwrite(out_path, gray_img)
            print(f"Saved grayscale image: {out_path}")
        else:
            print(f"Could not read image for grayscale save: {img_path}")
    # --- END ADDED ---
    # --- ADDED: Debug print ---
    print("Finished saving test images")
    # --- END ADDED ---

    # draw line
    cv2.line(control_gray, (left, top), (right, bottom), 255, 10)
    # --- ADDED: Debug print ---
    print("Drew line on control image")
    # --- END ADDED ---

    # crop image
    top_t = 250
    bottom_t = height
    left_t = 1000
    right_t = 3000

    t_length = right_t - left_t
    t_height = bottom_t - top_t

    imS = cv2.resize(control_gray[top_t:bottom_t,left_t:right_t], (int(t_length*scale), int(t_height*scale)))
    # --- ADDED: Debug print ---
    print("Resized cropped image")
    # --- END ADDED ---

    # --- CHANGED: Remove GUI code for headless execution ---
    # finish = 0
    # while not finish:
    #     cv2.imshow('scaling', imS)   
    #     key = cv2.waitKey(0)
    #     # press escape to exit
    #     if key == 27:
    #         finish = True
    # cv2.destroyAllWindows()
    # --- END CHANGED ---

    # width of impactor in mm
    scale_mm = 25.4

    # mm per pixel, from scale bar
    mm_length = 25.4 / get_dist((left, top), (right, bottom))

    # center of impactor
    impactor_center = int((right - left) * scale / 2)

    # left side of impactor
    impactor_left = int((left - left_t) * scale)

    print(f'mm/pixel: {mm_length}')
    # --- ADDED: Debug print ---
    print("About to call find_puck...")
    # --- END ADDED ---

    # %% find puck
    find_puck(imS)
    # --- ADDED: Debug print ---
    print("Finished find_puck")
    # --- END ADDED ---

    # --- ADDED: Debug print ---
    print("About to call find_compression...")
    # --- END ADDED ---

    # %% EDIT THIS FUNCTION
    # goes through every frame of video, calculating mse with initial frame
    # (also calls find_puck for every frame of video)
    # save the contour and mse for each frame in a dataframe
    # use mse to identify the frame at which compression starts
    find_compression(image_folder, test_info)
    # --- END CHANGED ---

    # %% WRITE THIS FUNCTION
    # choose a y-location somewhere along the puck
    # goes through dataframe, calculating width of puck at that y-location for each frame

    # %% WRITE THIS FUNCTION
    # plot left side of contour as a function of time

    # %% WRITE THIS SECTION
    # plot width of puck at a certain y-location as a function of time

    # %% save dataframe as .csv
    # rad.to_csv(f'Compression analysis\\{video}_compression.csv')
# --- END CHANGED ---