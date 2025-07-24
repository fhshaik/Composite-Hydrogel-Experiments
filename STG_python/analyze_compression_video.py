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

    finish = 0
    while not finish:
        cv2.imshow('scaling', img_contours)   
        key = cv2.waitKey(0)
        
        # press escape to exit
        if key == 27:
            finish = True
    
    cv2.destroyAllWindows()

# %% find first frame of moving impactor

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def find_compression(image_folder, test_info):
    
    # initial image
    img_init = cv2.imread(os.path.join(image_folder, images[0]))
    
    # crop image
    img_init = img_init[crop_top:-crop_bottom, crop_left:crop_right]
    
    # black and white image
    gray = cv2.cvtColor(img_init, cv2.COLOR_BGR2GRAY)

    # number of images to analyze
    num_frames = len(images)

    # array to score mse of every image from initial frame
    mse_scores = np.ndarray(num_frames)
    
    # number of processes, determined by cpu
    num = mp.cpu_count() - 1
    
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

    with mp.Manager() as manager:

        mse_scores_manager = manager.list(mse_scores)
        
        # zip arguments together for each video to be checked
        tasks = list(zip([images[s_idx:e_idx] for (s_idx, e_idx) in (zip(start, end))],
                         start,
                         [img_init] * num,
                         [image_folder] * num,
                         [mse_scores_manager] * num,
                         [test_info] * num))
                
        # make pool
        with mp.Pool(processes=num) as pool:
    
            # send all processes to analyze video
            pool.starmap(worker, tasks)
        
        # save all mse scores
        mse_scores = np.array(list(mse_scores_manager))
    
    # compression begins at first increase in mse
    # NEED TO FIX THIS LINE, NOT SURE YET HOW TO DEFINE START OF COMPRESSION
    start_compression = mse_scores.argmin()
    
    fig, axs = plt.subplots()
    
    plt.scatter(len(images),
                mse_scores)
    
    plt.vlines(start_compression, mse_scores.min(), mse_scores.max(), 'k')
    
    plt.ylabel('MSE')
    
    plt.xlabel('time [s]')
    
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

    return start_compression

def worker(captures, start, control, frame_folder, mse_scores_manager, test_info):
        
    for idx, i in enumerate(captures, start=start):
        
        img = cv2.imread(os.path.join(frame_folder, i))
                                
        mse_scores_manager[idx] = calculate_mse(control, img)
        
        print(f'Processed video {test_info.video}, frame {idx}', flush=True)


# %% set up program

# designate folder to get images from
date = '2025-07-21'
video = 'C2216'

image_folder = f"E:\\STG\\Videos\\{date}\\{video}"

# array of image names
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

# %% get test info

# all trials
test_info = pd.read_csv('test_info\\2025-07-18_test_info.csv')

# this trial
data = test_info[test_info.video == float(video.strip('C'))]

fps = 120 # frames per second of video
dt = 1.0 / fps # time between frames (seconds)

# %% make folder to store analyzed images

os.mkdir(f"E:\\STG\\Videos\\COMPRESSION_ANALYSIS\\{video}")

# %% crop image

# read in example image of impactor
example_img = cv2.imread(image_folder + '\\' + images[100])

height, length, _ = example_img.shape # height and length of image

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

# draw line
cv2.line(control_gray, (left, top), (right, bottom), 255, 10)

# crop image
top_t = 250
bottom_t = height
left_t = 1000
right_t = 3000

t_length = right_t - left_t
t_height = bottom_t - top_t

imS = cv2.resize(control_gray[top_t:bottom_t,left_t:right_t], (int(t_length*scale), int(t_height*scale)))

finish = 0
while not finish:
    cv2.imshow('scaling', imS)   
    key = cv2.waitKey(0)
    
    # press escape to exit
    if key == 27:
        finish = True

cv2.destroyAllWindows()

# width of impactor in mm
scale_mm = 25.4

# mm per pixel, from scale bar
mm_length = 25.4 / get_dist((left, top), (right, bottom))

# center of impactor
impactor_center = int((right - left) * scale / 2)

# left side of impactor
impactor_left = int((left - left_t) * scale)

print(f'mm/pixel: {mm_length}')

# %% find puck

find_puck(imS)

# %% EDIT THIS FUNCTION
# goes through every frame of video, calculating mse with initial frame
# (also calls find_puck for every frame of video)
# save the contour and mse for each frame in a dataframe
# use mse to identify the frame at which compression starts

find_compression(image_folder, test_info)

# %% WRITE THIS FUNCTION
# choose a y-location somewhere along the puck
# goes through dataframe, calculating width of puck at that y-location for each frame

# %% WRITE THIS FUNCTION
# plot left side of contour as a function of time

# %% WRITE THIS SECTION
# plot width of puck at a certain y-location as a function of time

# %% save dataframe as .csv

rad.to_csv(f'Compression analysis\\{video}_compression.csv')