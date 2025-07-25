# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 20:01:21 2023

@author: malco
"""

import cv2
import os
import numpy as np
import multiprocessing as mp

# make pool of processes
def run_pool(vid_info):
    
    # number of processes, determined by cpu
    num = mp.cpu_count() - 1
        
    # send all processes in pool to convert videos (one video at a time)
    # argument includes video and address of new folder
    for (video, folder, save_folder) in vid_info:
        
        # print current video
        print(video)
        
        capture = cv2.VideoCapture(folder + '.MP4') # open video
        
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames in video

        capture.release() # close the video

        # check if directory already exists
        if not os.path.exists(folder): # if no directory exists, make one
            os.mkdir(folder)
        
        # split check_images into num sections of equal length (last section might be short)
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

        # zip arguments together for each video to be converted
        tasks = list(zip([video] * num, [folder] * num, [save_folder] * num, start, end))
        
        # make pool
        with mp.Pool(processes=num) as pool:

            # send all processes to analyze one video at a time
            pool.starmap(convert_vid_part, tasks)

# convert a single video from .mp4 to .jpg and save frames in a new folder
def convert_vid_part(video, folder, save_folder, start, end):
        
    capture = cv2.VideoCapture(folder + '.MP4') # open video
        
    capture.set(cv2.CAP_PROP_POS_FRAMES, start) # go to first frame of this section
        
    # iterate through selected frames, converting one at a time
    for frameNr in range(start, end):
        
        if frameNr % 10 == 0:
            print(frameNr, flush=True)
        
        success, frame = capture.read()
    
        if success: # write frame to directory as .jpg
            if not os.path.exists(save_folder + '\\{}_{:>05d}.jpg'.format(video, frameNr)):            
                cv2.imwrite(save_folder +
                            '\\{}_{:>05d}.jpg'.format(video, frameNr), frame)
            else:
                continue
        else:
            break

    capture.release() # close the video

# %%

if __name__ == '__main__':
    # Set up for C2225.MP4 in the current directory
    video_filename = 'C2225'
    mp4_path = os.path.join(os.path.dirname(__file__), video_filename + '.MP4')
    frames_folder = os.path.join(os.path.dirname(__file__), video_filename + '_frames')

    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"Video file not found: {mp4_path}")
    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)

    # Prepare info for processing
    vid_info = [(video_filename, os.path.splitext(mp4_path)[0], frames_folder)]
    run_pool(vid_info)

    image_folder = os.path.join(os.path.dirname(__file__), 'C2225_frames')