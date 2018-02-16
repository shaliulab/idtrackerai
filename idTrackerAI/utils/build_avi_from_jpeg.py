from __future__ import absolute_import, print_function, division
import os
import numpy as np
import cv2
import sys
import glob

def get_width_and_height(file):
    cap = cv2.VideoCapture(file)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.shape


if __name__ == "__main__":
    path_to_folder_with_jpeg = '/home/themis/Desktop/49_jpg'
    files = glob.glob(path_to_folder_with_jpeg + '/*.jpg')
    files = sorted(files, key=lambda i: os.path.splitext(os.path.basename(i))[0])
    (height, width) = get_width_and_height(files[0])
    path_to_save_video = path_to_folder_with_jpeg + '/video.avi'
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    video_writer = cv2.VideoWriter(path_to_save_video, fourcc, 32.0,
                                    (width, height))
    print("The video will be saved at ", path_to_save_video)
    for file in files:
        print('****', file)
        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        video_writer.write(frame)
