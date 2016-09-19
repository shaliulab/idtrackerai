import cv2
import sys
sys.path.append('../utils')
from py_utils import *
import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import time
import re
from joblib import Parallel, delayed
import multiprocessing
import itertools
import cPickle as pickle

numSegment = 0
# paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
# frameIndices = pd.read_pickle('../Cafeina5peces/Caffeine5fish_frameIndices.pkl')
# allIdentities = pd.read_pickle('../Cafeina5peces/Caffeine5fish_identities.pkl')
paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
frameIndices = pd.read_pickle('../Conflict8/conflict3and4_frameIndices.pkl')
allIdentities = pd.read_pickle('../Conflict8/conflict3and4_identities.pkl')
path = paths[numSegment]

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(100, max_value, interval)]
    rgbcolorslist = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    return rgbcolorslist

def IdPlayer(path,allIdentities,frameIndices):

    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    sNumber = int(filename.split('_')[-1])
    folder = os.path.dirname(path)
    df = pd.read_pickle(folder +'/'+ filename + '.pkl')
    cap = cv2.VideoCapture(path)
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    numAnimals = 8
    colors = get_spaced_colors(8)
    print 'colors, ',colors
    def onChange(trackbarValue):
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        centroids = df.loc[trackbarValue,'centroids']
        pixels = df.loc[trackbarValue,'pixels']
        permutation = df.loc[trackbarValue,'permutation']

        #Get frame from video file
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        frameCopy = frame.copy()

        # Plot segmentated blobs



        # plot numbers if not crossing
        globalFrame = frameIndices[frameIndices['frame']== trackbarValue][frameIndices['segment']==sNumber].index[0]
        print 'permutation, ', permutation
        if not isinstance(permutation,float):
            print 'pass'
            # shadows
            if trackbarValue > 0:
                previousFrame = trackbarValue -1
                shadowsCounter = 1
                # frameShadows = np.zeros_like(frame)
                while not isinstance(df.loc[previousFrame,'permutation'],float):
                    # framePreviousShadows = np.zeros_like(frame)
                    previousPixels = df.loc[previousFrame,'pixels']
                    globalPreviousFrame = frameIndices[frameIndices['frame']== previousFrame][frameIndices['segment']==sNumber].index[0]
                    print 'globalPreviousFrame, ', globalPreviousFrame
                    for i, pixel in enumerate(previousPixels):
                        cur_id = allIdentities.loc[globalPreviousFrame,i]
                        px = np.unravel_index(pixel,(height,width))
                        # print px
                        # print cur_id
                        # framePreviousShadows[px[0],px[1],:] = colors[cur_id]
                        # print colors
                        frame[px[0],px[1],:] = np.multiply(colors[cur_id],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                    if previousFrame > 0 and shadowsCounter <= 11:
                        previousFrame = previousFrame-1
                        shadowsCounter += 1
                    else:
                        break
                    # frameShadows = cv2.addWeighted(frameShadows,.5.,framePreviousShadows,.5.,0)

                for i, centroid in enumerate(centroids):
                    # print centroid
                    cur_id = allIdentities.loc[globalFrame,i]
                    px = np.unravel_index(pixels[i],(height,width))
                    frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                    cv2.putText(frame,str(cur_id),centroid, font, 1,colors[cur_id],2)
                    cv2.circle(frame, centroid,2, colors[cur_id],2)



                    # frame[px[0],px[1],:] = colors[cur_id]


        # blendFrame = cv2.addWeighted(frame,.5,frameShadows,.5,0)
        # print 'shape blend Frame, ', blendFrame.shape
        cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

        # Visualization of the process
        cv2.imshow('IdPlayer',frame)
        pass

    cv2.namedWindow('IdPlayer')
    print '*************************************************'
    cv2.createTrackbar( 'start', 'IdPlayer', 0, numFrame-1, onChange )
    # cv2.createTrackbar( 'end'  , 'IdPlayer', numFrame-1, numFrame, onChange )

    onChange(1)
    cv2.waitKey()

    start = cv2.getTrackbarPos('start','IdPlayer')

    return raw_input('Which segment do you want to inspect?')

finish = False
while not finish:
    print 'I am here', numSegment
    numSegment = IdPlayer(paths[int(numSegment)],allIdentities,frameIndices)
    if numSegment == 'q':
        finish = True
