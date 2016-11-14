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
paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')

frameIndices = loadFile(paths[0], 'frameIndices', time=0)
videoInfo = loadFile(paths[0], 'videoInfo', time=0)
videoInfo = videoInfo.to_dict()[0]
stats = loadFile(paths[0], 'statistics', time=0)
stats = stats.to_dict()[0]

numAnimals = videoInfo['numAnimals']
width = videoInfo['width']
height = videoInfo['height']

allFragIds = stats['fragmentIds']
print allFragIds[:50]
allFragProbIds = stats['probFragmentIds']

allIds = stats['blobIds']
allProbIds = stats['probBlobIds']

statistics = [allFragProbIds, allIds, allProbIds]

def IdPlayer(path,allIdentities,frameIndices, numAnimals, width, height, stat):
    plusOne = False # if stat are the identities' indices we will sum 1, because it is nicer
    if stat.dtype == 'int64':
        plusOne = True

    df, sNumber = loadFile(path, 'segmentation', time=0)
    sNumber = int(sNumber)
    cap = cv2.VideoCapture(path)
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    colors = get_spaced_colors(numAnimals)
    # print 'colors, ',colors
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
            # print 'pass'
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
                        cur_id = allIdentities[globalPreviousFrame,i]
                        px = np.unravel_index(pixel,(height,width))
                        # print px
                        # print cur_id
                        # framePreviousShadows[px[0],px[1],:] = colors[cur_id]
                        # print colors
                        # print cur_id
                        frame[px[0],px[1],:] = np.multiply(colors[cur_id+1],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                    if previousFrame > 0 and shadowsCounter <= 11:
                        previousFrame = previousFrame-1
                        shadowsCounter += 1
                    else:
                        break
                    # frameShadows = cv2.addWeighted(frameShadows,.5.,framePreviousShadows,.5.,0)

                for i, centroid in enumerate(centroids):
                    # print centroid
                    cur_id = allIdentities[globalFrame,i]
                    if plusOne:
                        cur_stat = stat[globalFrame,i]
                        fontSize = 1
                        text = str(cur_stat)
                        color = [0,0,0]
                        thickness = 2
                    else:
                        text = str(np.round(stat[globalFrame,i,:],decimals=2))
                        fontSize = .5
                        thickness = 1
                        color = [0,0,0]
                    if not sum(stat[globalFrame,i,:]):
                        cur_id = -1


                    px = np.unravel_index(pixels[i],(height,width))
                    frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                    cv2.putText(frame,text,centroid, font, fontSize,color,thickness)
                    cv2.putText(frame,str(cur_id+1),(centroid[0]-10,centroid[1]-10) , font, 1,colors[cur_id+1],2)
                    # cv2.circle(frame, centroid,2, colors[cur_id],2)



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

    numSegment =  raw_input('Which segment do you want to inspect?')
    statNum = raw_input('Which statistics do you wanna visualize (allFragProbIds, allIds, allProbIds)?')
    return numSegment, statNum
    # return raw_input('Which statistics do you wanna visualize (0,1,2,3)?')

finish = False
statNum = 2
while not finish:
    print 'I am here', numSegment
    numSegment, statNum = IdPlayer(paths[int(numSegment)],allFragIds,frameIndices, numAnimals, width, height,statistics[int(statNum)])
    if numSegment == 'q':
        finish = True
