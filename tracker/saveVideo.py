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
allFragProbIds = stats['probFragmentIds']

allIds = stats['blobIds']
allProbIds = stats['probBlobIds']

statistics = [allFragProbIds, allIds, allProbIds]


# path = paths[numSegment]
def IdSaver(paths,allIdentities,frameIndices,numAnimals,width,height, stat, show=True):

    path = paths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)


    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    name = folder +'/'+ filename.split('_')[0]  + '_tracked'+ '.avi'
    out = cv2.VideoWriter(name, fourcc, 15.0, (width, height))
    for path in paths:
        df, sNumber = loadFile(path, 'segmentation', time=0)
        sNumber = int(sNumber)
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        colors = get_spaced_colors(numAnimals)
        currentFrame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        while currentFrame < numFrame:
            centroids = df.loc[currentFrame,'centroids']
            pixels = df.loc[currentFrame,'pixels']
            permutation = df.loc[currentFrame,'permutation']

            #Get frame from video file
            ret, frame = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            frameCopy = frame.copy()
            # plot numbers if not crossing
            # globalFrame = frameIndices[frameIndices['frame']== trackbarValue][frameIndices['segment']==sNumber].index[0]
            globalFrame = frameIndices[frameIndices['frame']== currentFrame][frameIndices['segment']==sNumber].index[0]
            if not isinstance(permutation,float):
                # shadows
                if currentFrame > 0:
                    previousFrame = currentFrame -1
                    shadowsCounter = 1
                    # frameShadows = np.zeros_like(frame)
                    while not isinstance(df.loc[previousFrame,'permutation'],float):
                        # framePreviousShadows = np.zeros_like(frame)
                        previousPixels = df.loc[previousFrame,'pixels']
                        # globalPreviousFrame = frameIndices[frameIndices['frame']== previousFrame][frameIndices['segment']==sNumber].index[0]
                        globalPreviousFrame = frameIndices[frameIndices['frame']== previousFrame][frameIndices['segment']==sNumber].index[0]
                        # print globalPreviousFrame, allIdentities[globalPreviousFrame,:]
                        for i, pixel in enumerate(previousPixels):
                            cur_id = allIdentities[globalPreviousFrame,i]
                            if not sum(stat[globalFrame,i,:]):
                                cur_id = -1
                            px = np.unravel_index(pixel,(height,width))
                            frame[px[0],px[1],:] = np.multiply(colors[cur_id+1],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                        if previousFrame > 0 and shadowsCounter <= 11:
                            previousFrame = previousFrame-1
                            shadowsCounter += 1
                        else:
                            break

                    for i, centroid in enumerate(centroids):
                        # print centroid
                        cur_id = allIdentities[globalFrame,i]
                        thickness = 2
                        size = 1
                        if not sum(stat[globalFrame,i,:]):
                            cur_id = -1
                            thickness = 1
                            size = 0.5

                        px = np.unravel_index(pixels[i],(height,width))
                        frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                        cv2.putText(frame,str(cur_id+1),centroid, font, size,colors[cur_id+1],thickness)
                        cv2.circle(frame, centroid,2, colors[cur_id+1],2)

            cv2.putText(frame,'frame: ' + str(globalFrame),(300,50), font, 1,(0,0,255))
            cv2.putText(frame,'segment: ' + str(sNumber),(50,50), font, 1,(255,0,0))

            # Visualise or save? This is the question,...
            if show == True:
                cv2.imshow('IdPlayer',frame)
                k = cv2.waitKey(30) & 0xFF
                if k == 27: #pres esc to quit
                    break
            elif show==False:
                out.write(frame)
            else:
                ValueError('Set show to True to display, or False to save the video')
            currentFrame += 1
statNum = 0
IdSaver(paths,allFragIds,frameIndices,numAnimals,width,height, statistics[int(statNum)],show=False)
