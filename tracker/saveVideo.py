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
paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
frameIndices = pd.read_pickle('../Cafeina5peces/Caffeine5fish_frameIndices.pkl')
allIdentities = pd.read_pickle('../Cafeina5peces/Caffeine5fish_identities_new.pkl')
videoInfoPath = '../Cafeina5peces/Caffeine5fish_videoInfo.pkl'
# paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
# frameIndices = pd.read_pickle('../Conflict8/conflict3and4_frameIndices.pkl')
# allIdentities = pd.read_pickle('../Conflict8/conflict3and4_identities.pkl')

videoInfo = pd.read_pickle(videoInfoPath)
numAnimals = videoInfo['numAnimals']
width = videoInfo['width']
height = videoInfo['Height']

# path = paths[numSegment]

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(100, max_value, interval)]
    rgbcolorslist = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    return rgbcolorslist

def IdSaver(paths,allIdentities,frameIndices,numAnimals,width,height, show=True):
    path = paths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)


    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    name = folder +'/'+ filename.split('_')[0]  + '_tracked'+ '.avi'
    out = cv2.VideoWriter(name, fourcc, 15.0, (width, height))
    for path in paths:
        video = os.path.basename(path)
        filename, extension = os.path.splitext(video)
        sNumber = int(filename.split('_')[-1])
        folder = os.path.dirname(path)
        df = pd.read_pickle(folder +'/'+ filename + '.pkl')
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
                        globalPreviousFrame = frameIndices[frameIndices['frame']== previousFrame][frameIndices['segment']==sNumber].index[0]
                        for i, pixel in enumerate(previousPixels):
                            cur_id = allIdentities.loc[globalPreviousFrame,i]-1
                            px = np.unravel_index(pixel,(height,width))
                            frame[px[0],px[1],:] = np.multiply(colors[cur_id],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                        if previousFrame > 0 and shadowsCounter <= 11:
                            previousFrame = previousFrame-1
                            shadowsCounter += 1
                        else:
                            break

                    for i, centroid in enumerate(centroids):
                        # print centroid
                        cur_id = allIdentities.loc[globalFrame,i]-1
                        px = np.unravel_index(pixels[i],(height,width))
                        frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                        cv2.putText(frame,str(cur_id),centroid, font, 1,colors[cur_id],2)
                        cv2.circle(frame, centroid,2, colors[cur_id],2)

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
IdSaver(paths,allIdentities,frameIndices,numAnimals,width,height, show=True)
