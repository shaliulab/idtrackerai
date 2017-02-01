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
# paths = scanFolder('../Cafeina5pecesLarge/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
# paths = scanFolder('../Medaka/20fish_20130909T191651_1.avi')
# paths = scanFolder('../Cafeina5pecesSmall/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('../BigGroup/manyFish_26dpf_20161110_1.avi')
paths = scanFolder('../38fish_adult_splitted/adult1darkenes_1.avi')

frameIndices = loadFile(paths[0], 'frameIndices', time=0)
videoInfo = loadFile(paths[0], 'videoInfo', time=0, hdfpkl='pkl')
stats = pickle.load( open( ckpt_dir + "/statistics.pkl", "rb" ) )
dfGlobal = loadFile(paths[0], 'portraits', time=0)

numAnimals = videoInfo['numAnimals']
width = videoInfo['width']
height = videoInfo['height']

allFragIds = stats['fragmentIds']
allFragProbIds = stats['probFragmentIds']

allIds = stats['blobIds']
allProbIds = stats['probBlobIds']

statistics = [allFragProbIds, allIds, allProbIds]


# path = paths[numSegment]
def IdSaver(paths,allIdentities,frameIndices,numAnimals,width,height, stat, dfGlobal,show=True,blackBkg = False):

    path = paths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    shadowsNumber = 9

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    name = folder +'/'+ filename.split('_')[0]  + '_tracked'+ '.avi'
    out = cv2.VideoWriter(name, fourcc, 15.0, (width, height))
    for i, path in enumerate(paths):
        print i
        df, sNumber = loadFile(path, 'segmentation', time=0)

        # allPortraits = allPortraits.sort_index(axis=0,ascending=True)
        sNumber = int(sNumber)
        if sNumber > 0:
            dfpast, _ = loadFile(paths[i-1], 'segmentation', time=0)
            numFramesPast = len(dfpast)
            for j in range(1,shadowsNumber):
                df.loc[-j] = dfpast.iloc[-j]
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        colors = get_spaced_colors(numAnimals)
        currentFrame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        while currentFrame < numFrame:
            ind = frameIndices[(frameIndices.segment == int(sNumber)) & (frameIndices.frame == currentFrame)].index[0]
            noses = dfGlobal.loc[ind,'noses']
            centroids = df.loc[currentFrame,'centroids']
            pixels = df.loc[currentFrame,'pixels']
            permutation = df.loc[currentFrame,'permutation']

            #Get frame from video file
            ret, frame = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            frameCopy = frame.copy()
            if blackBkg == True:
                frame = np.zeros_like(frame).astype('uint8')
            # plot numbers if not crossing
            # globalFrame = frameIndices[frameIndices['frame']== trackbarValue][frameIndices['segment']==sNumber].index[0]
            globalFrame = frameIndices[frameIndices['frame']== currentFrame][frameIndices['segment']==sNumber].index[0]
            if not isinstance(permutation,float):
                # shadows
                if currentFrame > 0 and sNumber == 1:
                    previousFrame = currentFrame -1
                    shadowsCounter = 1
                    # frameShadows = np.zeros_like(frame)
                    if blackBkg == False:
                        while not isinstance(df.loc[previousFrame,'permutation'],float):

                            # framePreviousShadows = np.zeros_like(frame)
                            previousPixels = df.loc[previousFrame,'pixels']
                            globalPreviousFrame = frameIndices[frameIndices['frame']== previousFrame][frameIndices['segment']==sNumber].index[0]
                            for k, pixel in enumerate(previousPixels):
                                cur_id = allIdentities[globalPreviousFrame,k]
                                if cur_id == 0 or cur_id == 14:
                                    px = np.unravel_index(pixel,(height,width))
                                    frame[px[0],px[1],:] = np.multiply(colors[cur_id+1],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                            if previousFrame > 0 and shadowsCounter <= shadowsNumber:
                                previousFrame = previousFrame-1
                                shadowsCounter += 1
                            else:
                                break

                    for l, (centroid,nose) in enumerate(zip(centroids,noses)):
                        # print centroid
                        cur_id = allIdentities[globalFrame,l]
                        thickness = 5
                        size = 2
                        if not sum(stat[globalFrame,l,:]):
                            cur_id = -1
                            thickness = 1
                            size = 0.5

                        if blackBkg == False:
                            px = np.unravel_index(pixels[l],(height,width))
                            frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                            if cur_id == 0:
                                cv2.putText(frame,'Mattia',centroid, font, size,colors[cur_id+1],thickness)
                            elif cur_id == 14:
                                cv2.putText(frame,'Paco',centroid, font, size,colors[cur_id+1],thickness)
                        #     elif cur_id == 14:
                        #         cv2.putText(frame,'Tom',centroid, font, size,colors[cur_id+1],thickness)
                        #     else:
                        #         cv2.putText(frame,str(cur_id+1),centroid, font, size,colors[cur_id+1],thickness)
                        # cv2.circle(frame, centroid,2, colors[cur_id+1],2)
                        # cv2.circle(frame, nose,2, colors[cur_id+1],2)

                elif sNumber > 1:
                    previousFrame = currentFrame -1

                    shadowsCounter = 1
                    # frameShadows = np.zeros_like(frame)
                    # print df.index
                    if blackBkg == False:
                        while not isinstance(df.loc[previousFrame,'permutation'],float):
                            # framePreviousShadows = np.zeros_like(frame)
                            previousPixels = df.loc[previousFrame,'pixels']
                            if previousFrame < 0:
                                globalPreviousFrame = frameIndices[frameIndices['frame'] == numFramesPast+previousFrame][frameIndices['segment']==sNumber-1].index[0]
                            else:
                                globalPreviousFrame = frameIndices[frameIndices['frame'] == previousFrame][frameIndices['segment']==sNumber].index[0]
                            # print globalPreviousFrame, allIdentities[globalPreviousFrame,:]
                            for k, pixel in enumerate(previousPixels):
                                cur_id = allIdentities[globalPreviousFrame,k]
                                if cur_id == 0 or cur_id == 14:
                                    px = np.unravel_index(pixel,(height,width))
                                    frame[px[0],px[1],:] = np.multiply(colors[cur_id+1],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                            if previousFrame > -(shadowsNumber-1) and shadowsCounter <= shadowsNumber:
                                previousFrame = previousFrame-1
                                shadowsCounter += 1
                            else:
                                break
                    for l, (centroid,nose) in enumerate(zip(centroids,noses)):
                        # print centroid
                        cur_id = allIdentities[globalFrame,l]
                        thickness = 5
                        size = 2
                        if not sum(stat[globalFrame,l,:]):
                            cur_id = -1
                            thickness = 1
                            size = 0.5
                        if blackBkg == False:
                            px = np.unravel_index(pixels[l],(height,width))
                            frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                            if cur_id == 0:
                                cv2.putText(frame,'Mattia',centroid, font, size,colors[cur_id+1],thickness)
                            elif cur_id == 14:
                                # cv2.putText(frame,'Paco',centroid, font, size,colors[cur_id+1],thickness)
                                cv2.putText(frame,'Paco',centroid, font, size,colors[cur_id+1],thickness)
                        #     elif cur_id == 14:
                        #         cv2.putText(frame,'Tom',centroid, font, size,colors[cur_id+1],thickness)
                        #     else:
                        #         cv2.putText(frame,str(cur_id+1),centroid, font, size,colors[cur_id+1],thickness)
                        # cv2.circle(frame, centroid,2, colors[cur_id+1],2)
                        # cv2.circle(frame, nose,2, colors[cur_id+1],2)


            # cv2.putText(frame,'frame: ' + str(globalFrame),(300,50), font, 1,(0,0,255))
            # cv2.putText(frame,'segment: ' + str(sNumber),(50,50), font, 1,(255,0,0))

            # Visualise or save? This is the question,...
            if show == True:
                cv2.imshow('IdPlayer',frame)
                m = cv2.waitKey(30) & 0xFF
                if m == 27: #pres esc to quit
                    break
            elif show==False:
                out.write(frame)
            else:
                ValueError('Set show to True to display, or False to save the video')
            currentFrame += 1
statNum = 0
IdSaver(paths,allFragIds,frameIndices,numAnimals,width,height, statistics[int(statNum)],dfGlobal,show=False,blackBkg=False)
