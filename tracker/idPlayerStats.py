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
# path = selectFile()
# paths = scanFolder(path)

# paths = scanFolder('../videos/Cafeina5pecesLarge/Caffeine5fish_20140206T122428_1.avi')
paths = scanFolder('../videos/motherfucker2/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('/media/chaos/New Volume/cafeina5pecesSmall/Caffeine5fish_20140206T122428_1.avi')
paths = scanFolder('/media/chaos/New Volume/motherfucker2/Caffeine5fish_20140206T122428_1.avi')
# '/media/chaos/New Volume/motherfucker2/Caffeine5fish_20140206T122428_1.avi'
# paths = scanFolder('../Cafeina5pecesLarge/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('../larvae1/trial_1_1.avi')
# paths = scanFolder('../nofragsError/_1.avi')
# paths = scanFolder('../videos/Conflict8/conflict3and4_20120316T155032_1.avi')
# paths = scanFolder('../Medaka/20fish_20130909T191651_1.avi')
# paths = scanFolder('../Cafeina5pecesSmall/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('../38fish_adult_splitted/adult1darkenes_1.avi')
# paths = scanFolder('/home/lab/Desktop/aggr/video_4/4.avi')
# print paths
videoPath = paths[0]
frameIndices = loadFile(videoPath, 'frameIndices')
videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')
# stats = loadFile(paths[0], 'statistics',hdfpkl='pkl')
def getLastSession(subFolders):
    if len(subFolders) == 0:
        lastIndex = 0
    else:
        subFolders = natural_sort(subFolders)[::-1]
        lastIndex = int(subFolders[0].split('_')[-1])
    return lastIndex

video = os.path.basename(videoPath)
folder = os.path.dirname(videoPath)
filename, extension = os.path.splitext(video)
subFolder = folder + '/CNN_models'
subSubFolders = glob.glob(subFolder +"/*")
lastIndex = getLastSession(subSubFolders)
sessionPath = subFolder + '/Session_' + str(lastIndex)

stats = pickle.load( open( sessionPath + "/statistics.pkl", "rb" ) )
# stats = loadFile(paths[0], 'statistics', time=0)
# stats = stats.to_dict()[0]
dfGlobal = loadFile(paths[0], 'portraits')
# IdsStatistics = {'blobIds':idSoftMaxAllVideo,
#     'probBlobIds':PSoftMaxAllVIdeo,
#     'fragmentIds':idLogP2FragAllVideo,
#     'probFragmentIds':logP2FragAllVideo,
#     'FreqFrag': freqFragAllVideo,
#     'P1Frag': P1FragAllVideo}

numAnimals = videoInfo['numAnimals']
width = videoInfo['width']
height = videoInfo['height']

allFragIds = stats['fragmentIds']
print allFragIds[:50]
allFragProbIds = stats['probFragmentIds']

allIds = stats['blobIds']
allProbIds = stats['probBlobIds']
FreqFrag = stats['FreqFrag']
normFreqFrag = stats['normFreqFragAllVideo']
P1Frag = stats['P1Frag']
P2Frag = stats['P2FragAllVideo']

statistics = [allFragProbIds, allIds, allProbIds, FreqFrag, normFreqFrag, P1Frag,P2Frag]

def IdPlayer(path,allIdentities,frameIndices, numAnimals, width, height, stat,statistics,dfGlobal):
    freq = statistics[3]
    normFreq = statistics[4]
    P1 = statistics[5]
    logP2 = statistics[0]
    P2 = statistics[6]
    statIdentity = False # if stat are the identities' indices we will sum 1, because it is nicer
    if stat.dtype == 'int64':
        statIdentity = True

    df, sNumber = loadFile(path, 'segmentation')
    sNumber = int(sNumber)
    cap = cv2.VideoCapture(path)
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    colors = get_spaced_colors_util(numAnimals)
    # print 'colors, ',colors
    def onChange(trackbarValue):
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        index = frameIndices[(frameIndices.segment == int(sNumber)) & (frameIndices.frame == trackbarValue)].index[0]
        noses = dfGlobal.loc[index,'noses']
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
        # print 'permutation, ', permutation
        if not isinstance(permutation,float):
            # print 'pass'
            # shadows
            if trackbarValue == 0:
                print 'Previous frame set 0 because we are in the first frame'
                previousFrame = 0
            else:
                previousFrame = trackbarValue -1
            shadowsCounter = 1
            # frameShadows = np.zeros_like(frame)
            while not isinstance(df.loc[previousFrame,'permutation'],float):
                # framePreviousShadows = np.zeros_like(frame)
                previousPixels = df.loc[previousFrame,'pixels']
                globalPreviousFrame = frameIndices[frameIndices['frame']== previousFrame][frameIndices['segment']==sNumber].index[0]
                # print 'globalPreviousFrame, ', globalPreviousFrame
                for i, pixel in enumerate(previousPixels):
                    cur_id = allIdentities[globalPreviousFrame,i]
                    px = np.unravel_index(pixel,(height,width))
                    frame[px[0],px[1],:] = np.multiply(colors[cur_id+1],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                if previousFrame > 0 and shadowsCounter <= 11:
                    previousFrame = previousFrame-1
                    shadowsCounter += 1
                else:
                    break

            print '\n *************** global frame, ', globalFrame
            for i, (centroid,nose) in enumerate(zip(centroids,noses)):
                # print centroid
                cur_id = allIdentities[globalFrame,i]
                if statIdentity:
                    cur_stat = stat[globalFrame,i]
                    fontSize = .5
                    text = str(cur_stat)
                    color = [0,0,0]
                    thickness = 1
                else:
                    # text = '{:.2f}'.format(np.round(stat[globalFrame,i,:],decimals=2))
                    textList = ["%.2f" % float(np.round(s,decimals=2)) for s in stat[globalFrame,i,:]]
                    text = str.join(", ",textList)
                    text = '[ ' + text + ' ]'
                    fontSize = .5
                    thickness = 1
                    color = [0,0,0]
                freqList = ["%0.f" % float(s) for s in freq[globalFrame,i,:]]
                freqText = str.join(", ",freqList)
                freqText = '[ ' + freqText + ' ]'
                normFreqList = ["%0.2f" % float(s) for s in normFreq[globalFrame,i,:]]
                normFreqText = str.join(", ",normFreqList)
                normFreqText = '[ ' + normFreqText + ' ]'
                P1List = ["%.4f" % float(s) for s in P1[globalFrame,i,:]]
                P1Text = str.join(", ",P1List)
                P1Text = '[ ' + P1Text + ' ]'
                logP2List = ["%.4f" % float(s) for s in logP2[globalFrame,i,:]]
                logP2Text = str.join(", ",logP2List)
                logP2Text = '[ ' + logP2Text + ' ]'
                P2List = ["%.8f" % float(s) for s in P2[globalFrame,i,:]]
                P2Text = str.join(", ",P2List)
                P2Text = '[ ' + P2Text + ' ]'
                print '--------- Id, ', cur_id+1
                # print 'Frequencies', freqText
                print 'normFrequencies', normFreqText
                # print 'P1, ', P1Text
                print 'P2, ', P2Text
                # print 'logP2, ', logP2Text
                # if not sum(stat[globalFrame,i,:]):
                #     cur_id = -1


                px = np.unravel_index(pixels[i],(height,width))
                frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                # cv2.putText(frame,text,centroid, font, fontSize,color,thickness)
                cv2.putText(frame,str(cur_id+1),(centroid[0]-10,centroid[1]-10) , font, 1,colors[cur_id+1],2)
                cv2.circle(frame, centroid,2, colors[cur_id+1],2)
                cv2.circle(frame, nose,2, colors[cur_id+1],2)



                    # frame[px[0],px[1],:] = colors[cur_id]


        # blendFrame = cv2.addWeighted(frame,.5,frameShadows,.5,0)
        # print 'shape blend Frame, ', blendFrame.shape
        cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))
        frame = cv2.resize(frame,None, fx = np.true_divide(1,1), fy = np.true_divide(1,1))
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

    # numSegment =  raw_input('Which segment do you want to inspect?')
    numSegment = getInput('Segment number','Type the segment to be visualized')
    # statNum = raw_input('Which statistics do you wanna visualize (allFragProbIds, allIds, allProbIds)?')
    statNum = getInput('Stats','Which statistics do you wanna visualize (0-allFragProbIds, 1-allIds, 2-allProbIds, 3-FreqFrag)?')
    return numSegment, statNum
    # return raw_input('Which statistics do you wanna visualize (0,1,2,3)?')

finish = False
statNum = 4
while not finish:
    print 'I am here', numSegment
    numSegment, statNum = IdPlayer(paths[int(numSegment)],allFragIds,frameIndices, numAnimals, width, height,statistics[int(statNum)],statistics,dfGlobal)
    if numSegment == 'q':
        finish = True
