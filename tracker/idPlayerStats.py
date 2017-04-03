import cv2
import sys
sys.path.append('IdTrackerDeep/utils')

from py_utils import *
from video_utils import *

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


videoPaths = scanFolder('IdTrackerDeep/videos/59indiv/video_03-23-17_11-36-23.000.avi')

frameIndices, segmPaths = getSegmPaths(videoPaths)
videoPath = videoPaths[0]

videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')

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
print sessionPath

stats = pickle.load( open( sessionPath + "/statistics.pkl", "rb" ) )
dfGlobal = loadFile(videoPaths[0], 'portraits')

numAnimals = videoInfo['numAnimals']
width = videoInfo['width']
height = videoInfo['height']

allFragIds = stats['fragmentIds']
allFragProbIds = stats['probFragmentIds']

allIds = stats['blobIds']
allProbIds = stats['probBlobIds']
FreqFrag = stats['FreqFrag']
normFreqFrag = stats['normFreqFragAllVideo']
P1Frag = stats['P1Frag']
P2Frag = stats['P2FragAllVideo']

statistics = [allFragProbIds, allIds, allProbIds, FreqFrag, normFreqFrag, P1Frag,P2Frag]

def IdPlayer(videoPaths,segmPaths,allIdentities,frameIndices, numAnimals, width, height, stat,statistics,dfGlobal, show):

    # Visualise or save? This is the question,...
    # if show == True:
    #     cv2.imshow('IdPlayer',frame)
    #     m = cv2.waitKey(30) & 0xFF
    #     if m == 27: #pres esc to quit
    #         break
    # elif show==False:
    #     out.write(frame)
    # else:
    #     ValueError('Set show to True to display, or False to save the video')
    # currentFrame += 1

    # Load statistics
    freq = statistics[3]
    normFreq = statistics[4]
    P1 = statistics[5]
    logP2 = statistics[0]
    P2 = statistics[6]
    softmaxProb = statistics[2]
    statIdentity = False # if stat are the identities' indices we will sum 1, because it is nicer
    if stat.dtype == 'int64':
        statIdentity = True

    if not show:
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        name = folder +'/'+ filename.split('_')[0]  + '_tracked'+ '.avi'
        out = cv2.VideoWriter(name, fourcc, 15.0, (width, height))

    # Load first segment
    global segmDf, cap, currentSegment, numFrameCurSegment
    segmDf, sNumber = loadFile(segmPaths[0], 'segmentation')
    currentSegment = int(sNumber)
    cap = cv2.VideoCapture(videoPaths[0])
    numFrames = len(frameIndices)
    numFrameCurSegment = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    colors = get_spaced_colors_util(numAnimals)

    def onChange(trackbarValue):
        global segmDf, cap, currentSegment, numFrameCurSegment

        # Select segment dataframe and change cap if needed
        sNumber = frameIndices.loc[trackbarValue,'segment']
        sFrame = frameIndices.loc[trackbarValue,'frame']
        # print len(videoPaths)
        # print sNumber
        if sNumber != currentSegment: # we are changing segment
            print 'Changing segment...'
            prevSegmDf, _ = loadFile(segmPaths[sNumber-2], 'segmentation')
            segmDf, _ = loadFile(segmPaths[sNumber-1], 'segmentation')
            currentSegment = sNumber

            if len(videoPaths) > 1:
                cap = cv2.VideoCapture(videoPaths[sNumber-1])

        #Get frame from video file
        if len(videoPaths) > 1:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,sFrame)
        else:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)

        ret, frame = cap.read()
        print ret
        font = cv2.FONT_HERSHEY_SIMPLEX
        frameCopy = frame.copy()

        print '**********************************'
        print 'sNumber, ', sNumber
        print 'sFrame, ', sFrame
        print 'trackbarValue, ', trackbarValue
        print '----------------------------------'

        noses = dfGlobal.loc[trackbarValue,'noses']
        centroids = dfGlobal.loc[trackbarValue,'centroids']
        pixels = segmDf.loc[sFrame,'pixels']
        permutation = segmDf.loc[sFrame,'permutation']

        # Plot segmentated blobs
        if not isinstance(permutation,float):
            # shadows
            if trackbarValue == 0:
                print 'Previous frame set 0 because we are in the first frame'
                previousSegFrame = 0
                previousGlobFrame = 0
            else:
                previousSegFrame = sFrame - 1
                previousGlobFrame = trackbarValue - 1

                if previousSegFrame < 0:
                    previousSegFrame = len(prevSegmDf)+previousSegFrame

            print '----------------------------------'
            print 'sNumber, ', sNumber
            print 'sFrame, ', sFrame
            print 'previousSegFrame, ', previousSegFrame
            print 'globFrame, ', trackbarValue
            print 'previousGlobFrame, ', previousGlobFrame


            shadowsCounter = 1
            # frameShadows = np.zeros_like(frame)
            # print '----------------------------------'
            # print 'Drawing sadows...'
            while not isinstance(segmDf.loc[previousSegFrame,'permutation'],float):
                # framePreviousShadows = np.zeros_like(frame)
                if previousSegFrame > sFrame:
                    previousPixels = prevSegmDf.loc[previousSegFrame,'pixels']
                else:
                    previousPixels = segmDf.loc[previousSegFrame,'pixels']

                for i, pixel in enumerate(previousPixels):
                    cur_id = allIdentities[previousGlobFrame,i]
                    px = np.unravel_index(pixel,(height,width))
                    frame[px[0],px[1],:] = np.multiply(colors[cur_id+1],.3).astype('uint8')+np.multiply(frame[px[0],px[1],:],.7).astype('uint8')
                if previousSegFrame > 0 and shadowsCounter <= 11:
                    previousSegFrame = previousSegFrame - 1
                    previousGlobFrame = previousGlobFrame - 1
                    # print '----------------------------------'
                    # print 'previousSegFrame, ', previousSegFrame
                    # print 'previousGlobFrame, ', previousGlobFrame
                    shadowsCounter += 1
                else:
                    # print 'I finish drawing all the shadows'
                    break

            print '**********************************'

            for i, (centroid,nose) in enumerate(zip(centroids,noses)):
                # print centroid
                cur_id = allIdentities[trackbarValue,i]
                if statIdentity:
                    cur_stat = stat[trackbarValue,i]
                    fontSize = .5
                    text = str(cur_stat)
                    color = [0,0,0]
                    thickness = 1
                else:
                    # text = '{:.2f}'.format(np.round(stat[globalFrame,i,:],decimals=2))
                    textList = ["%.2f" % float(np.round(s,decimals=2)) for s in stat[trackbarValue,i,:]]
                    text = str.join(", ",textList)
                    text = '[ ' + text + ' ]'
                    fontSize = .5
                    thickness = 1
                    color = [0,0,0]

                if show:
                    freqList = ["%0.f" % float(s) for s in freq[trackbarValue,i,:]]
                    freqText = str.join(", ",freqList)
                    freqText = '[ ' + freqText + ' ]'
                    normFreqList = ["%0.2f" % float(s) for s in normFreq[trackbarValue,i,:]]
                    normFreqText = str.join(", ",normFreqList)
                    normFreqText = '[ ' + normFreqText + ' ]'
                    P1List = ["%.4f" % float(s) for s in P1[trackbarValue,i,:]]
                    P1Text = str.join(", ",P1List)
                    P1Text = '[ ' + P1Text + ' ]'
                    logP2List = ["%.4f" % float(s) for s in logP2[trackbarValue,i,:]]
                    logP2Text = str.join(", ",logP2List)
                    logP2Text = '[ ' + logP2Text + ' ]'
                    P2List = ["%.8f" % float(s) for s in P2[trackbarValue,i,:]]
                    P2Text = str.join(", ",P2List)
                    P2Text = '[ ' + P2Text + ' ]'
                    softmaxProbList = ["%.8f" % float(s) for s in softmaxProb[trackbarValue,i,:]]
                    softmaxProbText = str.join(", ",softmaxProbList)
                    softmaxProbText = '[ ' + softmaxProbText + ' ]'
                    print '--------- Id, ', cur_id+1
                    # print 'Frequencies', freqText
                    print 'normFrequencies', normFreqText
                    # print 'P1, ', P1Text
                    print 'P2, ', P2Text
                    print 'softmaxProb, ', softmaxProbText
                    # print 'logP2, ', logP2Text
                    # if not sum(stat[globalFrame,i,:]):
                    #     cur_id = -1


                px = np.unravel_index(pixels[i],(height,width))
                frame[px[0],px[1],:] = frameCopy[px[0],px[1],:]
                # cv2.putText(frame,text,centroid, font, fontSize,color,thickness)
                cv2.putText(frame,str(cur_id+1),(centroid[0]-10,centroid[1]-10) , font, 3,colors[cur_id+1],3)
                cv2.circle(frame, centroid,2, colors[cur_id+1],2)
                cv2.circle(frame, nose,2, colors[cur_id+1],2)


        # blendFrame = cv2.addWeighted(frame,.5,frameShadows,.5,0)
        # print 'shape blend Frame, ', blendFrame.shape
        cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

        if show == True:
            frame = cv2.resize(frame,None, fx = np.true_divide(1,4), fy = np.true_divide(1,4))
            # Visualization of the process
            cv2.imshow('IdPlayer',frame)
            pass
        else:
            out.write(frame)

    cv2.namedWindow('IdPlayer')
    cv2.createTrackbar( 'start', 'IdPlayer', 0, numFrames-1, onChange )

    if show == True:
        onChange(0)
        cv2.waitKey()
    else:
        for i in range(numFrames):
            print 'numFrames, ', numFrames
            print 'currentFrame, ', i
            onChange(i)

    return
    # return raw_input('Which statistics do you wanna visualize (0,1,2,3)?')

statNum = 4
IdPlayer(videoPaths,segmPaths,allFragIds,frameIndices, numAnimals, width, height,statistics[int(statNum)],statistics,dfGlobal,show=False)
