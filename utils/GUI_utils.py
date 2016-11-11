import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from segmentation_ROIPreview import *
from fragmentation import *
from get_portraits import *
from video_utils import *
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
import math
from natsort import natsorted, ns
from os.path import isdir, isfile
import scipy.spatial.distance as scisd
from tkFileDialog import askopenfilename

''' ****************************************************************************
ROI selector GUI
*****************************************************************************'''
def ROIselector(frame):
    plt.ion()
    f, ax = plt.subplots()
    ax.imshow(frame, interpolation='nearest', cmap='gray')
    props = {'facecolor': '#000070',
             'edgecolor': 'white',
             'alpha': 0.3}
    rect_tool = RectangleTool(ax, rect_props=props)
    N = 2
    params = plt.gcf()
    plSize = params.get_size_inches()
    params.set_size_inches( (plSize[0]*N, plSize[1]*N) )
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    # thismanager = plt.get_current_fig_manager()
    # thismanager.window.SetPosition((500, 0))
    plt.show()

    numROIs = getInput('Number of ROIs','Type the number of ROIs to be selected')
    numROIs = int(numROIs)
    print 'The number of ROIs to select is ', numROIs
    counter = 0
    ROIsCoords = []
    centers = []
    ROIsShapes = []
    mask = np.ones_like(frame,dtype='uint8')*255
    while counter < numROIs:
        ROIshape = getInput('Roi shape','r= rect, c=circ')

        if ROIshape == 'r' or ROIshape == 'c':
            ROIsShapes.append(ROIshape)

            rect_tool.callback_on_enter(rect_tool.extents)
            coord = np.asarray(rect_tool.extents).astype('int')

            print 'ROI coords, ', coord
            text = 'Is ' + str(coord) + ' the ROI you wanted to select? y/n'
            goodROI = getInput('Confirm selection',text)
            if goodROI == 'y':
                ROIsCoords.append(coord)
                if ROIshape == 'r':
                    cv2.rectangle(mask,(coord[0],coord[2]),(coord[1],coord[3]),0,-1)
                    centers.append(None)
                if ROIshape == 'c':
                    center = ((coord[1]+coord[0])/2,(coord[3]+coord[2])/2)
                    angle = 90
                    axes = tuple(sorted(((coord[1]-coord[0])/2,(coord[3]-coord[2])/2)))
                    print center, angle, axes
                    cv2.ellipse(mask,center,axes,angle,0,360,0,-1)
                    centers.append(center)

        counter = len(ROIsCoords)
    plt.close("all")

    return mask, centers

def checkROI(useROI, usePreviousROI, frame, videoPath):
    ''' Select ROI '''
    if useROI:
        if usePreviousROI:
            mask = loadFile(videoPath, 'ROI',0)
            mask = np.asarray(mask)
            centers= loadFile(videoPath, 'centers',0)
            centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
        else:
            print '\n Selecting ROI ...'
            mask, centers = ROIselector(frame)
    else:
        print '\n No ROI selected ...'
        mask = np.zeros_like(frame)
        centers = []
    return mask, centers

''' ****************************************************************************
First preview for ROI, numAnimals, inspect segmentation,
**************************************************************************** '''

def playPreview(paths, useBkg, usePreviousBkg, useROI, usePreviousROI, numSegment=0):
    """
    loads a preview of the video for manual fine-tuning
    """
    print '\n'
    print '***** Starting playPreview to selectROI and Bkg...'
    # global numSegment
    # width, height = getVideoInfo(paths)
    # video = os.path.basename(paths[0])
    # folder = os.path.dirname(paths[0])
    # filename, extension = os.path.splitext(video)
    # subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    # subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
    # subFolder = subFolders[0]

    cap2 = cv2.VideoCapture(paths[0])
    flag, frame = cap2.read()
    cap2.release()
    height = frame.shape[0]
    width = frame.shape[1]

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, centers = checkROI(useROI, usePreviousROI, frameGray, paths[0])
    saveFile(paths[0], mask, 'ROI',time = 0)
    saveFile(paths[0], centers, 'centers',time = 0)
    bkg = checkBkg(useBkg, usePreviousBkg, paths, 0, width, height)

    return width, height, bkg, mask, centers

''' ****************************************************************************
Segmentation inspector
**************************************************************************** '''

# def SegmentationPreview(path, width, height, bkg, maxIntensity, maxBkg, mask, useBkg, minArea = 150, maxArea = 1000, minThreshold = 0.85, maxThreshold = 2, size = 1):
#     print '\n'
#     print '*****Entering segmentation preview...'
#     numAnimals = getInput('Number of animals','Type the number of animals')
#     numAnimals = int(numAnimals)
#     global maxRangeTh, maxTB
#     maxTB = 200
#     if bkg == None:
#         maxRangeTh = maxIntensity
#     else:
#         maxRangeTh = maxBkg
#     print 'maxRangeTh, ', maxRangeTh
#     maxThreshold = maxRangeTh ###NOTE this is because we are not using the maxThreshold and we set it to the maximum for it not to affect the segmentation
#     # print 'Ready to get the cap from the path'
#     cap = cv2.VideoCapture(path)
#     print 'path GUI_utils line 161,    ', path
#     # print 'Ready to get eh number of frames'
#     numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#     print 'numFrame GUI_utils line 163   ', numFrame
#
#     def TB2th(TB,maxRangeTh,maxTB):
#         th = maxRangeTh * TB / maxTB
#         return th
#     def th2TB(th,maxRangeTh,maxTB):
#         TB = maxTB * th / maxRangeTh
#         return int(TB)
#
#     def thresholder(minTh, maxTh):
#         global frameValue, maxRangeTh, maxTB
#         minTh = TB2th(minTh,maxRangeTh,maxTB)
#         maxTh = TB2th(maxTh,maxRangeTh,maxTB)
#         # print 'I am in thresholder'
#         #threshold the frame, find contours and get portraits of the fish
#         toile = np.zeros_like(frameGray, dtype='uint8')
#         # print 'I am going to call segmentVideo'
#         segmentedFrame = segmentVideo(origFrame, minTh, maxTh, bkg, mask, useBkg)
#         # print '1'
#         contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         # contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#         # print '2'
#         maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
#         minArea = cv2.getTrackbarPos('minArea', 'Bars')
#         # minArea = 250
#         # print '3'
#         # print '*********************_______________maxArea__________________', maxArea
#         # print '*********************_______________minArea__________________', minArea
#         goodContours = filterContoursBySize(contours,minArea, maxArea)
#         cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
#         shower = cv2.addWeighted(origFrame,1,toile,.5,0)
#         showerCopy = shower.copy()
#         resUp = cv2.getTrackbarPos('ResUp', 'Bars')
#         resDown = cv2.getTrackbarPos('ResDown', 'Bars')
#         # print '4'
#         # showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
#         # showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))
#
#         bbs, miniFrames, _, _, _, bkgSamples = getBlobsInfoPerFrame(origFrame, goodContours, height, width)
#         # print '5'
#         numColumns = 5
#         numGoodContours = len(goodContours)
#         # print '**************************', numGoodContours
#         numBlackPortraits = numColumns - numGoodContours % numColumns
#         numPortraits = numGoodContours + numBlackPortraits
#         # print '6'
#         j = 0
#         sizePortrait = 32
#         portraitsMat = []
#         rowPortrait = []
#         # print '****** new frame *******, ', frameValue
#         ### Uncomment to plot
#         # plt.ion()
#         # plt.close("all")
#         # plt.figure()
#         while j < numPortraits:
#             # print "portrait, ", j
#             if j < numGoodContours:
#                 # print 'good'
#                 # print miniFrames
#                 # portrait, curvature, cnt,maxCoord, sorted_locations = getPortrait(miniFrames[j],goodContours[j],bbs[j],bkgSamples[j],frameValue)
#                 portrait = getPortrait(miniFrames[j],goodContours[j],bbs[j],bkgSamples[j],frameValue)
#                 portrait = np.squeeze(portrait)
#                 ### Uncomment to plot
#                 # plt.subplot(2,8,j+1)
#                 # plt.plot(curvature)
#                 # plt.scatter(sorted_locations[0],curvature[sorted_locations[0]],c='y',s=30)
#                 # plt.scatter(sorted_locations[1],curvature[sorted_locations[1]],c='r',s=30)
#                 #
#                 # plt.subplot(2,8,j+8+1)
#                 # plt.plot(cnt[:,0],cnt[:,1],'-')
#                 # plt.scatter(maxCoord[0][0],maxCoord[0][1],c='y',s=30)
#                 # plt.scatter(maxCoord[1][0],maxCoord[1][1],c='r',s=30)
#                 # plt.axis('equal')
#
#
#             else:
#                 # print 'black'
#                 portrait = np.zeros((sizePortrait,sizePortrait),dtype='uint8')
#             rowPortrait.append(portrait)
#             if (j+1) % numColumns == 0:
#                 # print 'length rowPortrait, ', len(rowPortrait)
#                 # print 'shape one portraits, '
#                 # for portrait in rowPortrait:
#                 #     print portrait.shape
#                 portraitsMat.append(np.hstack(rowPortrait))
#                 rowPortrait = []
#             j += 1
#         # print '7'
#         plt.show()
#         plt.pause(.5)
#
#         portraitsMat = np.vstack(portraitsMat)
#         #show window containing the trackbars
#         # print '8'
#         cv2.imshow('Bars',np.squeeze(portraitsMat))
#         #show frame, and react to changes in Bars
#         # print '9'
#         cv2.imshow('IdPlayer', showerCopy)
#         cv2.moveWindow('Bars', 10,10 )
#         cv2.moveWindow('IdPlayer', 200, 10 )
#         # print '10'
#
#     def scroll(trackbarValue):
#         global avFrame, frameGray, origFrame, frameValue
#         frameValue = trackbarValue
#         # print 'setting frame position inside scroll function'
#         cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
#         #Get frame from video file
#         ret, frame = cap.read()
#
#         #Color to gray scale
#         frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         #make a copy of frameGray
#         origFrame = frameGray.copy()
#         #average frame
#         avIntensity = frameAverager(origFrame)
#         # avFrameCopy = avFrame.copy()
#
#         # font = cv2.FONT_HERSHEY_SIMPLEX
#         # cv2.putText(origFrame,str(trackbarValue),(50,50), font, 3,255)
#         #read thresholds from trackbars
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end scroll'
#         pass
#
#
#     def changeMinTh(minTh):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end changeMinTh'
#         pass
#
#     def changeMaxTh(maxTh):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end changeMaxTh'
#         pass
#
#     def changeMinArea(minArea):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end changeMinArea'
#         pass
#
#     def changeMaxArea(maxArea):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end changeMaxArea'
#         pass
#
#     def resizeImageUp(res):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end resizeImageUp'
#         pass
#
#     def resizeImageDown(res):
#         minTh = cv2.getTrackbarPos('minTh', 'Bars')
#         maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
#         thresholder(minTh, maxTh)
#         # print 'end resizeImageDown'
#         pass
#     # print 'Ready create trackbars in Bars window'
#     print 'numFrame in gui utils', numFrame
#     cv2.createTrackbar('start', 'Bars', 0, numFrame-1, scroll )
#     cv2.createTrackbar('minTh', 'Bars', 0, int(maxTB), changeMinTh)
#     cv2.createTrackbar('maxTh', 'Bars', 0, int(maxTB), changeMaxTh)
#     cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
#     cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
#     cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
#     cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)
#     # print 'Ready to set the default values'
#     defFrame = 1
#     defMinTh = th2TB(minThreshold,maxRangeTh,maxTB)
#     defMaxTh = th2TB(maxThreshold,maxRangeTh,maxTB)
#     defMinA = minArea
#     defMaxA = maxArea
#     defRes = size
#     # print 'Ready to call the scroll function'
#     scroll(defFrame)
#     cv2.setTrackbarPos('start', 'Bars', defFrame)
#     changeMaxArea(defMaxA)
#     cv2.setTrackbarPos('maxArea', 'Bars', defMaxA)
#     changeMinArea(defMinA)
#     cv2.setTrackbarPos('minArea', 'Bars', defMinA)
#     changeMinTh(defMinTh)
#     cv2.setTrackbarPos('minTh', 'Bars', defMinTh)
#     changeMaxTh(defMaxTh)
#     cv2.setTrackbarPos('maxTh', 'Bars', defMaxTh)
#     resizeImageUp(defRes)
#     cv2.setTrackbarPos('ResUp', 'Bars', defRes)
#     resizeImageDown(defRes)
#     cv2.setTrackbarPos('ResDown', 'Bars', defRes)
#     # print 'Ready to get trackbarPosition'
#     start = cv2.getTrackbarPos('start','Bars')
#     minThresholdStart = cv2.getTrackbarPos('minTh', 'Bars')
#     minAreaStart = cv2.getTrackbarPos('minArea', 'Bars')
#     maxAreaStart = cv2.getTrackbarPos('maxArea', 'Bars')
#     # print 'Waiting for keypress'
#     cv2.waitKey(0)
#     # print 'Creating dictionary of preprocParams'
#     preprocParams = {'minThreshold': TB2th(cv2.getTrackbarPos('minTh', 'Bars'),maxRangeTh,maxTB),
#         'maxThreshold': TB2th(cv2.getTrackbarPos('maxTh', 'Bars'),maxRangeTh,maxTB),
#         'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
#         'maxArea': cv2.getTrackbarPos('maxArea', 'Bars'),
#         'numAnimals': numAnimals}
#     # print 'Saving dictionary of preprocParams'
#     saveFile(path, preprocParams, 'preprocparams', time = 0)
#     print 'The video will be preprocessed according to the following parameters: ', preprocParams
#
#     cap.release()
#     cv2.destroyAllWindows()

def SegmentationPreview(path, width, height, bkg, mask, useBkg, minArea = 150, maxArea = 1000, minThreshold = 136, maxThreshold = 255, size = 1):
    numAnimals = getInput('Number of animals','Type the number of animals')
    numAnimals = int(numAnimals)
    # print 'Ready to get the cap from the path'
    global cap
    cap = cv2.VideoCapture(path)
    # print 'Ready to get eh number of frames'
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def thresholder(minTh, maxTh):
        # print 'I am in thresholder'
        #threshold the frame, find contours and get portraits of the fish
        toile = np.zeros_like(avFrame, dtype='uint8')
        # print 'I am going to call segmentVideo'
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, bkg, mask, useBkg)
        # currentFrame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        # print 'The frame saved is, ', currentFrame
        # pd.to_pickle(bkg, '/home/lab/Desktop/TF_models/IdTracker/Conflict8Small/bkgGUI.pkl')
        # pd.to_pickle(frame, '/home/lab/Desktop/TF_models/IdTracker/Conflict8Small/frameGUI.pkl')
        # pd.to_pickle(origFrame, '/home/lab/Desktop/TF_models/IdTracker/Conflict8Small/origFrame1GUI.pkl')
        # pd.to_pickle(avFrameCopy, '/home/lab/Desktop/TF_models/IdTracker/Conflict8Small/avFrame1GUI.pkl')
        # pd.to_pickle(segmentedFrame, '/home/lab/Desktop/TF_models/IdTracker/Conflict8Small/Segmentedframe1GUI.pkl')
        # print '1'
        contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # print '2'
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        # print 'minArea, ', minArea
        # print 'maxArea, ', maxArea
        # print 'avIntensity, ', avIntensity
        # print '3'
        # goodContours = filterContoursBySize(contours,minArea, maxArea)
        # cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        # shower = cv2.addWeighted(origFrame,1,toile,.5,0)
        # showerCopy = shower.copy()
        # resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        # resDown = cv2.getTrackbarPos('ResDown', 'Bars')
        # # print '4'
        # showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        # showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))
        bbs, miniFrames, _, _, _, goodContours, bkgSamples = blobExtractor(segmentedFrame, origFrame, minArea, maxArea, height, width)

        # bbs, miniFrames, _, _, _, bkgSamples = getBlobsInfoPerFrame(origFrame, goodContours, height, width)

        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(origFrame,1,toile,.5,0)
        showerCopy = shower.copy()
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')
        # print '5'
        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackPortraits = numColumns - numGoodContours % numColumns
        numPortraits = numGoodContours + numBlackPortraits
        # print '6'
        j = 0
        sizePortrait = 32
        portraitsMat = []
        rowPortrait = []
        while j < numPortraits:
            if j < numGoodContours:
                portrait = getPortrait(miniFrames[j],goodContours[j],bbs[j],bkgSamples[j])
                portrait = np.squeeze(portrait)
            else:
                portrait = np.zeros((sizePortrait,sizePortrait),dtype='uint8')
            rowPortrait.append(portrait)
            if (j+1) % numColumns == 0:
                portraitsMat.append(np.hstack(rowPortrait))
                rowPortrait = []
            j += 1
        # print '7'

        portraitsMat = np.vstack(portraitsMat)
        #show window containing the trackbars
        # print '8'
        cv2.imshow('Bars',np.squeeze(portraitsMat))
        #show frame, and react to changes in Bars
        # print '9'
        cv2.imshow('IdPlayer', showerCopy)
        cv2.moveWindow('Bars', 10,10 )
        cv2.moveWindow('IdPlayer', 200, 10 )
        # print '10'

    def scroll(trackbarValue):
        global frame, avFrame, frameGray, origFrame, avFrameCopy
        # print 'setting frame position inside scroll function'
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        #Get frame from video file
        ret, frame = cap.read()

        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #make a copy of frameGray
        origFrame = frameGray.copy()
        #average frame
        avFrame = np.divide(frameGray,np.mean(frameGray))
        avFrameCopy = avFrame.copy()

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(origFrame,str(trackbarValue),(50,50), font, 3,255)
        #read thresholds from trackbars
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end scroll'
        pass


    def changeMinTh(minTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end changeMinTh'
        pass

    def changeMaxTh(maxTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end changeMaxTh'
        pass

    def changeMinArea(x):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end changeMinArea'
        pass

    def changeMaxArea(maxArea):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end changeMaxArea'
        pass

    def resizeImageUp(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end resizeImageUp'
        pass

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        # print 'end resizeImageDown'
        pass
    # print 'Ready create trackbars in Bars window'
    cv2.createTrackbar('start', 'Bars', 0, numFrame-1, scroll )
    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
    cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
    cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
    cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)
    # print 'Ready to set the default values'
    defFrame = 1
    defMinTh = minThreshold
    defMaxTh = maxThreshold
    defMinA = minArea
    defMaxA = maxArea
    defRes = size
    # print 'Ready to call the scroll function'
    scroll(defFrame)
    cv2.setTrackbarPos('start', 'Bars', defFrame)
    changeMaxArea(defMaxA)
    cv2.setTrackbarPos('maxArea', 'Bars', defMaxA)
    changeMinArea(defMinA)
    cv2.setTrackbarPos('minArea', 'Bars', defMinA)
    changeMinTh(defMinTh)
    cv2.setTrackbarPos('minTh', 'Bars', defMinTh)
    changeMaxTh(defMaxTh)
    cv2.setTrackbarPos('maxTh', 'Bars', defMaxTh)
    resizeImageUp(defRes)
    cv2.setTrackbarPos('ResUp', 'Bars', defRes)
    resizeImageDown(defRes)
    cv2.setTrackbarPos('ResDown', 'Bars', defRes)
    # print 'Ready to get trackbarPosition'
    start = cv2.getTrackbarPos('start','Bars')
    minThresholdStart = cv2.getTrackbarPos('minTh', 'Bars')
    minAreaStart = cv2.getTrackbarPos('minArea', 'Bars')
    maxAreaStart = cv2.getTrackbarPos('maxArea', 'Bars')
    # print 'Waiting for keypress'
    cv2.waitKey(0)
    # print 'Creating dictionary of preprocParams'
    preprocParams = {'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
        'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
        'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
        'maxArea': cv2.getTrackbarPos('maxArea', 'Bars'),
        'numAnimals': numAnimals}
    # print 'Saving dictionary of preprocParams'
    saveFile(path, preprocParams, 'preprocparams', time = 0)
    print 'The video will be preprocessed according to the following parameters: ', preprocParams

    cap.release()
    cv2.destroyAllWindows()

''' ****************************************************************************
Fragmentation inspector
*****************************************************************************'''
def playFragmentation(paths,visualize = False):
    """
    IdInspector
    """
    info = loadFile(paths[0], 'videoInfo', time=0)
    info = info.to_dict()[0]
    width = info['width']
    height = info['height']
    numAnimals = info['numAnimals']
    maxNumBlobs = info['maxNumBlobs']
    numSegment = 0
    # paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
    # paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi') #'../Conflict8/conflict3and4_20120316T155032_1.pkl'
    path = paths[numSegment]

    def IdPlayerFragmentation(path,numAnimals, width, height,visualize):
        df,sNumber = loadFile(path, 'segmentation', time=0)
        # video = os.path.basename(path)
        # filename, extension = os.path.splitext(video)
        # sNumber = int(filename.split('_')[-1])
        # folder = os.path.dirname(path)
        # df = pd.read_pickle(folder +'/'+ filename + '.pkl')
        print 'Visualizing video %s' % path
        # print df
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        def onChange(trackbarValue):
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
            centroids = df.loc[trackbarValue,'centroids']
            pixelsA = df.loc[trackbarValue-1,'pixels']
            pixelsB = df.loc[trackbarValue,'pixels']
            permutation = df.loc[trackbarValue,'permutation']
            print '------------------------------------------------------------'
            print 'previous frame, ', str(trackbarValue-1), ', permutation, ', df.loc[trackbarValue-1,'permutation']
            print 'current frame, ', str(trackbarValue), ', permutation, ', permutation
            trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
            print 'overlapMat, '
            print overlapMat
            print 'permutation, ', s
            # if sNumber == 1 and trackbarValue > 100:
            #     trueFragment, s = computeFrameIntersection(df.loc[trackbarValue-1,'pixels'],df.loc[trackbarValue,'pixels'],5)
            #     print trueFragment, s
            #     result = df.loc[trackbarValue-1,'permutation'][s]
            #     print 'result, ', result
            #Get frame from video file
            ret, frame = cap.read()
            #Color to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Plot segmentated blobs
            for i, pixel in enumerate(pixelsB):
                px = np.unravel_index(pixel,(height,width))
                frame[px[0],px[1]] = 255

            # plot numbers if not crossing
            # if not isinstance(permutation,float):
                # print 'pass'
            for i, centroid in enumerate(centroids):
                cv2.putText(frame,'i'+ str(permutation[i]) + '|h' +str(i),centroid, font, .7,0)

            cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

            # Visualization of the process
            cv2.imshow('IdPlayerFragmentation',frame)
            pass

        cv2.namedWindow('IdPlayerFragmentation')
        cv2.createTrackbar( 'start', 'IdPlayerFragmentation', 0, numFrame-1, onChange )
        # cv2.createTrackbar( 'end'  , 'IdPlayer', numFrame-1, numFrame, onChange )

        onChange(1)
        if visualize: ### FIXME this is because otherwise we have a Fatal Error on the "Bars" window. Apparently the backend needs a waitkey(1)...
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
            return 'q'

        start = cv2.getTrackbarPos('start','IdPlayerFragmentation')
        return raw_input('Which segment do you want to inspect?')

    finish = False
    while not finish:
        # print 'I am here', numSegment
        numSegment = IdPlayerFragmentation(paths[int(numSegment)],numAnimals, width, height,visualize)
        if numSegment == 'q':
            finish = True
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
