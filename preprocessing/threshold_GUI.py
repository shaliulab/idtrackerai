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


def IdPlayerPreview(path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold):
    #load video
    # global numSegment
    cap = cv2.VideoCapture(path)
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    # cv2.namedWindow('Bars')


    def thresholder(minTh, maxTh):
        #threshold the frame, find contours and get portraits of the fish
        toile = np.zeros_like(avFrame, dtype='uint8')
        segmentedFrame = segmentVideo(avFrame, minTh, maxTh, bkg, mask, bkgSubstraction)
        contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        goodContours = filterContoursBySize(contours,minArea, maxArea)
        # print minArea, maxArea, minTh, maxTh
        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(origFrame,1,toile,.5,0)
        showerCopy = shower.copy()
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')
        # showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        # showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))

        bbs, miniFrames, _, _, _, bkgSamples = getBlobsInfoPerFrame(origFrame, goodContours, height, width)
        ####Uncomment to plot miniframes#######
        # for i,mini in enumerate(miniFrames):
        #     cv2.imshow(str(i), mini)
        #######################################
        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackPortraits = numColumns - numGoodContours % numColumns
        numPortraits = numGoodContours + numBlackPortraits
        # print 'numPortraits, ', numPortraits
        # print 'numGoodContours, ', numGoodContours
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
            if j % (numColumns-1) == 0  and j != 0:
                portraitsMat.append(np.hstack(rowPortrait))
                rowPortrait = []
            j += 1

        portraitsMat = np.vstack(portraitsMat)
        #show window containing the trackbars
        cv2.imshow('Bars',np.squeeze(portraitsMat))
        #show frame, and react to changes in Bars
        cv2.imshow('IdPlayer', showerCopy)

    # cv2.namedWindow('Bars')
    # cv2.namedWindow('IdPlayer')

    def scroll(trackbarValue):
        global avFrame, frameGray, origFrame
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        #Get frame from video file
        ret, frame = cap.read()
        # print 'the frame is', frame
        # print '--------------------------->8'
        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #make a copy of frameGray
        origFrame = frameGray.copy()
        #average frame
        avFrame, avIntensity = frameAverager(origFrame)
        avFrameCopy = avFrame.copy()
        #print frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(origFrame,str(trackbarValue),(50,50), font, 3,255)
        #read thresholds from trackbars
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass


    def changeMinTh(minTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMaxTh(maxTh):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMinArea(x):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def changeMaxArea(maxArea):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def resizeImageUp(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass

    def resizeImageDown(res):
        minTh = cv2.getTrackbarPos('minTh', 'Bars')
        maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
        thresholder(minTh, maxTh)
        pass



    cv2.createTrackbar('start', 'Bars', 0, numFrame-1, scroll )
    cv2.createTrackbar('minTh', 'Bars', 0, 255, changeMinTh)
    cv2.createTrackbar('maxTh', 'Bars', 0, 255, changeMaxTh)
    cv2.createTrackbar('minArea', 'Bars', 0, 1000, changeMinArea)
    cv2.createTrackbar('maxArea', 'Bars', 0, 60000, changeMaxArea)
    cv2.createTrackbar('ResUp', 'Bars', 1, 20, resizeImageUp)
    cv2.createTrackbar('ResDown', 'Bars', 1, 20, resizeImageDown)

    defFrame = 10
    defMinTh = 150
    defMaxTh = 255
    defMinA = 150
    defMaxA = 1000
    defRes = 1

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

    start = cv2.getTrackbarPos('start','Bars')
    minThresholdStart = cv2.getTrackbarPos('minTh', 'Bars')
    minAreaStart = cv2.getTrackbarPos('minArea', 'Bars')
    maxAreaStart = cv2.getTrackbarPos('maxArea', 'Bars')

    cv2.waitKey(0)

    # print '****************************************************************'
    # numSegment = raw_input('Which segment do you want to inspect?')
    # numSegment = getInput('Segment number','Type the segment to be visualized')
    #FIXME I am saving here the final parameters because of a a Bad Window error, that seems really difficult to tackle.
    preprocParams = {'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
        'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
        'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
        'maxArea': cv2.getTrackbarPos('maxArea', 'Bars')}

    saveFile(path, preprocParams, 'preprocparams', time = 0)
    print 'The video will be preprocessed according to the following parameters: ', preprocParams

    cap.release()
    cv2.destroyAllWindows()
    # cv2.destroyWindow('Bars')
    # cv2.destroyWindow('IdPlayer')


# def playPreview(paths, bkgSubstraction, selectROI, numSegment=0):
#     """
#     loads a preview of the video for manual fine-tuning
#     """
#     # print 'Starting playPreview'
#     # global numSegment
#     width, height = getVideoInfo(paths)
#     video = os.path.basename(paths[0])
#     folder = os.path.dirname(paths[0])
#     filename, extension = os.path.splitext(video)
#     subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
#     subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
#     # if len(subFolders) > 0:
#     #     subFolder = subFolders[0]
#     # else:
#     if numSegment==0:
#         createFolder(paths[0], name = '', timestamp = True)
#     subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
#     subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
#     subFolder = subFolders[0]
#     #Load frame to choose ROIs
#     cap2 = cv2.VideoCapture(paths[0])
#     _, frame = cap2.read()
#     cap2.release()
#     # cv2.destroyAllWindows()
#     #frame to grayscale
#     frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     if numSegment == 0:
#         print 'segment 0 is here'
#         mask, centers = checkROI(selectROI, frameGray, paths[0])
#         saveFile(paths[0], mask, 'mask',time = 0)
#         saveFile(paths[0], centers, 'centers',time = 0)
#     else:
#         mask = loadFile(paths[0], 'mask', time=0)
#
#     #set the name for the background if it does not exists
#     filename = subFolder + filename.split('_')[0] + '_bkg.pkl'
#     #if it exists, load it!
#     bkg = checkBkg(bkgSubstraction, paths, 0, width, height)
#     #set parameters for the GUI
#     maxArea = 1000
#     minArea = 150
#     minThreshold = 150
#
#     #save mask and centers
#     #FIXME: probably everytime we save we should check if the file is already there and ask either to overwrite or generate a unique name...
#     print 'numSegment, ', numSegment
#     path = paths[int(numSegment)]
#
#     return path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold

if __name__ == '__main__':
    videoPath = '../data/library/25dpf/group_1_camera_1/group_1_camera_1_20160519T103108_1.avi'
    # videoPath = '../data/library/33dpf/group_1/group_1.avi'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = videoPath, type = str)
    # parser.add_argument('--num_animals', default = 8, type = int)
    parser.add_argument('--bkg_subtraction', default = 1, type = int)
    parser.add_argument('--ROI_selection', default = 1, type = int)

    args = parser.parse_args()

    paths = scanFolder(args.path)
    bkgSubstraction = args.bkg_subtraction
    selectROI = args.ROI_selection
    cv2.namedWindow('Bars')

    numSegment = 0
    path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold = playPreview(paths, bkgSubstraction, selectROI)
    IdPlayerPreview(path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold)


    print '------------------'
    print 'finished first run'
    print '------------------'
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    numSegment = getInput('Segment number','Type the segment to be visualized')


    print '------------------'
    print 'starting second run'
    print '------------------'
    end = False
    while not end:
        numSegment = getInput('Segment number','Type the segment to be visualized')
        if numSegment == 'q' or numSegment == 'quit' or numSegment == 'exit':
            end = True
        else:
            end = False
            path = paths[int(numSegment)]
            preprocParams= loadFile(paths[0], 'preprocparams',0)
            minThreshold = preprocParams['minThreshold']
            maxThreshold = preprocParams['maxThreshold']
            minArea = preprocParams['minArea']
            maxArea = preprocParams['maxArea']
            mask= loadFile(paths[0], 'mask',0)
            centers= loadFile(paths[0], 'centers',0)
            bkg = loadFile(paths[0], 'bkg', 0)
            cv2.namedWindow('Bars')
            IdPlayerPreview(path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold)
