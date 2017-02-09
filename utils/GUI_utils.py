import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *
from ROIselect import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
     ROIsCoords, ROIsShapes = get_ROI(frame)
     mask = np.ones_like(frame,dtype='uint8')*255
     centers = []
     print ROIsCoords
     for coord,shape in zip(ROIsCoords, ROIsShapes):
          coord = np.asarray(coord).astype('int')
          if shape == 'r':
               cv2.rectangle(mask,(coord[0],coord[2]),(coord[1],coord[3]),0,-1)
               centers.append(None)
          if shape == 'c':
               center = ((coord[1]+coord[0])/2,(coord[3]+coord[2])/2)
               angle = 90
               axes = tuple(sorted(((coord[1]-coord[0])/2,(coord[3]-coord[2])/2)))
               print center, angle, axes
               cv2.ellipse(mask,center,axes,angle,0,360,0,-1)
               centers.append(center)
     return mask, centers


def checkROI(useROI, usePreviousROI, frame, videoPath):
    ''' Select ROI '''
    if useROI:
        if usePreviousROI:
            mask = loadFile(videoPath, 'ROI')
            mask = np.asarray(mask)
            centers= loadFile(videoPath, 'centers')
            centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
        else:
            print '\n Selecting ROI ...'
            mask, centers = ROIselector(frame)
    else:
        print '\n No ROI selected ...'
        mask = np.zeros_like(frame)
        centers = []
    return mask, centers

def ROISelectorPreview(paths, useROI, usePreviousROI, numSegment=0):
    """
    loads a preview of the video for manual fine-tuning
    """
    cap2 = cv2.VideoCapture(paths[0])
    flag, frame = cap2.read()
    cap2.release()
    height = frame.shape[0]
    width = frame.shape[1]
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, centers = checkROI(useROI, usePreviousROI, frameGray, paths[0])
    saveFile(paths[0], mask, 'ROI')
    saveFile(paths[0], centers, 'centers')

    return width, height, mask, centers

''' ****************************************************************************
First preview numAnimals, inspect parameters for segmentation and portraying
**************************************************************************** '''

def SegmentationPreview(path, width, height, bkg, mask, useBkg, numAnimals = None, minArea = 150, maxArea = 60000, minThreshold = 136, maxThreshold = 255, size = 1):
    if numAnimals == None:
        numAnimals = getInput('Number of animals','Type the number of animals')
        numAnimals = int(numAnimals)

    global cap
    cap = cv2.VideoCapture(path)
    numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def thresholder(minTh, maxTh):
        toile = np.zeros_like(avFrame, dtype='uint8')
        segmentedFrame = segmentVideo(origFrame, minTh, maxTh, bkg, mask, useBkg)
        contours, hierarchy = cv2.findContours(segmentedFrame,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        maxArea = cv2.getTrackbarPos('maxArea', 'Bars')
        minArea = cv2.getTrackbarPos('minArea', 'Bars')
        bbs, miniFrames, _, _, _, goodContours, bkgSamples = blobExtractor(segmentedFrame, origFrame, minArea, maxArea, height, width)

        cv2.drawContours(toile, goodContours, -1, color=255, thickness = -1)
        shower = cv2.addWeighted(origFrame,1,toile,.5,0)
        showerCopy = shower.copy()
        resUp = cv2.getTrackbarPos('ResUp', 'Bars')
        resDown = cv2.getTrackbarPos('ResDown', 'Bars')

        showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
        showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))

        numColumns = 5
        numGoodContours = len(goodContours)
        numBlackPortraits = numColumns - numGoodContours % numColumns
        numPortraits = numGoodContours + numBlackPortraits

        j = 0
        sizePortrait = 32
        portraitsMat = []
        rowPortrait = []
        while j < numPortraits:
            if j < numGoodContours:
                portrait,_ = getPortrait(miniFrames[j],goodContours[j],bbs[j],bkgSamples[j])
                portrait = np.squeeze(portrait)
            else:
                portrait = np.zeros((sizePortrait,sizePortrait),dtype='uint8')
            rowPortrait.append(portrait)
            if (j+1) % numColumns == 0:
                portraitsMat.append(np.hstack(rowPortrait))
                rowPortrait = []
            j += 1

        portraitsMat = np.vstack(portraitsMat)

        cv2.imshow('Bars',np.squeeze(portraitsMat))

        cv2.imshow('IdPlayer', showerCopy)
        cv2.moveWindow('Bars', 10,10 )
        cv2.moveWindow('IdPlayer', 200, 10 )

    def scroll(trackbarValue):
        global frame, avFrame, frameGray, origFrame, avFrameCopy
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
        ret, frame = cap.read()
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origFrame = frameGray.copy()
        avFrame = np.divide(frameGray,np.mean(frameGray))
        avFrameCopy = avFrame.copy()
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

    defFrame = 1
    defMinTh = minThreshold
    defMaxTh = maxThreshold
    defMinA = minArea
    defMaxA = maxArea
    defRes = size

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

    preprocParams = {'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
        'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
        'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
        'maxArea': cv2.getTrackbarPos('maxArea', 'Bars'),
        'numAnimals': numAnimals}

    # saveFile(path, preprocParams, 'preprocparams',hdfpkl='pkl')

    cap.release()
    cv2.destroyAllWindows()

    return preprocParams

def selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg):
    if not usePreviousPrecParams:
        videoPath = videoPaths[0]
        preprocParams = SegmentationPreview(videoPath, width, height, bkg, mask, useBkg)

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        # numSegment = getInput('Segment number','Type the segment to be visualized')

        end = False
        while not end:
            numSegment = getInput('Segment number','Type the segment to be visualized')
            if numSegment == 'q' or numSegment == 'quit' or numSegment == 'exit':
                end = True
            else:
                cv2.namedWindow('Bars')
                end = False
                usePreviousBkg = 1
                path = videoPaths[int(numSegment)]
                numAnimals = preprocParams['numAnimals']
                minThreshold = preprocParams['minThreshold']
                maxThreshold = preprocParams['maxThreshold']
                minArea = int(preprocParams['minArea'])
                maxArea = int(preprocParams['maxArea'])

                mask = loadFile(videoPaths[0], 'ROI')
                mask = np.asarray(mask)
                centers= loadFile(videoPaths[0], 'centers')
                centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
                EQ = 0
                ### FIXME put usePreviousBkg to 1 no to recompute it everytime we change the segment
                bkg = checkBkg(videoPaths, useBkg, usePreviousBkg, EQ, width, height)

                preprocParams = SegmentationPreview(path, width, height, bkg, mask, useBkg, numAnimals, minArea, maxArea, minThreshold, maxThreshold)

            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        saveFile(path, preprocParams, 'preprocparams',hdfpkl='pkl')
    else:
        preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
    return preprocParams

''' ****************************************************************************
Fragmentation inspector
*****************************************************************************'''
def playFragmentation(paths,dfGlobal,visualize = False):
    from fragmentation import computeFrameIntersection ### FIXME For some reason it does not import well in the top and I have to import it here
    """
    IdInspector
    """
    print dfGlobal.loc[80:90]
    info = loadFile(paths[0], 'videoInfo', hdfpkl = 'pkl')
    width = info['width']
    height = info['height']
    numAnimals = info['numAnimals']
    maxNumBlobs = info['maxNumBlobs']
    numSegment = 0
    frameIndices = loadFile(paths[0], 'frameIndices')
    path = paths[numSegment]

    def IdPlayerFragmentation(path,numAnimals, width, height,visualize):
        df,sNumber = loadFile(path, 'segmentation')
        print 'Visualizing video %s' % path
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        def onChange(trackbarValue):
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
            index = frameIndices[(frameIndices.segment == int(sNumber)) & (frameIndices.frame == trackbarValue)].index[0]
            print index
            permutation = dfGlobal.loc[index,'permutations']
            centroids = df.loc[trackbarValue,'centroids']
            pixelsA = df.loc[trackbarValue-1,'pixels']
            pixelsB = df.loc[trackbarValue,'pixels']
            print '------------------------------------------------------------'
            print 'previous frame, ', str(trackbarValue-1), ', permutation, ', dfGlobal.loc[index-1,'permutations']
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

            cv2.putText(frame,str(index),(50,50), font, 3,(255,0,0))

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
        numSegment = getInput('Segment number','Type the segment to be visualized')
        return numSegment
        # return raw_input('Which segment do you want to inspect?')

    finish = False
    while not finish:
        # print 'I am here', numSegment
        numSegment = IdPlayerFragmentation(paths[int(numSegment)],numAnimals, width, height,visualize)
        if numSegment == 'q':
            finish = True
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
