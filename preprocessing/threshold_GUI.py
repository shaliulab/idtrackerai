import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')
from segmentation_ROI import *
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

def playPreview(paths, bkgSubstraction, selectROI):
    """
    loads a preview of the video for manual fine-tuning
    """
    width, height = getVideoInfo(paths)
    numSegment = 0
    video = os.path.basename(paths[0])
    folder = os.path.dirname(paths[0])
    filename, extension = os.path.splitext(video)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    # if len(subFolders) > 0:
    #     subFolder = subFolders[0]
    # else:
    createFolder(paths[0], name = '', timestamp = True)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    subFolder = subFolders[0]
    #Load frame to choose ROIs
    cap = cv2.VideoCapture(paths[0])
    _, frame = cap.read()
    cap.release()
    # cv2.destroyAllWindows()
    #frame to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, centers = checkROI(selectROI, frameGray, paths[0])
    #save mask and centers
    #FIXME: probably everytime we save we should check if the file is already there and ask either to overwrite or generate a unique name...
    saveFile(paths[0], mask, 'mask',time = 0)
    saveFile(paths[0], centers, 'centers',time = 0)
    #set the name for the background if it does not exists
    filename = subFolder + filename.split('_')[0] + '_bkg.pkl'
    #if it exists, load it!
    bkg = checkBkg(bkgSubstraction, paths, 0, width, height)
    #set parameters for the GUI
    maxArea = 1000
    minArea = 150
    minThreshold = 150
    path = paths[numSegment]

    def IdPlayerPreview(path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold):
        #load video
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

        def scroll(trackbarValue):
            global avFrame, frameGray, origFrame
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
            #Get frame from video file
            ret, frame = cap.read()
            #Color to gray scale
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #make a copy of frameGray
            origFrame = frameGray.copy()
            #average frame
            avFrame, avIntensity = frameAverager(origFrame)
            avFrameCopy = avFrame.copy()
            #print frame number
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(origFrame,str(trackbarValue),(50,50), font, 3,(255,0,0))
            #read thresholds from trackbars
            minTh = cv2.getTrackbarPos('minTh', 'Bars')
            maxTh = cv2.getTrackbarPos('maxTh', 'Bars')
            thresholder(minTh, maxTh)
            pass

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
            showerCopy = cv2.resize(showerCopy,None,fx = resUp, fy = resUp)
            showerCopy = cv2.resize(showerCopy,None, fx = np.true_divide(1,resDown), fy = np.true_divide(1,resDown))

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

        cv2.namedWindow('IdPlayer')
        cv2.namedWindow('Bars')

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

        cv2.waitKey()

        #FIXME I am saving here the final parameters because of a a Bad Window error, that seems really difficult to tackle.
        preprocParams = {'minThreshold': cv2.getTrackbarPos('minTh', 'Bars'),
            'maxThreshold': cv2.getTrackbarPos('maxTh', 'Bars'),
            'minArea': cv2.getTrackbarPos('minArea', 'Bars'),
            'maxArea':cv2.getTrackbarPos('maxArea', 'Bars'),
            'mask':mask}
        saveFile(path, preprocParams, 'preprocparams', time = 0)
        print 'The video will be preprocessed according to the following parameters: ', preprocParams
        cap.release()
        cv2.destroyAllWindows()
        # return raw_input('Which segment do you want to inspect?')
    IdPlayerPreview(path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold)
    # finish = False
    # while not finish:
    #     print 'You are visualizing segment ', numSegment
    #     numSegment = IdPlayerPreview(path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold)
    #     if numSegment == 'q':
    #         finish = True
    # return cv2.getTrackbarPos('maxArea', 'Bars'), cv2.getTrackbarPos('minArea', 'Bars'), cv2.getTrackbarPos('minTh', 'Bars'),cv2.getTrackbarPos('maxTh', 'Bars')

if __name__ == '__main__':
    # videoPath = '../data/library/25dpf/group_1_camera_1/group_1_camera_1_20160519T103108_1.avi'
    videoPath = '../data/library/33dpf/group_1/group_1.avi'

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = videoPath, type = str)
    # parser.add_argument('--num_animals', default = 8, type = int)
    parser.add_argument('--bkg_subtraction', default = 1, type = int)
    parser.add_argument('--ROI_selection', default = 1, type = int)

    args = parser.parse_args()

    paths = scanFolder(args.path)
    bkgSubstraction = args.bkg_subtraction
    selectROI = args.ROI_selection
    playPreview(paths, bkgSubstraction, selectROI)
