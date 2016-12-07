import cv2
import sys
sys.path.append('../utils')

from py_utils import *
from video_utils import *
# from threshold_GUI import *

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
import cPickle as pickle
import gc
import datetime
from skimage import data
from skimage.viewer.canvastools import RectangleTool, PaintTool
from skimage.viewer import ImageViewer
from scipy import ndimage
import Tkinter, tkSimpleDialog

def segmentAndSave(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea):

    print 'Segmenting video %s' % path
    cap = cv2.VideoCapture(path)
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    numSegment = int(filename.split('_')[-1])
    numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    counter = 0
    df = pd.DataFrame(columns=('avIntensity', 'boundingBoxes','miniFrames', 'contours', 'centroids', 'areas', 'pixels', 'numberOfBlobs', 'bkgSamples'))
    maxNumBlobs = 0
    while counter < numFrames:
        #Get frame from video file
        ret, frame = cap.read()
        # print ret
        # if ret == False:
            # print "*********************************** ret false"
        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origFrame = frameGray.copy()
        # print avIntensity
        avIntensity = np.mean(origFrame)
        segmentedFrame = segmentVideo(origFrame, minThreshold, maxThreshold, bkg, mask, useBkg)
        segmentedFrameCopy = segmentedFrame.copy()
        # Find contours in the segmented image
        boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples = blobExtractor(segmentedFrame, origFrame, minArea, maxArea, height, width)
        # print 'minArea, ', minArea
        # print 'maxArea, ', maxArea
        # print 'minTh, ', minThreshold
        # print 'maxTh, ', maxThreshold
        # print counter, len(centroids)
        # if len(centroids) == 0:
            # print "*********************************** 0 blobs detected"
        if len(centroids) > maxNumBlobs:
            maxNumBlobs = len(centroids)
        ### UNCOMMENT TO PLOT ##################################################
        # cv2.drawContours(origFrame,goodContoursFull,-1,color=(255,0,0),thickness=-1)
        # cv2.imshow('checkcoord', origFrame)
        # k = cv2.waitKey(100) & 0xFF
        # if k == 27: #pres esc to quit
        #     break
        ########################################################################

        # Add frame imformation to DataFrame
        df.loc[counter] = [avIntensity, boundingBoxes, miniFrames, goodContoursFull, centroids, areas, pixels, len(centroids), bkgSamples]
        counter += 1

    cap.release()
    cv2.destroyAllWindows()
    saveFile(path, df, 'segment', time = 0)
    # lst = [df,cap]
    # del df, cap
    # del lst
    gc.collect()

    return np.multiply(numSegment,np.ones(numFrames)).astype('int').tolist(), np.arange(numFrames).tolist(), maxNumBlobs

def segment(paths,preprocParams, mask, centers, useBkg, bkg, EQ):

    numAnimals = preprocParams['numAnimals']
    minThreshold = preprocParams['minThreshold']
    maxThreshold = preprocParams['maxThreshold']
    minArea = preprocParams['minArea']
    maxArea = preprocParams['maxArea']

    width, height = getVideoInfo(paths)
    ''' splitting paths list into sublists '''
    pathsSubLists = [paths[i:i+4] for i in range(0,len(paths),4)]
    ''' Entering loop for segmentation of the video '''
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    print 'Entering to the parallel loop...\n'
    allSegments = []
    numBlobs = []
    for pathsSubList in pathsSubLists:
        OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea) for path in pathsSubList)
        allSegmentsSubList = [(out[0],out[1]) for out in OupPutParallel]
        allSegments.append(allSegmentsSubList)
        numBlobs.append([out[2] for out in OupPutParallel])
    allSegments = flatten(allSegments)
    maxNumBlobs = max(flatten(numBlobs))
    # OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, mask, useBkg, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea) for path in paths)
    # allSegments = [(out[0],out[1]) for out in OupPutParallel]
    # # print allSegments
    # maxNumBlobs = max([out[2] for out in OupPutParallel])
    # # print maxNumBlobs
    allSegments = sorted(allSegments, key=lambda x: x[0][0])
    numFrames = generateVideoTOC(allSegments, paths[0])
    collectAndSaveVideoInfo(paths[0], numFrames, height, width, numAnimals, num_cores, minThreshold,maxThreshold,maxArea,maxNumBlobs)
