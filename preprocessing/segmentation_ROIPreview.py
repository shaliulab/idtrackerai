import cv2
import sys
sys.path.append('../utils')

from py_utils import *
from video_utils import *
from threshold_GUI import *

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

def segmentAndSave(path, height, width, mask, bkgSubstraction, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea):

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
        #Color to gray scale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        origFrame = frameGray.copy()
        avFrame, avIntensity = frameAverager(origFrame)
        avFrameCopy = avFrame.copy()
        segmentedFrame = segmentVideo(avFrame, minThreshold, maxThreshold, bkg, mask, bkgSubstraction)
        # Find contours in the segmented image
        boundingBoxes, miniFrames, centroids, areas, pixels, goodContoursFull, bkgSamples = blobExtractor(segmentedFrame, origFrame, minArea, maxArea, height, width)

        if len(centroids) > maxNumBlobs:
            maxNumBlobs = len(centroids)
        ### UNCOMMENT TO PLOT ##################################################
        # cv2.drawContours(origFrame,goodContoursFull,-1,color=(255,0,0),thickness=-1)
        # cv2.imshow('checkcoord', origFrame)
        # k = cv2.waitKey(30) & 0xFF
        # if k == 27: #pres esc to quit
        #     break
        ########################################################################

        # Add frame imformation to DataFrame
        df.loc[counter] = [avIntensity, boundingBoxes, miniFrames, goodContoursFull, centroids, areas, pixels, len(centroids), bkgSamples]
        counter += 1

    # cap.release()
    cv2.destroyAllWindows()
    saveFile(path, df, 'segment', time = 0)

    return np.multiply(numSegment,np.ones(numFrames)).astype('int').tolist(), np.arange(numFrames).tolist(), maxNumBlobs


def segment(paths, numAnimals,
            mask, centers, bkgSubstraction, bkg, EQ,
            minThreshold, maxThreshold,
            minArea, maxArea):

    ''' Entering loop for segmentation of the video '''
    num_cores = multiprocessing.cpu_count()

    width, height = getVideoInfo(paths)
    # num_cores = 1
    print 'Entering to the parallel loop'
    OupPutParallel = Parallel(n_jobs=num_cores)(delayed(segmentAndSave)(path, height, width, mask, bkgSubstraction, bkg, EQ, minThreshold, maxThreshold, minArea, maxArea) for path in paths)
    allSegments = [(out[0],out[1]) for out in OupPutParallel]
    # print allSegments
    maxNumBlobs = max([out[2] for out in OupPutParallel])
    # print maxNumBlobs
    allSegments = sorted(allSegments, key=lambda x: x[0][0])
    generateVideoTOC(allSegments, paths[0])
    collectAndSaveVideoInfo(paths[0], height, width, mask, centers, numAnimals, num_cores, minThreshold,maxThreshold,maxArea,maxNumBlobs)

# if __name__ == '__main__':
#
#     # videoPath = '../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi'
#     # videoPath = '../data/library/25dpf/group_1_camera_1/group_1_camera_1_20160519T103108_1.avi'
#     # videoPath = '../Conflict8/conflict3and4_20120316T155032_1.avi'
#     videoPath = '../data/library/33dpf/group_1/group_1.avi'
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--path', default = videoPath, type = str)
#     parser.add_argument('--folder_name', default = '', type = str)
#     parser.add_argument('--bkg_subtraction', default = 1, type = int)
#     parser.add_argument('--ROI_selection', default = 1, type = int)
#     parser.add_argument('--Eq_image', default = 0, type = int)
#     parser.add_argument('--min_th', default =60, type = int)
#     parser.add_argument('--max_th', default = 255, type = int)
#     parser.add_argument('--min_area', default = 250, type = int)
#     parser.add_argument('--max_area', default = 50000, type = int)
#     parser.add_argument('--num_animals', default = 8, type = int)
#     args = parser.parse_args()
#
#     ''' Parameters for the segmentation '''
#     numAnimals = args.num_animals
#     bkgSubstraction = args.bkg_subtraction
#     selectROI = args.ROI_selection
#     EQ = args.Eq_image
#     minThreshold = args.min_th
#     maxThreshold = args.max_th
#     minArea = args.min_area # in pixels
#     maxArea = args.max_area # in pixels
#
#     ''' Path to video/s '''
#     paths = scanFolder(args.path)
#     name  = args.folder_name
#
#     # prevAndSegm(paths, bkgSubstraction, selectROI,name,numAnimals,EQ)
#
#     playPreview(paths, bkgSubstraction, selectROI)
#     #load parameters after preview
#     preprocParams= loadFile(paths[0], 'preprocparams',0)
#     minThreshold = preprocParams['minThreshold']
#     maxThreshold = preprocParams['maxThreshold']
#     minArea = preprocParams['minArea']
#     maxArea = preprocParams['maxArea']
#     mask= loadFile(paths[0], 'mask',0)
#     centers= loadFile(paths[0], 'centers',0)
#     bkg = loadFile(paths[0], 'bkg', 0)
#
#     segment(paths, name, numAnimals,
#             mask, centers, bkgSubstraction, bkg, EQ,
#             minThreshold, maxThreshold,
#             minArea, maxArea)
