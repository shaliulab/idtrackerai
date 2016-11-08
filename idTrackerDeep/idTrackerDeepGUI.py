import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from segmentation_ROIPreview import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *

import time
import h5py
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
from pprint import pprint

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    ''' ************************************************************************
    Selecting library directory
    ************************************************************************ '''
    print '***** Selecting the path to the videos...'
    pathToVideos = selectDir()
    print 'The path selected is, ', pathToVideos
    # pathToVideos = '/home/lab/Desktop/TF_models/IdTracker/data/library/25dpf'

    ''' Path to video/s '''
    videoPath = natural_sort([v for v in os.listdir(pathToVideos) if isfile(pathToVideos +'/'+ v) if '.avi' in v])[0]
    videoPath = pathToVideos + '/' + videoPath
    videoPaths = scanFolder(videoPath)
    print 'The list of videos is ', videoPaths

    ''' ************************************************************************
    GUI to select the preprocessing parameters
    *************************************************************************'''
    print '\n'
    print '***** Selecting properties for bkg and ROI...'
    prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
    useBkg = prepOpts['bkg']
    useROI =  prepOpts['ROI']
    useBkg = int(useBkg)
    useROI = int(useROI)
    print 'useBkg, ', useBkg
    print 'useROI, ', useROI

    #Check for preexistent files generated during a previous session. If they
    #exist and one wants to keep them they will be loaded
    processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragmentation','portraits']
    print '\n'
    print '***** Looking for finished steps in previous session...'
    processesDict, srcSubFolder = copyExistentFiles(videoPath, processesList, time=1)
    print 'List of processes finished, ', processesDict
    print '\n'
    print '***** Selecting files to load from previous session...'
    loadPreviousDict = selectOptions(processesList, processesDict, text='Already processed steps in this video \n (check to load from ' + srcSubFolder + ')')
    print 'List of files that will be used, ', loadPreviousDict
    usePreviousBkg = loadPreviousDict['bkg']
    usePreviousROI = loadPreviousDict['ROI']
    print 'usePreviousBkg, ', usePreviousBkg
    print 'usePreviousROI, ', usePreviousROI

    ''' ROI selection and bkg loading'''
    width, height, bkg, maxIntensity, maxBkg, mask, centers = playPreview(videoPaths, useBkg, usePreviousBkg, useROI, usePreviousROI)

    ''' Segmentation inspection '''
    if not loadPreviousDict['preprocparams']:
        SegmentationPreview(videoPath, width, height, bkg, maxIntensity, maxBkg, mask, useBkg)

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        numSegment = getInput('Segment number','Type the segment to be visualized')

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
                preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
                preprocParams = preprocParams.to_dict()[0]
                numAnimals = preprocParams['numAnimals']
                minThreshold = preprocParams['minThreshold']
                maxThreshold = preprocParams['maxThreshold']
                minArea = int(preprocParams['minArea'])
                maxArea = int(preprocParams['maxArea'])
                mask = loadFile(videoPaths[0], 'ROI',0)
                mask = np.asarray(mask)
                centers= loadFile(videoPaths[0], 'centers',0)
                centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
                EQ = 0
                ### FIXME put usePreviousBkg to 1 no to recompute it everytime we change the segment
                bkg, maxIntensity, maxBkg = checkBkg(useBkg, usePreviousBkg, videoPaths, EQ, width, height)
                SegmentationPreview(path, width, height, bkg, maxIntensity, maxBkg, mask, useBkg, minArea, maxArea, minThreshold, maxThreshold)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    else:
        preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
        preprocParams = preprocParams.to_dict()[0]
        numAnimals = preprocParams['numAnimals']
        minThreshold = preprocParams['minThreshold']
        maxThreshold = preprocParams['maxThreshold']
        minArea = int(preprocParams['minArea'])
        maxArea = int(preprocParams['maxArea'])
    img = cv2.imread('../utils/loadingIdDeep.png')
    cv2.imshow('Bars',img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    ''' ************************************************************************
    Segmentation
    ************************************************************************ '''
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if not loadPreviousDict['segmentation']:
        preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
        preprocParams = preprocParams.to_dict()[0]
        numAnimals = preprocParams['numAnimals']
        minThreshold = preprocParams['minThreshold']
        maxThreshold = preprocParams['maxThreshold']
        minArea = preprocParams['minArea']
        maxArea = preprocParams['maxArea']
        EQ = 0
        print preprocParams
        segment(videoPaths, numAnimals,
                    mask, centers, useBkg, bkg, EQ,
                    minThreshold, maxThreshold,
                    minArea, maxArea)

    ''' ************************************************************************
    Fragmentation
    *************************************************************************'''
    if not loadPreviousDict['fragmentation']:
        fragment(videoPaths)
        playFragmentation(videoPaths,True) # last parameter is to visualize or not
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    ''' ************************************************************************
    Portraying
    ************************************************************************ '''
    if not loadPreviousDict['portraits']:
        portrait(videoPaths)
    portraits = loadFile(videoPaths[0], 'portraits', time=0)

    ''' ************************************************************************
    Tracker
    ************************************************************************ '''
