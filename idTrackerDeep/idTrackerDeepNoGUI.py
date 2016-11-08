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
    ''' ************************************************************************
    Selecting library directory
    ************************************************************************ '''
    print '***** Selecting the path to the videos...'
    # pathToVideos = selectDir()
    pathToVideos = '/home/lab/Desktop/TF_models/IdTracker/Conflict8Small'
    # pathToVideos = '/home/lab/Desktop/TF_models/IdTracker/Cafeina5pecesSmall'

    print 'The path selected is, ', pathToVideos

    ''' Path to video/s '''
    videoPath = natural_sort([v for v in os.listdir(pathToVideos) if isfile(pathToVideos +'/'+ v) if '.avi' in v])[0]
    videoPath = pathToVideos + '/' + videoPath
    videoPaths = scanFolder(videoPath)
    print 'The list of videos is ', videoPaths
    createFolder(videoPaths[0], name = '', timestamp = False)

    ''' ************************************************************************
    GUI to select the preprocessing parameters
    *************************************************************************'''
    print '\n'
    print '***** Selecting properties for bkg and ROI...'
    # prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
    prepOpts = {'bkg': 1, 'ROI': 0}
    useBkg = prepOpts['bkg']
    useROI =  prepOpts['ROI']
    useBkg = int(useBkg)
    useROI = int(useROI)
    print 'useBkg, ', useBkg
    print 'useROI, ', useROI

    print '***** Selecting files to load from previous session...'
    usePreviousBkg = 0
    usePreviousROI = 0
    print 'usePreviousBkg, ', usePreviousBkg
    print 'usePreviousROI, ', usePreviousROI

    '''' ************************************************************************
    ROI, bkg and preprocparams
    ************************************************************************ '''
    cap2 = cv2.VideoCapture(videoPaths[0])
    flag, frame = cap2.read()
    cap2.release()
    height = frame.shape[0]
    width = frame.shape[1]

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask, centers = checkROI(useROI, usePreviousROI, frameGray, videoPaths[0])
    saveFile(videoPaths[0], mask, 'ROI',time = 0)
    saveFile(videoPaths[0], centers, 'centers',time = 0)
    bkg, maxIntensity, maxBkg = checkBkg(useBkg, usePreviousBkg, videoPaths, 0, width, height)

    preprocParams = dict()
    # preprocParams['maxAvIntensity'] = maxAvIntensity
    preprocParams['numAnimals'] = 8
    preprocParams['minThreshold'] = 0.85
    preprocParams['maxThreshold'] = 255
    preprocParams['minArea'] = 150
    preprocParams['maxArea'] = 60000


    ''' ************************************************************************
    Segmentation
    ************************************************************************ '''
    EQ = 0
    numAnimals = preprocParams['numAnimals']
    minThreshold = preprocParams['minThreshold']
    maxThreshold = preprocParams['maxThreshold']
    minArea = preprocParams['minArea']
    maxArea = preprocParams['maxArea']
    # maxAvIntensity = preprocParams['maxAvIntensity']
    print preprocParams
    segment(videoPaths, numAnimals,
                mask, centers, useBkg, bkg, EQ,
                minThreshold, maxThreshold,
                minArea, maxArea)

    ''' ************************************************************************
    Fragmentation
    *************************************************************************'''
    fragment(videoPaths)
    playFragmentation(videoPaths,True) # last parameter is to visualize or not
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    ''' ************************************************************************
    Portraying
    ************************************************************************ '''
    portrait(videoPaths)
    portraits = loadFile(videoPaths[0], 'portraits', time=0)

    ''' ************************************************************************
    Tracker
    ************************************************************************ '''
