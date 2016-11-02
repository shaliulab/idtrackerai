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

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    ''' ************************************************************************
    Selecting video to track
    ************************************************************************ '''
    videoPath = selectFile()

    ''' ************************************************************************
    Starting GUI to select the preprocessing parameters
    *************************************************************************'''
    prepOpts = selectOptions(['BKG', 'ROI'])
    bkgSubstraction = prepOpts['BKG']
    selectROI = prepOpts['ROI']
    paths = scanFolder(videoPath)
    numSegment = 0
    path, width, height, bkg, mask, bkgSubstraction, minArea, maxArea, minThreshold, numAnimals = playPreview(paths, bkgSubstraction, selectROI)

    ''' Segmentation inspection '''
    ### TODO we need a checkbox for the background subtraction. It might be useful to check the segmentation with or without bkg subtraction
    SegmentationPreview(path, width, height, bkg, mask, bkgSubstraction, numAnimals)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    numSegment = getInput('Segment number','Type the segment to be visualized')

    end = False
    while not end:
        numSegment = getInput('Segment number','Type the segment to be visualized')
        if numSegment == 'q' or numSegment == 'quit' or numSegment == 'exit':
            end = True
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        else:
            end = False
            path = paths[int(numSegment)]
            preprocParams= loadFile(paths[0], 'preprocparams',0)
            numAnimals = preprocParams['numAnimals']
            minThreshold = preprocParams['minThreshold']
            maxThreshold = preprocParams['maxThreshold']
            minArea = preprocParams['minArea']
            maxArea = preprocParams['maxArea']
            mask= loadFile(paths[0], 'mask',0)
            centers= loadFile(paths[0], 'centers',0)
            EQ = 0
            bkg = checkBkg(bkgSubstraction, paths, EQ, width, height)
            cv2.namedWindow('Bars')
            SegmentationPreview(path, width, height, bkg, mask, bkgSubstraction, numAnimals, minArea, maxArea, minThreshold, maxThreshold)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    ''' ************************************************************************
    Segmentation
    ************************************************************************ '''
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    preprocParams= loadFile(paths[0], 'preprocparams',0)
    numAnimals = preprocParams['numAnimals']
    minThreshold = preprocParams['minThreshold']
    maxThreshold = preprocParams['maxThreshold']
    minArea = preprocParams['minArea']
    maxArea = preprocParams['maxArea']
    mask= loadFile(paths[0], 'mask',0)
    centers= loadFile(paths[0], 'centers',0)
    EQ = 0
    bkg = checkBkg(bkgSubstraction, paths, EQ, width, height)
    print preprocParams

    segment(paths, numAnimals,
                mask, centers, bkgSubstraction, bkg, EQ,
                minThreshold, maxThreshold,
                minArea, maxArea)

    ''' ************************************************************************
    Fragmentation
    *************************************************************************'''
    fragment(paths)
    playFragmentation(paths)

    ''' ************************************************************************
    Portraying
    ************************************************************************ '''
    portrait(paths)

    ''' ************************************************************************
    Tracker
    ************************************************************************ '''
