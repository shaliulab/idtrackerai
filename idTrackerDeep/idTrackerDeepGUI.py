import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../tracker')

from segmentation import *
from fragmentation_serie import *
from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *
from tracker import *

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
    print
    print '********************************************************************'
    print 'Selecting the path to the videos...'
    print '********************************************************************\n'

    initialDir = ''
    pathToVideos = selectDir(initialDir)
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
    print '********************************************************************'
    print 'Selecting properties for bkg and ROI...'
    print '********************************************************************\n'

    prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
    useBkg = prepOpts['bkg']
    useROI =  prepOpts['ROI']
    useBkg = int(useBkg)
    useROI = int(useROI)
    print 'useBkg set to ', useBkg
    print 'useROI set to ', useROI

    print '\nLooking for finished steps in previous session...'
    processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragmentation','portraits']
    reUseAll = getInput('Reuse all preprocessing, ', 'Do you wanna reuse all previos preprocessing? ([y]/n)')
    if reUseAll == 'n':
        processesDict, srcSubFolder = copyExistentFiles(videoPath, processesList, time=1)
        print 'List of processes finished, ', processesDict
        print '\nSelecting files to load from previous session...'
        loadPreviousDict = selectOptions(processesList, processesDict, text='Already processed steps in this video \n (check to load from ' + srcSubFolder + ')')
    elif reUseAll == '' or reUseAll.lower() == 'y' :
        loadPreviousDict = {'ROI': 1, 'bkg': 1, 'preprocparams': 1, 'segmentation': 1, 'fragmentation': 1, 'portraits': 1}
    else:
        raise ValueError('The input introduces do not match the possible options')

    print 'List of files that will be used, ', loadPreviousDict
    usePreviousBkg = loadPreviousDict['bkg']
    usePreviousROI = loadPreviousDict['ROI']
    print 'usePreviousBkg set to ', usePreviousBkg
    print 'usePreviousROI set to ', usePreviousROI

    ''' ROI selection and bkg loading'''
    width, height, bkg, mask, centers = playPreview(videoPaths, useBkg, usePreviousBkg, useROI, usePreviousROI)

    ''' Segmentation inspection '''
    if not loadPreviousDict['preprocparams']:
        SegmentationPreview(videoPath, width, height, bkg, mask, useBkg)

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
                bkg = checkBkg(useBkg, usePreviousBkg, videoPaths, EQ, width, height)
                SegmentationPreview(path, width, height, bkg, mask, useBkg, minArea, maxArea, minThreshold, maxThreshold)
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
    print '********************************************************************'
    print 'Segmentation'
    print '********************************************************************\n'
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
        print 'The preprocessing parameters dictionary loaded is ', preprocParams
        segment(videoPaths, numAnimals,
                    mask, centers, useBkg, bkg, EQ,
                    minThreshold, maxThreshold,
                    minArea, maxArea)

    ''' ************************************************************************
    Fragmentation
    *************************************************************************'''
    print '********************************************************************'
    print 'Fragmentation'
    print '********************************************************************\n'
    if not loadPreviousDict['fragmentation']:
        dfGlobal, fragmentsDict = fragment(videoPaths,videoInfo=None)

        playFragmentation(videoPaths,False) # last parameter is to visualize or not

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        dfGlobal = loadFile(videoPaths[0],'portraits',time=0)
        fragmentsDict = loadFile(videoPaths[0],'fragments',time=0,hdfpkl='pkl')

    ''' ************************************************************************
    Portraying
    ************************************************************************ '''
    print '********************************************************************'
    print 'Portraying'
    print '********************************************************************\n'
    if not loadPreviousDict['portraits']:
        portraits = portrait(videoPaths,dfGlobal)
    else:
        portraits = loadFile(videoPaths[0], 'portraits', time=0)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    ''' ************************************************************************
    Tracker
    ************************************************************************ '''
    print '********************************************************************'
    print 'Tracker'
    print '********************************************************************\n'
    loadCkpt_folder = selectDir(initialDir)
    loadCkpt_folder = os.path.relpath(loadCkpt_folder)
    # inputs = getMultipleInputs('Training parameters', ['ckptName','batch size', 'num. epochs', 'learning rate', 'train (1 (from strach) or 2 (from last check point))'])
    # print 'inputs, ', inputs
    print 'Entering into the fineTuner...'
    ckptName = 'test'
    batchSize = 50 #int(inputs[1])
    numEpochs = 100 #int(inputs[2])
    lr = 0.001 #np.float32(inputs[3])
    train = 1 #int(inputs[4])
    trainDict = {
        'loadCkpt_folder':loadCkpt_folder,
        'ckptName': ckptName,
        'batchSize': batchSize,
        'numEpochs': numEpochs,
        'lr': lr,
        'train':train}

    ''' first fraining '''
    print '************** First training'
    fineTuner(videoPath,trainDict,[0],fragmentsDict,portraits)
    print 'Engering into the idAssigner...'
    normFreqFragments, portraits = idAssigner(videoPath,trainDict,fragmentsDict,portraits)
    # print portraits
    # print normFreqFragments
    fragsForTrain = bestFragmentFinder([0],normFreqFragments,fragmentsDict,numAnimals)
    # print fragsForTrain

    ''' second training '''
    print '************** Second training'
    trainDict = {
        'loadCkpt_folder':loadCkpt_folder,
        'ckptName': ckptName,
        'batchSize': batchSize,
        'numEpochs': 200,
        'lr': lr,
        'train':2}
    fineTuner(videoPath,trainDict,fragsForTrain,fragmentsDict,portraits)
    print 'Engering into the idAssigner...'
    normFreqFragments, portraits = idAssigner(videoPath,trainDict,fragmentsDict,portraits)
    # print normFreqFragments
    fragsForTrain = bestFragmentFinder(fragsForTrain,normFreqFragments,fragmentsDict,numAnimals)
    print fragsForTrain

    ''' Third training '''
    print '************** Third training'
    trainDict = {
        'loadCkpt_folder':loadCkpt_folder,
        'ckptName': ckptName,
        'batchSize': batchSize,
        'numEpochs': 300,
        'lr': lr,
        'train':2}
    fineTuner(videoPath,trainDict,fragsForTrain,fragmentsDict,portraits)
    print 'Engering into the idAssigner...'
    normFreqFragments, portraits = idAssigner(videoPath,trainDict,fragmentsDict,portraits)
    # print normFreqFragments
    fragsForTrain = bestFragmentFinder(fragsForTrain,normFreqFragments,fragmentsDict,numAnimals)
    print fragsForTrain

    ''' Forth training '''
    print '************** Forth training'
    trainDict = {
        'loadCkpt_folder':loadCkpt_folder,
        'ckptName': ckptName,
        'batchSize': batchSize,
        'numEpochs': 400,
        'lr': lr,
        'train':2}
    fineTuner(videoPath,trainDict,fragsForTrain,fragmentsDict,portraits)
    print 'Engering into the idAssigner...'
    normFreqFragments, portraits = idAssigner(videoPath,trainDict,fragmentsDict,portraits)
    # print normFreqFragments
    fragsForTrain = bestFragmentFinder(fragsForTrain,normFreqFragments,fragmentsDict,numAnimals)
    print fragsForTrain
