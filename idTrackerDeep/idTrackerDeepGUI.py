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
from plotters import *

import time
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
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
    Selecting video directory
    ************************************************************************ '''
    print
    print '********************************************************************'
    print 'Selecting the path to the videos...'
    print '********************************************************************\n'

    initialDir = ''
    pathToVideo = selectFile()
    pathToVideos = os.path.dirname(pathToVideo)
    print 'The path selected is, ', pathToVideos
    # pathToVideos = '/home/lab/Desktop/TF_models/IdTracker/data/library/25dpf'

    ''' Path to video/s '''
    # videoPath = natural_sort([v for v in os.listdir(pathToVideos) if isfile(pathToVideos +'/'+ v) if '.avi' in v])[0]
    extensions = ['.avi', '.mp4']
    videoPath = natural_sort([v for v in os.listdir(pathToVideos) if isfile(pathToVideos +'/'+ v) if any( ext in v for ext in extensions)])[0]
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
    processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragments','portraits']
    reUseAll = getInput('Reuse all preprocessing, ', 'Do you wanna reuse all previos preprocessing? ([y]/n)')
    if reUseAll == 'n':

        existentFiles, srcSubFolder = getExistentFiles(videoPath, processesList)
        print 'are you the path? ', srcSubFolder
        print 'List of processes finished, ', existentFiles
        print '\nSelecting files to load from previous session...'
        loadPreviousDict = selectOptions(processesList, existentFiles, text='Already processed steps in this video \n (check to load from ' + srcSubFolder + ')')

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        loadPreviousDict = {'ROI': 1, 'bkg': 1, 'preprocparams': 1, 'segmentation': 1, 'fragments': 1, 'portraits': 1}

    else:
        raise ValueError('The input introduced do not match the possible options')

    print 'List of files that will be used, ', loadPreviousDict
    usePreviousBkg = loadPreviousDict['bkg']
    usePreviousROI = loadPreviousDict['ROI']
    print 'usePreviousBkg set to ', usePreviousBkg
    print 'usePreviousROI set to ', usePreviousROI

    ''' ROI selection and bkg loading'''
    width, height, bkg, mask, centers = playPreview(videoPaths, useBkg, usePreviousBkg, useROI, usePreviousROI, numSegment=0)

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
                preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
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
                bkg = checkBkg(useBkg, usePreviousBkg, videoPaths, EQ, width, height)
                SegmentationPreview(path, width, height, bkg, mask, useBkg, minArea, maxArea, minThreshold, maxThreshold)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    else:
        preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
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
        preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
        EQ = 0
        print 'The preprocessing parameters dictionary loaded is ', preprocParams
        segment(videoPaths, preprocParams, mask, centers, useBkg, bkg, EQ)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    ''' ************************************************************************
    Fragmentation
    *************************************************************************'''
    print '********************************************************************'
    print 'Fragmentation'
    print '********************************************************************\n'
    if not loadPreviousDict['fragments']:
        dfGlobal, fragmentsDict = fragment(videoPaths,videoInfo=None)

        playFragmentation(videoPaths,dfGlobal,True) # last parameter is to visualize or not

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        dfGlobal = loadFile(videoPaths[0],'portraits')
        fragmentsDict = loadFile(videoPaths[0],'fragments',hdfpkl='pkl')

    ''' ************************************************************************
    Portraying
    ************************************************************************ '''
    print '********************************************************************'
    print 'Portraying'
    print '********************************************************************\n'
    if not loadPreviousDict['portraits']:
        portraits = portrait(videoPaths,dfGlobal)
    else:
        portraits = loadFile(videoPaths[0], 'portraits')

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    '''
    ************************************************************************
    Tracker
    ************************************************************************
    '''
    print '********************************************************************'
    print 'Tracker'
    print '********************************************************************\n'
    preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
    numAnimals = preprocParams['numAnimals']

    restoreFromAccPoint = getInput('Restore from a previous accumulation step','Do you want to restore from an accumulation point? y/[n]')

    if restoreFromAccPoint == 'n' or restoreFromAccPoint == '':
        loadCkpt_folder = selectDir(initialDir) #select where to load the model
        loadCkpt_folder = os.path.relpath(loadCkpt_folder)
        # inputs = getMultipleInputs('Training parameters', ['batch size', 'num. epochs', 'learning rate', 'train (1 (from strach) or 2 (from last check point))'])
        # print 'inputs, ', inputs
        print 'Entering into the fineTuner...'
        batchSize = 50 #int(inputs[1])
        numEpochs = 100 #int(inputs[2])
        lr = 0.01 #np.float32(inputs[3])
        train = 1 #int(inputs[4])

        ''' Initialization of variables for the accumulation loop'''
        def createSessionFolder(videoPath):
            def getLastSession(subFolders):
                if len(subFolders) == 0:
                    lastIndex = 0
                else:
                    subFolders = natural_sort(subFolders)[::-1]
                    lastIndex = int(subFolders[0].split('_')[-1])
                return lastIndex

            video = os.path.basename(videoPath)
            folder = os.path.dirname(videoPath)
            filename, extension = os.path.splitext(video)
            subFolder = folder + '/CNN_models'
            subSubFolders = glob.glob(subFolder +"/*")
            lastIndex = getLastSession(subSubFolders)
            sessionPath = subFolder + '/Session_' + str(lastIndex + 1)
            os.makedirs(sessionPath)
            print 'You just created ', sessionPath
            figurePath = sessionPath + '/figures'
            os.makedirs(figurePath)
            print 'You just created ', figurePath

            return sessionPath, figurePath

        sessionPath, figurePath = createSessionFolder(videoPath)
        pickle.dump( preprocParams , open( sessionPath + "/preprocparams.pkl", "wb" ))

        accumDict = {
                'counter': 0,
                'thVels': 0.5,
                'minDist': 0,
                'fragsForTrain': [], # to be saved
                'newFragForTrain': [],
                'badFragments': [], # to be saved
                'overallP2': [1./numAnimals],
                'continueFlag': True}

        trainDict = {
                'loadCkpt_folder':loadCkpt_folder,
                'ckpt_dir': '',
                'fig_dir': figurePath,
                'sess_dir': sessionPath,
                'batchSize': batchSize,
                'numEpochs': numEpochs,
                'lr': lr,
                'keep_prob': 1.,
                'train':train,
                'lossAccDict':{},
                'refDict':{},
                'framesColumnsRefDict': {}, #to be saved
                'usedIndivIntervals': [],
                'idUsedIntervals': []}

        handlesDict = {'restoring': False}

        normFreqFragments = None
    elif restoreFromAccPoint == 'y':
        restoreFromAccPointPath = selectDir('./')

        if 'AccumulationStep_' not in restoreFromAccPointPath:
            raise ValueError('Select an AccumulationStep folder to restore from it.')
        else:
            countpkl = 0
            for file in os.listdir(restoreFromAccPointPath):
                if file.endswith(".pkl"):
                    countpkl += 1
            if countpkl != 3:
                raise ValueError('It is not possible to restore from here. Select an accumulation point in which statistics.pkl, accumDict.pkl, and trainDict.pkl have been saved.')
            else:

                statistics = pickle.load( open( restoreFromAccPointPath + "/statistics.pkl", "rb" ) )
                accumDict = pickle.load( open( restoreFromAccPointPath + "/accumDict.pkl", "rb" ) )
                trainDict = pickle.load( open( restoreFromAccPointPath + "/trainDict.pkl", "rb" ) )
                normFreqFragments = statistics['normFreqFragsAll']
                portraits = accumDict['portraits']

        handlesDict = {'restoring': True}
    else:
        raise ValueError('You typed ' + restoreFromAccPoint + ' the accepted values are y or n.')

    while accumDict['continueFlag']:
        print '\n*** Accumulation ', accumDict['counter'], ' ***'

        ''' Best fragment search '''
        accumDict = bestFragmentFinder(accumDict, normFreqFragments, fragmentsDict, numAnimals, portraits)
        # fragmentAccumPlotter(fragmentsDict,portraits,accumDict,figurePath)

        pprint(accumDict)
        print '---------------\n'

        ''' Fine tuning '''
        trainDict, handlesDict = fineTuner(videoPath, accumDict, trainDict, fragmentsDict, handlesDict, portraits)

        print 'loadCkpt_folder ', trainDict['loadCkpt_folder']
        print 'ckpt_dir ', trainDict['ckpt_dir']
        print '---------------\n'

        ''' Identity assignation '''
        normFreqFragments, portraits, overallP2 = idAssigner(videoPath, trainDict, accumDict['counter'], fragmentsDict, portraits)

        # P2AccumPlotter(fragmentsDict,portraits,accumDict,figurePath,trainDict['ckpt_dir'])

        ''' Updating training Dictionary'''
        trainDict['train'] = 2
        trainDict['numEpochs'] = 10000
        accumDict['counter'] += 1
        accumDict['portraits'] = portraits
        accumDict['overallP2'].append(overallP2)
        # Variables to be saved in order to restore the accumulation
        print 'saving dictionaries to enable restore from accumulation'
        pickle.dump( accumDict , open( trainDict['ckpt_dir'] + "/accumDict.pkl", "wb" ) )
        pickle.dump( trainDict , open( trainDict['ckpt_dir'] + "/trainDict.pkl", "wb" ) )
        print 'dictionaries saved in ', trainDict['ckpt_dir']
