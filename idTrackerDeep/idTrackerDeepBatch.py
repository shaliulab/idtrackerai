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
    Selecting directory containing benchmark videos
    ************************************************************************ '''
    benchmarkFolder = selectDir('./')
    videosFolders = getSubfolders(benchmarkFolder)

    ''' ************************************************************************
    Loop on the video directories
    ************************************************************************ '''
    def checkVideoRequirements(videoPath, requiredFile):
        # to be executed in batch we expect a video to contain a folder preprocessing
        # and a certain number of subfolder
        videoPathSubfolders = getSubfolders(videoPath)
        joinSubfolders = ''.join(videoPathSubfolders)
        if 'preprocessing' not in joinSubfolders:
            discardedVideos.append(videoPath)
            print 'no preprocessing'
            return False

        files, subfolders = getFilesAndSubfolders(videoPath + '/preprocessing')
        if 'segmentation' not in subfolders:
            print 'no segmentation'
            return False

        for f in requiredFile:
            if f not in files:
                print 'file missing', f
                return False

        return True

    # list of file that has to be already computed and stored in a video folder
    # to enable batch model
    requiredFile = ['centers.hdf5', 'dfGlobal.hdf5', 'fragments.pkl', 'frameIndices.hdf5', 'portraits.hdf5', 'preprocparams.pkl', 'ROI.hdf5', 'videoInfo.pkl']
    admissibleVideos = []
    discardedVideos = []

    for videoPath in videosFolders:
        if checkVideoRequirements(videoPath, requiredFile):
            admissibleVideos.append(videoPath)
        else:
            discardedVideos.append(videoPath)
            continue

        loadPreviousDict = {'ROI': 1, 'bkg': 1, 'preprocparams': 1, 'segmentation': 1, 'fragments': 1, 'portraits': 1}

        usePreviousBkg = loadPreviousDict['bkg']
        usePreviousROI = loadPreviousDict['ROI']
        print 'usePreviousBkg set to ', usePreviousBkg
        print 'usePreviousROI set to ', usePreviousROI


        '''
        ************************************************************************
        Loading from preprocessing
        ************************************************************************
        '''
        # trick to reuse the load function
        videoPath = videoPath + '/foo.avi'

        dfGlobal = loadFile(videoPath,'portraits')
        fragmentsDict = loadFile(videoPath,'fragments',hdfpkl='pkl')
        portraits = loadFile(videoPath, 'portraits')

        '''
        ************************************************************************
        Tracker
        ************************************************************************
        '''

        preprocParams= loadFile(videoPath, 'preprocparams',hdfpkl = 'pkl')
        numAnimals = preprocParams['numAnimals']
        #set path to the model
        loadCkpt_folder = '../CNN/ckpt_dir_new3_xavierSGD_maxImages_300epoch_lr01'
        #set network params
        batchSize = 50
        numEpochs = 100
        lr = 0.01
        train = 1

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
                'usedIndivIntervals': []}

        normFreqFragments = None

        while accumDict['continueFlag']:
            print '\n*** Accumulation ', accumDict['counter'], ' ***'

            ''' Best fragment search '''
            accumDict = bestFragmentFinder(accumDict, normFreqFragments, fragmentsDict, numAnimals, portraits)
            # fragmentAccumPlotter(fragmentsDict,portraits,accumDict,figurePath)

            pprint(accumDict)
            print '---------------\n'

            ''' Fine tuning '''
            trainDict = fineTuner(videoPath, accumDict, trainDict, fragmentsDict, portraits)

            print 'loadCkpt_folder ', trainDict['loadCkpt_folder']
            print 'ckpt_dir ', trainDict['ckpt_dir']
            print '---------------\n'

            ''' Identity assignation '''
            normFreqFragments, portraits, overallP2 = idAssigner(videoPath, trainDict, accumDict['counter'], fragmentsDict, portraits)

            # P2AccumPlotter(fragmentsDict,portraits,accumDict,figurePath,trainDict['ckpt_dir'])

            ''' Updating training Dictionary'''
            trainDict['train'] = 2
            trainDict['numEpochs'] = 2000
            accumDict['counter'] += 1
            accumDict['portraits'] = portraits
            accumDict['overallP2'].append(overallP2)
            # Variables to be saved in order to restore the accumulation
            print 'saving dictionaries to enable restore from accumulation'
            pickle.dump( accumDict , open( trainDict['ckpt_dir'] + "/accumDict.pkl", "wb" ) )
            pickle.dump( trainDict , open( trainDict['ckpt_dir'] + "/trainDict.pkl", "wb" ) )
            print 'dictionaries saved in ', trainDict['ckpt_dir']
