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

def orderVideo(matrixToOrder,permutations,maxNumBlobs):
    matrixOrdered = np.zeros_like(matrixToOrder)

    for frame in range(len(permutations)):
        for i in range(maxNumBlobs):
            index = list(np.where(permutations[frame]==i)[0])
            # print index
            if len(index) == 1:
                matrixOrdered[frame,i] = matrixToOrder[frame,index]
            else:
                matrixOrdered[frame,i] = -1

    return matrixOrdered

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
    if not loadPreviousDict['fragmentation']:
        dfGlobal, fragmentsDict = fragment(videoPaths,videoInfo=None)

        playFragmentation(videoPaths,dfGlobal,True) # last parameter is to visualize or not

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
    loadCkpt_folder = selectDir(initialDir) #select where to load the model
    # loadCkpt_folder = '/home/lab/Desktop/TF_models/IdTracker/CNN/ckpt_Train_25dpf_60indiv_25000_transfer'
    loadCkpt_folder = os.path.relpath(loadCkpt_folder)
    # inputs = getMultipleInputs('Training parameters', ['ckptName','batch size', 'num. epochs', 'learning rate', 'train (1 (from strach) or 2 (from last check point))'])
    # print 'inputs, ', inputs
    print 'Entering into the fineTuner...'
    ckptName = 'test'
    batchSize = 50 #int(inputs[1])
    numEpochs = 100 #int(inputs[2])
    lr = 0.01 #np.float32(inputs[3])
    train = 1 #int(inputs[4])
    trainDict = {
        'loadCkpt_folder':loadCkpt_folder,
        'ckptName': ckptName,
        'batchSize': batchSize,
        'numEpochs': numEpochs,
        'lr': lr,
        'train':train}

    preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
    preprocParams = preprocParams.to_dict()[0]
    numAnimals = preprocParams['numAnimals']

    indexFragment = 0
    avVels = [0,0]
    thVels = 0.9
    badFragments = []
    framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][indexFragment]
    while any(np.asarray(avVels)<=thVels):
        avVels = []
        print 'Checking whether the fragmentNumber ', indexFragment, ' is good for training'
        for framesAndColumnsInterval in framesAndColumnsGlobalFrag:
            avVels.append(getAvVelFragment(portraits,framesAndColumnsInterval))
        print 'The average velocities for each blob are (pixels/frame), ', avVels
        if any(np.asarray(avVels)<=thVels):
            badFragments.append(indexFragment)
            indexFragment += 1
            framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][indexFragment]
            print 'There is some animal that does not move enough. Going to next longest fragment'
            print 'Bad fragments, ', badFragments


    print 'The fine-tuning will start with the ', indexFragment, ' longest fragment'
    fragsForTrain = [indexFragment]
    continueFlag = True
    counter = 0
    minLen = 150
    while continueFlag:
        print '\n************** Training ', counter
        print 'training dictionary, ', trainDict

        ''' Fine tuning '''
        fineTuner(videoPath,trainDict,fragsForTrain,fragmentsDict,portraits)

        ''' plot and save fragment selected '''
        fragments = fragmentsDict['fragments']
        permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
        maxNumBlobs = len(permutations[0])
        permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
        permOrdered = permOrdered.T.astype('float32')

        plt.close()
        fig, ax = plt.subplots(figsize=(25, 5))
        permOrdered[permOrdered >= 0] = .5
        im = plt.imshow(permOrdered,cmap=plt.cm.gray,interpolation='none',vmin=0.,vmax=1.)
        im.cmap.set_under('r')
        # im.set_clim(0, 1.)
        # cb = plt.colorbar(im)

        # for i in range(len(fragments)):
        for i in fragsForTrain:
            ax.add_patch(
                patches.Rectangle(
                    (fragments[i,0], -0.5),   # (x,y)
                    fragments[i,1]-fragments[i,0],  # width
                    maxNumBlobs,          # height
                    fill=True,
                    edgecolor=None,
                    facecolor='b',
                    alpha = 0.5
                )
            )

        plt.axis('tight')
        plt.xlabel('Frame number')
        plt.ylabel('Blob index')
        plt.gca().set_yticks(range(0,maxNumBlobs,4))
        plt.gca().set_yticklabels(range(1,maxNumBlobs+1,4))
        plt.gca().invert_yaxis()
        plt.tight_layout()

        print 'Saving figure...'
        ckpt_dir = getCkptvideoPath(videoPath,ckptName,train=2,time =0)
        figname = ckpt_dir + '/figures/fragments_' + str(counter) + '.pdf'
        fig.savefig(figname)

        ''' Identity assignation '''
        normFreqFragments, portraits = idAssigner(videoPath,trainDict,fragmentsDict,portraits)
        ''' Computing best next fragments '''
        fragsForTrain,continueFlag,minLen,badFragments = bestFragmentFinder(fragsForTrain,normFreqFragments,fragmentsDict,numAnimals,minLen,badFragments,portraits,thVels)

        ''' Plotting and saving probability matrix'''
        statistics = loadFile(videoPath, 'statistics', time=0)
        statistics = statistics.to_dict()[0]
        P2 = statistics['P2FragAllVideo']
        P2Ordered =  orderVideo(P2,permutations,maxNumBlobs)
        P2good = np.max(P2Ordered,axis=2).T

        plt.close()
        fig, ax = plt.subplots(figsize=(25, 5))
        im2 = plt.imshow(P2good,cmap=plt.cm.gray,interpolation='none')
        im2.cmap.set_under('r')
        im2.set_clim(0, 1)
        cb = plt.colorbar(im2)
        # fig.colorbar(im, ax=ax)
        plt.axis('tight')
        plt.xlabel('Frame number')
        plt.ylabel('Blob index')
        plt.gca().set_yticks(range(0,maxNumBlobs,4))
        plt.gca().set_yticklabels(range(1,maxNumBlobs+1,4))
        plt.gca().invert_yaxis()
        plt.tight_layout()

        print 'Saving figure...'
        figname = ckpt_dir + '/figures/P2_' + str(counter) + '.pdf'
        fig.savefig(figname)

        ''' Updating training Dictionary'''
        trainDict = {
            'loadCkpt_folder':loadCkpt_folder,
            'ckptName': ckptName,
            'batchSize': batchSize,
            'numEpochs': 2000,
            'lr': lr,
            'train':2}
        counter += 1
