import sys
sys.path.append('../utils')
sys.path.append('../CNN')

from py_utils import *
from video_utils import *
from idTrainerTracker import *

import time
import numpy as np
np.set_printoptions(precision=2)
import numpy.matlib
import argparse
import os
import glob
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import cPickle as pickle
import tensorflow as tf
from tf_utils import *
from input_data_cnn import *
from cnn_utils import *
from pprint import pprint
from collections import Counter
import collections
import datetime

def DataFineTuning(accumDict, trainDict, fragmentsDict, portraits,numAnimals, printFlag = True):

    # get fragments data
    fragments = np.asarray(fragmentsDict['fragments'])
    framesAndBlobColumns = fragmentsDict['framesAndBlobColumnsDist']
    minLenIndivCompleteFragments = fragmentsDict['minLenIndivCompleteFragments']
    intervals = fragmentsDict['intervalsDist']

    # get accumulation data
    newFragForTrain = accumDict['newFragForTrain']

    # get training data
    refDict = trainDict['refDict']
    framesColumnsRefDict = trainDict['framesColumnsRefDict']
    usedIndivIntervals = trainDict['usedIndivIntervals']
    idUsedIntervals = trainDict['idUsedIntervals']

    ''' First I save all the images of each identified individual in a dictionary '''
    if printFlag:
        print '\n**** Creating dictionary of references ****'

    for j, frag in enumerate(newFragForTrain): # for each complete fragment that has to be used for the training
        if printFlag:
            print '\nGetting references from global fragment ', frag

        fragment = fragments[frag] # I take the fragment
        framesColumnsIndivFrags = framesAndBlobColumns[frag] # I take the list of individual fragments in frames and columns
        intervalsIndivFrags = intervals[frag] # I take the list of individual fragments in terms of intervals

        for i, (framesColumnsIndivFrag,intervalsIndivFrag) in enumerate(zip(framesColumnsIndivFrags,intervalsIndivFrags)):
            framesColumnsIndivFrag = np.asarray(framesColumnsIndivFrag)

            if not intervalsIndivFrag in usedIndivIntervals: # I only use individual fragments that have not been used before
                frames = framesColumnsIndivFrag[:,0]
                columns = framesColumnsIndivFrag[:,1]
                identity = portraits.loc[frames[0],'identities'][columns[0]]

                if not identity in refDict.keys(): # if the identity has not been added to the dictionary, I initialize the list
                    refDict[identity] = []
                    framesColumnsRefDict[identity] = []

                for frame,column in zip(frames,columns): # I loop in all the frames of the individual fragment to add them to the dictionary of references
                    refDict[identity].append(portraits.loc[frame,'images'][column])
                    framesColumnsRefDict[identity].append((frame,column))

                idUsedIntervals.append(identity)
                usedIndivIntervals.append(intervalsIndivFrag)

    if accumDict['counter'] == 0:
        refDict = {i: refDict[key] for i, key in enumerate(refDict.keys())}
        if printFlag:
            print '\n The keys of the refDict are ', refDict.keys()

    # if len(refDict.keys()) != numAnimals:
    #     raise ValueError('The number of identities should be the same as the number of animals. This means that a global fragment does not have as many individual fragments as number of animals ')

    ''' Update dictionary of references '''
    trainDict['refDict'] = refDict
    trainDict['framesColumnsRefDict'] = framesColumnsRefDict
    trainDict['usedIndivIntervals'] = usedIndivIntervals
    trainDict['idUsedIntervals'] = idUsedIntervals

    ''' I compute the minimum number of references I can take '''
    minNumRef = np.min([len(refDict[iD]) for iD in refDict.keys()])

    if printFlag:
        print '\nMinimum number of references per identities: ', minNumRef

    ''' I build the images and labels to feed the network '''
    if printFlag:
        print '\nBuilding arrays of images and labels'

    images = []
    labels = []
    for iD in refDict.keys():
        imagesList = np.asarray(refDict[iD])
        indexes = np.linspace(0,len(imagesList)-1,minNumRef).astype('int')
        images.append(imagesList[indexes])
        labels.append(np.ones(minNumRef)*iD)
    images = np.vstack(images)
    images = np.expand_dims(images,axis=1)
    imagesDims = images.shape
    imsize = (imagesDims[1],imagesDims[2], imagesDims[3])
    labels = flatten(labels)
    labels = map(int,labels)
    labels = dense_to_one_hot(labels, numAnimals)
    numImages = len(labels)

    np.random.seed(0)
    perm = np.random.permutation(numImages)
    images = images[perm]
    labels = labels[perm]

    images_max = np.max(images)
    if images_max > 1:
        images = images/255.

    numTrain = np.ceil(np.true_divide(numImages,10)*9).astype('int')
    X_train = images[:numTrain]
    Y_train = labels[:numTrain]
    X_val = images[numTrain:]
    Y_val = labels[numTrain:]

    resolution = np.prod(imsize)
    X_train = np.reshape(X_train, [numTrain, resolution])
    X_val = np.reshape(X_val, [numImages - numTrain, resolution])

    return imsize, X_train, Y_train, X_val, Y_val, trainDict

def getCkptvideoPath(videoPath, accumCounter, train=0):
    """
    train = 0 (id assignation)
    train = 1 (first fine-tuning)
    train = 2 (further tuning from previons checkpoint with more references)
    """

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

    sessionPath = subFolder + '/Session_' + str(lastIndex)
    ckptvideoPath = sessionPath + '/AccumulationStep_' + str(accumCounter)
    if train == 0:
        print 'you will assign identities from the last checkpoint in ', ckptvideoPath
    elif train == 2:
        print 'you will keep training from the last checkpoint in ', ckptvideoPath
    elif train == 1:
        print 'model checkpoints will be saved in ', ckptvideoPath

    return ckptvideoPath

def fineTuner(videoPath, accumDict, trainDict, fragmentsDict, handlesDict, portraits, videoInfo = [], plotFlag = True, printFlag = True):
    if printFlag:
        print '\n--- Entering the fineTuner ---'

    # Load data if needed
    if videoInfo == []:
        videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')

    # get information from videoInfo and portraits
    numFrames =  len(portraits)
    numAnimals = int(videoInfo['numAnimals'])
    maxNumBlobs = videoInfo['maxNumBlobs']

    # get information from trainDict
    loadCkpt_folder = trainDict['loadCkpt_folder']
    batchSize = trainDict['batchSize']
    numEpochs =  trainDict['numEpochs']
    lr = trainDict['lr']
    train = trainDict['train']
    lossAccDict = trainDict['lossAccDict']

    # get information from acuumDict
    accumCounter = accumDict['counter']

    if printFlag:
        print '\nGetting next checkpoint folder'

    ckpt_dir = getCkptvideoPath(videoPath, accumCounter, train)
    trainDict['ckpt_dir'] = ckpt_dir

    imsize,\
    X_train, Y_train,\
    X_val, Y_val,\
    trainDict = DataFineTuning(accumDict, trainDict, fragmentsDict, portraits,numAnimals)

    if printFlag:
        print '\n fine tune train size:    images  labels'
        print X_train.shape, Y_train.shape
        print 'validation fine tune size:    images  labels'
        print X_val.shape, Y_val.shape

    channels, width, height = imsize
    resolution = np.prod(imsize)
    classes = numAnimals

    numImagesT = Y_train.shape[0]
    numImagesV = Y_val.shape[0]
    Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batchSize)
    Vindices, Viter_per_epoch = get_batch_indices(numImagesV,batchSize)

    if printFlag:
        print '\nrunning with the devil'
        print 'The models will be loaded from (loadCkpt_folder)', loadCkpt_folder
        print '\nEntering training\n'
    trainDict, handlesDict = run_training(X_train, Y_train, X_val, Y_val,
                    width, height, channels, classes, resolution,
                    trainDict, accumDict, fragmentsDict, handlesDict, portraits,
                    Tindices, Titer_per_epoch,
                    Vindices, Viter_per_epoch)
    trainDict['loadCkpt_folder'] = ckpt_dir
    return trainDict, handlesDict
