import sys
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/CNN')

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

def DataFineTuning(accumDict, trainDict, fragmentsDict, portraits, statistics, numAnimals, printFlag = True):
    ### Fix maximal number of images:
    maximalRefPerAnimal = 3000
    # get fragments data
    fragments = np.asarray(fragmentsDict['fragments'])

    framesAndBlobColumns = fragmentsDict['framesAndBlobColumnsDist']
    minLenIndivCompleteFragments = fragmentsDict['minLenIndivCompleteFragments']
    intervals = fragmentsDict['intervalsDist']

    # identities
    identities = statistics['fragmentIds'].tolist()

    # get accumulation data
    newFragForTrain = accumDict['newFragForTrain']
    print '------------NEWFRAGS------------))))))))))))_______________'
    print newFragForTrain
    print len(newFragForTrain)
    print '------------------------))))))))))))_______________'


    # get training data
    refDict = trainDict['refDict']
    framesColumnsRefDict = trainDict['framesColumnsRefDict']
    usedIndivIntervals = trainDict['usedIndivIntervals']
    idUsedIntervals = trainDict['idUsedIntervals']
    refDictTemp = {}

    ''' First I save all the images of each identified individual in a dictionary '''
    if printFlag:
        print '\n**** Creating dictionary of references ****'

    # create temporary refDict with same keys as refDict and empty values
    #
    # # take length of dict before updating
    # numRefBeforeUpdate =
    # # how long?
    # numSamplesPerAnimal = maximalRefPerAnimal - minNumRef
    # # if longer than max --> sample
    minAccRefs = []
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
                identity = identities[frames[0]][columns[0]]

                if not identity in refDictTemp.keys(): # if the identity has not been added to the dictionary, I initialize the list
                    # refDict[identity] = []
                    framesColumnsRefDict[identity] = []
                    refDictTemp[identity] = []

                for frame,column in zip(frames,columns): # I loop in all the frames of the individual fragment to add them to the dictionary of references
                    # refDict[identity].append(portraits.loc[frame,'images'][column])
                    framesColumnsRefDict[identity].append((frame,column))
                    refDictTemp[identity].append(portraits.loc[frame,'images'][column])

                idUsedIntervals.append(identity)
                usedIndivIntervals.append(intervalsIndivFrag)

    if accumDict['counter'] == 0:
        refDictTemp = {i: refDictTemp[key] for i, key in enumerate(refDictTemp.keys())} # this is done to order the ids in the refDict

        if printFlag:
            print '\n The keys of the refDict are ', refDictTemp.keys()



    ''' I compute the minimum number of references I can take  given the new fragments added during the accumulation'''
    minNumRefTemp = np.min([len(refDictTemp[iD]) for iD in refDictTemp.keys()]) # minimal number of references from new fragments
    if len(refDict) != 0:
        minNumRef = np.min([len(refDict[iD]) for iD in refDict.keys()]) # minimal number of references from old dictionary of references
    else:
        minNumRef = 0

    print 'number of new references gained while accumulating: ', minNumRefTemp
    print 'number of old references: ', minNumRef
    overallRefs = minNumRef + minNumRefTemp

    if printFlag:
        print '\nMinimum number of references per identities: ', minNumRef

    ''' I build the images and labels to feed the network '''
    if printFlag:
        print '\nBuilding arrays of images and labels'

    images = []
    labels = []
    if overallRefs <= maximalRefPerAnimal:
        print '*************************************************'
        print 'we are under the threshold:', overallRefs, ' <= ', maximalRefPerAnimal
        print '*************************************************'
        for iD in refDictTemp.keys():
            print 'refDictTemp ', iD, len(refDictTemp[iD])
            if accumDict['counter'] == 0:
                print '*************************************************'
                print 'it is the first accumulation'
                print '*************************************************'
                imagesList = np.asarray(refDictTemp[iD])# this should be equivalent to what we were doing before
                indexes = np.linspace(0,len(imagesList)-1,minNumRefTemp).astype('int')
                images.append(imagesList[indexes])
                labels.append(np.ones(minNumRefTemp)*iD)
                refDict[iD] = refDictTemp[iD]
                print len(refDict[iD])
            else:
                print '*************************************************'
                print 'it is not the first accum'
                print '*************************************************'
                print iD
                print 'refDict ', iD, len(refDict[iD])
                print 'refDictTemp ', iD, len(refDictTemp[iD])

                # refDict[iD].append(refDictTemp[iD])
                refDict[iD] = np.vstack((refDict[iD],refDictTemp[iD]))
                print 'refDict ', iD, len(refDict[iD])

                print len(refDict[iD])

                imagesList = np.asarray(refDict[iD])

                indexes = np.linspace(0,len(imagesList)-1,overallRefs).astype('int')
                images.append(imagesList[indexes])
                labels.append(np.ones(overallRefs)*iD)

    elif overallRefs > maximalRefPerAnimal:
        #sample from old dict:
        # compute the number of samples to be taken from the old dict
        ratioOld = .5
        ratioNew = .5
        numSamplesOld = maximalRefPerAnimal * ratioOld
        numSamplesNew = maximalRefPerAnimal * ratioNew
        print 'old references to be retained: ', numSamplesOld
        print 'old references to be added: ', numSamplesNew

        for iD in refDictTemp.keys():
            sampledImagesListOld = np.asarray(refDict[iD])
            samplesIndexesOld = np.linspace(0,len(sampledImagesListOld)-1,numSamplesOld).astype('int')
            sampledImagesListOld = sampledImagesListOld[samplesIndexesOld]

            sampledImagesListNew = np.asarray(refDictTemp[iD])
            samplesIndexesNew = np.linspace(0,len(sampledImagesListNew)-1,numSamplesNew).astype('int')
            sampledImagesListNew = sampledImagesListNew[samplesIndexesNew]

            imagesList = np.vstack((sampledImagesListOld, sampledImagesListNew))
            images.append(imagesList)
            labels.append(np.ones(maximalRefPerAnimal)*iD)
            print 'images and labels should have the same length:'
            print 'length labels ', len(labels)
            print 'length images ', len(images)
            print 'max num of references: ', maximalRefPerAnimal
            print '-------------------------------------------------'
            #update the refDict so that it always contains everything
            refDict[iD] = np.vstack((refDict[iD],refDictTemp[iD]))

    images = np.vstack(images)
    images = np.expand_dims(images,axis=1)
    print 'shape of the images ', images.shape
    print 'length of labels', len(labels)
    imagesDims = images.shape
    imsize = (imagesDims[1],imagesDims[2], imagesDims[3])
    labels = flatten(labels)
    print 'size labels after flatten', len(labels)
    print 'label sample', labels[0:10]
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

    ''' Update dictionary of references '''
    trainDict['refDict'] = refDict
    trainDict['framesColumnsRefDict'] = framesColumnsRefDict
    trainDict['usedIndivIntervals'] = usedIndivIntervals
    trainDict['idUsedIntervals'] = idUsedIntervals

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

def fineTuner(videoPath, accumDict, trainDict, fragmentsDict, handlesDict, portraits, statistics, videoInfo = [], plotFlag = True, printFlag = True):
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
    trainDict = DataFineTuning(accumDict, trainDict, fragmentsDict, portraits, statistics, numAnimals)

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
                    Vindices, Viter_per_epoch,
                    onlySoftmax=True) #NOTE:hard-coded flag for testing purpose
    trainDict['loadCkpt_folder'] = ckpt_dir
    return trainDict, handlesDict
