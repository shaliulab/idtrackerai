import sys
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/CNN')
sys.path.append('IdTrackerDeep/tracker')

from py_utils import *
from video_utils import *
from idTrainerTracker import *
from cnn_utils import standarizeImages
from cnn_utils import getCkptvideoPath

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

def DataFineTuning(accumDict, trainDict, fragmentsDict, portraits, statistics, numAnimals, weighted_flag, printFlag = True):
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

    for j, frag in enumerate(newFragForTrain): # for each complete fragment that has to be used for the training
        if printFlag:
            print '\nGetting references from global fragment ', frag

        framesColumnsIndivFrags = framesAndBlobColumns[frag] # I take the list of individual fragments in frames and columns
        intervalsIndivFrags = intervals[frag] # I take the list of individual fragments in terms of intervals

        ids_checked_for_ref = []
        ids_checked_for_ref1 = []
        for i, (framesColumnsIndivFrag,intervalsIndivFrag) in enumerate(zip(framesColumnsIndivFrags,intervalsIndivFrags)):
            if accumDict['counter'] != 0:
                blobIndex = intervalsIndivFrag[0]
                fragNum = intervalsIndivFrag[1]
                id1 = statistics['idFragsAll'][blobIndex][fragNum]
                ids_checked_for_ref1.append(id1)

            framesColumnsIndivFrag = np.asarray(framesColumnsIndivFrag)
            frames = framesColumnsIndivFrag[:,0]
            columns = framesColumnsIndivFrag[:,1]
            identity = identities[frames[0]][columns[0]]
            ids_checked_for_ref.append(identity)
            if not intervalsIndivFrag in usedIndivIntervals: # I only use individual fragments that have not been used before
                print 'Adding references from individual interval'
                print intervalsIndivFrag
                print 'first frame and column', frames[0], columns[0]
                if accumDict['counter'] == 0:
                    print 'The identity of this individual interval is ', identity
                else:
                    print 'The identity of this individual interval is ', identity, id1

                if not identity in refDictTemp.keys(): # if the identity has not been added to the dictionary, I initialize the list
                    framesColumnsRefDict[identity] = []
                    refDictTemp[identity] = []

                for frame,column in zip(frames,columns): # I loop in all the frames of the individual fragment to add them to the dictionary of references
                    framesColumnsRefDict[identity].append((frame,column))
                    refDictTemp[identity].append(portraits.loc[frame,'images'][column])
                # if intervalsIndivFrag[0] == 0 and intervalsIndivFrag[1] == 133:
                #     pickle.dump(np.asarray(refDictTemp[0]),open('./finetuner-imagesInterval0-133.pkl','wb'))

                if accumDict['counter'] != 0:
                    idUsedIntervals.append(identity)
                usedIndivIntervals.append(intervalsIndivFrag)

        all_identities = range(numAnimals)
        if len(set(ids_checked_for_ref)) < numAnimals:
            print 'The identities that we tried to add for references are'
            print 'identities, ', all_identities
            print 'ids, ', ids_checked_for_ref
            repeated_ids = set([x for x in ids_checked_for_ref if ids_checked_for_ref.count(x) > 1])
            print 'The identities ', list(repeated_ids), ' are repeated'
            missing_ids = set(all_identities).difference(set(ids_checked_for_ref))
            print 'The identities ', list(missing_ids), 'are missing'

            if accumDict['counter'] != 0 and len(set(ids_checked_for_ref1)) < numAnimals:
                print '\nThe identities that we tried to add for references are'
                print 'ids, ', ids_checked_for_ref1
                repeated_ids = set([x for x in ids_checked_for_ref1 if ids_checked_for_ref1.count(x) > 1])
                print 'The identities ', list(repeated_ids), ' are repeated'
                missing_ids = set(all_identities).difference(set(ids_checked_for_ref1))
                print 'The identities ', list(missing_ids), 'are missing'
                raise ValueError('We are adding bad references')

            raise ValueError('We are adding bad references')

    if accumDict['counter'] == 0:
        refDictTemp = {i: refDictTemp[key] for i, key in enumerate(refDictTemp.keys())} # this is done to order the ids in the refDict
        idUsedIntervals = refDictTemp.keys()
        if printFlag:
            print '\n The keys of the refDict are ', refDictTemp.keys()
            print 'The intervals used are ', usedIndivIntervals

    ''' Updating refDict '''
    if accumDict['counter'] == 0:
        iDList = refDictTemp.keys()
    else:
        iDList = refDict.keys()

    min_num_ref_available = 1000000 # We initialize to a high value so that it fullfilles the if conditions inside the loop
    for iD in iDList:
        if accumDict['counter'] == 0:
            refDict[iD] = refDictTemp[iD]
        else:
            if iD in refDictTemp.keys():
                refDict[iD] = np.vstack((refDict[iD],refDictTemp[iD]))

        print 'refDict ', iD, len(refDict[iD])
        if len(refDict[iD]) < min_num_ref_available:
            min_num_ref_available = len(refDict[iD])

    if printFlag:
        print '\nMinimum number of references per identities: ', min_num_ref_available

    ''' I build the images and labels to feed the network '''
    if printFlag:
        print '\nBuilding arrays of images and labels'

    images = []
    labels = []

    for iD in iDList:
        if weighted_flag:
            print 'We are using an unbalanced dataset with weighted loss'
            if len(refDict[iD]) <= maximalRefPerAnimal or accumDict['counter'] == 0:
                print 'The number of references for id ', iD, ' is ', len(refDict[iD]), ', smaller than ', maximalRefPerAnimal
                print 'We take them all'
                imagesList = np.asarray(refDict[iD])
                images.append(imagesList)
                labels.append(np.ones(len(refDict[iD]))*iD)

            elif len(refDict[iD]) > maximalRefPerAnimal:
                print 'The number of references for id ', iD, ' is ', len(refDict[iD]), ', bigger than ', maximalRefPerAnimal
                print 'We sample ', maximalRefPerAnimal, ', .6 from the old ones and .4 from the new ones.'

                ratioOld = .6
                ratioNew = .4

                if iD in refDictTemp.keys():
                    numSamplesNew = maximalRefPerAnimal * ratioNew

                    sampledImagesListNew = np.asarray(refDictTemp[iD])
                    samplesIndexesNew = np.linspace(0,len(sampledImagesListNew)-1,numSamplesNew).astype('int')
                    sampledImagesListNew = sampledImagesListNew[samplesIndexesNew]
                elif iD not in refDictTemp.keys():
                    print iD
                    ratioOld = 1.

                numSamplesOld = maximalRefPerAnimal * ratioOld
                sampledImagesListOld = np.asarray(refDict[iD])
                samplesIndexesOld = np.linspace(0,len(sampledImagesListOld)-1,numSamplesOld).astype('int')
                sampledImagesListOld = sampledImagesListOld[samplesIndexesOld]

                #update the refDict so that it always contains everything
                if iD in refDictTemp.keys():
                    imagesList = np.vstack((sampledImagesListOld, sampledImagesListNew))
                    refDict[iD] = np.vstack((refDict[iD],refDictTemp[iD]))
                elif iD not in refDictTemp.keys():
                    print iD
                    imagesList = sampledImagesListOld

                images.append(imagesList)
                labels.append(np.ones(maximalRefPerAnimal)*iD)

        elif not weighted_flag:
            print 'We are using a balanced dataset '
            if min_num_ref_available <= maximalRefPerAnimal or accumDict['counter'] == 0:
                print 'The minimum number of references is ', min_num_ref_available, ', smaller than ', maximalRefPerAnimal
                print 'We sample the references of this iD to take ', min_num_ref_available

                imagesList = np.asarray(refDict[iD])

                indexes = np.linspace(0,len(imagesList)-1,min_num_ref_available).astype('int')
                images.append(imagesList[indexes])
                labels.append(np.ones(min_num_ref_available)*iD)

            elif min_num_ref_available > maximalRefPerAnimal:
                print 'The minimum number of references is ', min_num_ref_available, ', bigger than ', maximalRefPerAnimal
                print 'We sample ', maximalRefPerAnimal, ', .6 from the old ones and .4 from the new ones.'

                ratioOld = .6
                ratioNew = .4

                if iD in refDictTemp.keys():
                    numSamplesNew = maximalRefPerAnimal * ratioNew

                    sampledImagesListNew = np.asarray(refDictTemp[iD])
                    samplesIndexesNew = np.linspace(0,len(sampledImagesListNew)-1,numSamplesNew).astype('int')
                    sampledImagesListNew = sampledImagesListNew[samplesIndexesNew]
                elif iD not in refDictTemp.keys():
                    print iD
                    ratioOld = 1.

                numSamplesOld = maximalRefPerAnimal * ratioOld
                sampledImagesListOld = np.asarray(refDict[iD])
                samplesIndexesOld = np.linspace(0,len(sampledImagesListOld)-1,numSamplesOld).astype('int')
                sampledImagesListOld = sampledImagesListOld[samplesIndexesOld]

                #update the refDict so that it always contains everything
                if iD in refDictTemp.keys():
                    imagesList = np.vstack((sampledImagesListOld, sampledImagesListNew))
                    refDict[iD] = np.vstack((refDict[iD],refDictTemp[iD]))
                elif iD not in refDictTemp.keys():
                    print iD
                    imagesList = sampledImagesListOld

                images.append(imagesList)
                labels.append(np.ones(maximalRefPerAnimal)*iD)

    print 'images and labels should have the same length:'
    print 'length labels ', [len(lab) for lab in labels]
    print 'length images ', [len(im) for im in images]
    print 'max num of references: ', maximalRefPerAnimal
    print '-------------------------------------------------'
    images = np.vstack(images)
    images = np.expand_dims(images,axis=3)
    images = cropImages(images,32)
    print 'shape of the images ', images.shape
    print 'length of labels', len(labels)
    imagesDims = images.shape
    imsize = (imagesDims[1],imagesDims[2], imagesDims[3])
    labels = flatten(labels)
    print 'size labels after flatten', len(labels)
    # print 'label sample', labels[0:10]
    labels = map(int,labels)
    print np.unique(labels)
    labels0Indices = np.where(np.asarray(labels) == 0)
    print 'labels0Indices, ', labels0Indices
    labels = dense_to_one_hot(labels, numAnimals)
    numImages = len(labels)

    # Standarization of images
    # print 'Standarizing images...'
    # if accumDict['counter'] == 0:
    #     images133 = images[labels0Indices]
    #     pickle.dump(images133,open('./finetuner-imagesInterval0-133-NOstandarized.pkl','wb'))

    images = standarizeImages(images)

    # if accumDict['counter'] == 0:
    #     images133 = images[labels0Indices]
    #     pickle.dump(images133,open('./finetuner-imagesInterval0-133-standarized.pkl','wb'))

    # np.random.seed(0)
    perm = np.random.permutation(numImages)
    images = images[perm]
    labels = labels[perm]

    numTrain = np.ceil(np.true_divide(numImages,10)*9).astype('int')
    X_train = images[:numTrain]
    Y_train = labels[:numTrain]
    X_val = images[numTrain:]
    Y_val = labels[numTrain:]

    # Data augmentation
    # if len(X_train) < 500*numAnimals:
    #     print 'Performing data augmentation...'
    #     X_train, Y_train = dataAugment(X_train,Y_train,dataAugment = True)

    print 'X_train shape, ', X_train.shape
    print 'X_val shape, ', X_val.shape

    ''' Update dictionary of references '''
    trainDict['refDict'] = refDict
    trainDict['framesColumnsRefDict'] = framesColumnsRefDict
    trainDict['usedIndivIntervals'] = usedIndivIntervals
    trainDict['idUsedIntervals'] = idUsedIntervals

    return imsize, X_train, Y_train, X_val, Y_val, trainDict

def fineTuner(videoPath,
            accumDict, trainDict, fragmentsDict, handlesDict,
            portraits, statistics, videoInfo = [],
            plotFlag = True,
            printFlag = True,
            onlyFullyConnected = True,
            onlySoftmax = True,
            weighted_flag = True):

    if printFlag:
        print '\n--- Entering the fineTuner ---'
        if onlyFullyConnected:
            print 'you will train only fully-connected and softmax'
        elif onlySoftmax:
            print 'you will train only the softmax'
        else:
            print 'you will train the entire network'

    # Load data if needed
    if videoInfo == []:
        videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')

    # get information from videoInfo and portraits
    numFrames =  len(portraits)
    numAnimals = int(videoInfo['numAnimals'])
    maxNumBlobs = videoInfo['maxNumBlobs']

    # get information from trainDict
    loadCkpt_folder = trainDict['load_ckpt_folder']
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
    trainDict = DataFineTuning(accumDict, trainDict, fragmentsDict, portraits, statistics, numAnimals, weighted_flag)

    if printFlag:
        print '\n fine tune train size:    images  labels'
        print X_train.shape, Y_train.shape
        print 'validation fine tune size:    images  labels'
        print X_val.shape, Y_val.shape

    width, height, channels = imsize
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
                    width, height, channels, classes,
                    trainDict, accumDict, fragmentsDict, handlesDict, portraits,
                    Tindices, Titer_per_epoch,
                    Vindices, Viter_per_epoch,
                    onlyFullyConnected = onlyFullyConnected,
                    onlySoftmax = onlySoftmax,
                    weighted_flag = weighted_flag)

    trainDict['load_ckpt_folder'] = ckpt_dir
    return trainDict, handlesDict
