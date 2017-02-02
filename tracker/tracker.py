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

def getAvVelFragment(portraits,framesAndColumns):
    centroids = []
    for (frame,column) in framesAndColumns:
        centroids.append(portraits.loc[frame,'centroids'][column])
    centroids = np.asarray(centroids).astype('float32')
    vels = np.sqrt(np.sum(np.diff(centroids,axis=0)**2,axis=1))
    return np.mean(vels)

def DataFineTuning(fragsForTrain, fragmentsDict, portraits,numAnimals):
    fragments = np.asarray(fragmentsDict['fragments'])
    # print 'fragments ', fragments
    framesAndBlobColumns = fragmentsDict['framesAndBlobColumnsDist']
    # print 'framesAndBlobColumns ', framesAndBlobColumns
    minLenIndivCompleteFragments = fragmentsDict['minLenIndivCompleteFragments']
    # print 'minLenIndivCompleteFragments', minLenIndivCompleteFragments

    intervals = fragmentsDict['intervals']
    # print 'intervals ', intervals
    ''' First I save all the images of each identified individual in a dictionary '''
    usedIndivIntervals = [] # We do not want to reuse individual fragments used already
    refDict = {}
    print '**** Creating dictionary of references ****'
    for j, frag in enumerate(fragsForTrain): # for each comple fragment that has to be used for the training
        # print 'fragment for finetuning number, ', j
        fragment = fragments[frag] # I take the fragment
        # print 'fragment ', fragment
        framesColumnsIndivFrags = framesAndBlobColumns[frag] # I take the list of individual fragments in frames and columns
        # print 'framesColumnsIndivFrags ', framesColumnsIndivFrags
        intervalsIndivFrags = intervals[frag] # I take the list of individual fragments in terms of intervals
        # print 'intervalsIndivFrags', intervalsIndivFrags
        # print '------------------------------------------------------------'
        # print 'the big zip', zip(framesColumnsIndivFrags,intervalsIndivFrags)
        # print '------------------------------------------------------------'
        for i, (framesColumnsIndivFrag,intervalsIndivFrag) in enumerate(zip(framesColumnsIndivFrags,intervalsIndivFrags)):
            framesColumnsIndivFrag = np.asarray(framesColumnsIndivFrag)
            # print 'individual fragment, ', i
            if not intervalsIndivFrag in usedIndivIntervals: # I only use individual fragments that have not been used before
                frames = framesColumnsIndivFrag[:,0]
                columns = framesColumnsIndivFrag[:,1]
                identity = portraits.loc[frames[0],'identities'][columns[0]]
                # print 'identity ', identity
                if not identity in refDict.keys(): # if the identity has not been added to the dictionary, I initialize the list
                    refDict[identity] = []
                for frame,column in zip(frames,columns): # I loop in all the frames of the individual fragment to add them to the dictionary of references
                    refDict[identity].append(portraits.loc[frame,'images'][column])

                usedIndivIntervals.append(intervalsIndivFrag)
    # print 'refDict keys ', refDict.keys()
    refDict = {i: refDict[key] for i, key in enumerate(refDict.keys())}
    print 'refDict keys ', refDict.keys()
    # print 'num fish ', numAnimals

    if len(refDict.keys()) != numAnimals:
        raise ValueError('The number of identities should be the same as the number of animals. This means that a global fragment does not have as many individual fragments as number of animals ')

    ''' I compute the minimum number of references I can take '''
    minNumRef = np.min([len(refDict[iD]) for iD in refDict.keys()])
    print 'minNumRef, ', minNumRef
    images = []
    labels = []
    for iD in refDict.keys():
        imagesList = np.asarray(refDict[iD])
        indexes = np.linspace(0,len(imagesList)-1,minNumRef).astype('int')
        images.append(imagesList[indexes])
        labels.append(np.ones(minNumRef)*iD)
    images = np.vstack(images)
    images = np.expand_dims(images,axis=1)
    print 'images shape, ', images.shape
    imagesDims = images.shape
    imsize = (imagesDims[1],imagesDims[2], imagesDims[3])
    labels = flatten(labels)
    labels = map(int,labels)
    print 'labels length, ', len(labels)
    labels = dense_to_one_hot(labels, numAnimals)
    numImages = len(labels)

    perm = np.random.permutation(numImages)
    images = images[perm]
    labels = labels[perm]

    images_max = np.max(images)
    if images_max > 1:
        print 'I am normalizing the images since their maximum is ', images_max
        images = images/255.

    print '\n values of the images: max min'
    print np.max(images), np.min(images)

    numTrain = np.ceil(np.true_divide(numImages,10)*9).astype('int')
    X_train = images[:numTrain]
    Y_train = labels[:numTrain]
    X_val = images[numTrain:]
    Y_val = labels[numTrain:]

    resolution = np.prod(imsize)
    X_train = np.reshape(X_train, [numTrain, resolution])
    X_val = np.reshape(X_val, [numImages - numTrain, resolution])

    return imsize, X_train, Y_train, X_val, Y_val

def DataIdAssignation(portraits, indivFragments):
    portraitsFrag = np.asarray(portraits.loc[:,'images'].tolist())
    height = 32
    width = 32
    imsize = (1, height, width)
    portsFragments = []
    # print 'indivFragments', indivFragments
    for indivFragment in indivFragments:
        # print 'indiv fragment, ', indivFragment
        portsFragment = []
        # print 'portsFragment, ', portsFragment
        for (frame, column) in indivFragment:
            portsFragment.append(portraitsFrag[frame][column])

        portsFragments.append(np.reshape(np.asarray(portsFragment), [len(indivFragment),height*width]))
    # print portsFragments
    return imsize, np.vstack(portsFragments)


def get_batch(batchNum, iter_per_epoch, indices, images_pl, keep_prob_pl, images, keep_prob):
    if iter_per_epoch > 1:
        images_feed = images[indices[batchNum]:indices[batchNum+1]]
    else:
        images_feed = images

    feed_dict = {
      images_pl: images_feed,
      keep_prob_pl: keep_prob
    }
    return feed_dict

def run_batch(sess, opsList, indices, batchNum, iter_per_epoch, images_pl, keep_prob_pl, images, keep_prob):
    feed_dict = get_batch(batchNum, iter_per_epoch, indices, images_pl, keep_prob_pl, images, keep_prob)
    # Run forward step to compute loss, accuracy...
    outList = sess.run(opsList, feed_dict=feed_dict)
    outList.append(feed_dict)

    return outList

def fragmentProbId(X_t, width, height, channels, classes, resolution, loadCkpt_folder, batchSize, Tindices, Titer_per_epoch , keep_prob = 1.0):
    with tf.Graph().as_default():

        images_pl = tf.placeholder(tf.float32, [None, resolution], name = 'images')
        keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')

        logits, relu, (W1,W3,W5) = inference1(images_pl, width, height, channels, classes, keep_prob_pl)
        predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)

        saver_model = createSaver('soft', False, 'saver_model')
        saver_softmax = createSaver('soft', True, 'saver_softmax')
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.initialize_all_variables().run()

            # Load weights from a pretrained model if there is not any model saved
            # in the ckpt folder of the test
            if loadCkpt_folder:
                # print 'loading weigths from ' + loadCkpt_folder
                restoreFromFolder(loadCkpt_folder + '/model', saver_model, sess)
                restoreFromFolder(loadCkpt_folder + '/softmax', saver_softmax, sess)

            else:
                warnings.warn('It is not possible to perform knowledge transfer, give a folder containing a trained model')

            # print "Start from:", start
            opListProb = [predictions, logits, relu]

            ''' Getting idProb fragment '''
            softMaxId = []
            softMaxProbs = []
            for iter_i in range(Titer_per_epoch):

                predictions, logits, relu, feed_dict = run_batch(
                    sess, opListProb, Tindices, iter_i, Titer_per_epoch,
                    images_pl, keep_prob_pl,
                    X_t, keep_prob = keep_prob)

                exp = np.exp(logits)
                sumsExps = np.sum(exp,axis=1)
                probs = []
                for i, sumExps in enumerate(sumsExps):
                    probs.append(np.true_divide(exp[i,:],sumExps))

                softMaxProbs.append(probs)
                softMaxId.append(predictions)

            softMaxId = flatten(softMaxId)
            softMaxProbs = flatten(softMaxProbs)
    return np.asarray(softMaxProbs), np.asarray(softMaxId).astype('int')

def computeP1(IdProbs):
    """
    IdProbs: list of IdProb, [IdProb1, IdProb2]
    IdProb: are the probabilities of assignment for each frame of the individual fragment (the output of the softmax for each portrait of the individual fragment)
    IdProb.shape: (num frames in fragment, maxNumBlobs, numAnimals)
    """
    float_info = sys.float_info
    maxFloat = float_info[0]
    P1Frags = []
    P1FragsForMat = []
    freqFrags = []
    idFreqFragForMat = []
    normFreqFrags = []
    normFreqFragsForMat = []
    numAnimals = IdProbs[0].shape[1]
    for IdProb in IdProbs: #looping on the individual fragments
        # IdTracker way assuming that the indivValAcc is the probability of good assignment for each identity
        # idsFragment = np.argmax(IdProb,axis=1) # assignment of identities for single frames according to the output of the softmax
        idsFragment = []
        for p in IdProb:
            if np.max(p)>0.99:
                idsFragment.append(np.argmax(p))
            else:
                idsFragment.append(np.nan)
        idsFragment = np.asarray(idsFragment)
        # print '---------------------------'
        # print 'idsFragment, ', idsFragment
        fragLen = IdProb.shape[0]
        # print 'numAnimals, ', numAnimals
        frequencies = np.asarray([np.sum(idsFragment == i) for i in range(numAnimals)]) # counting the frequency of every assignment
        # print 'frequencies, ', frequencies
        numerator = 2.**frequencies
        if np.any(numerator == np.inf):
            numerator[numerator == np.inf] = maxFloat
        # print 'numerator, ', numerator
        denominator = np.sum(numerator)
        # print 'denominator, ', denominator
        P1Frag = numerator / denominator
        # print 'P1Frag, ', P1Frag
        if np.any(P1Frag == 0.):
            raise ValueError('P1Frag cannot be 0')
        # P1Frag[P1Frag == 1.] = 1. - np.sum(P1Frag[P1Frag!=1.])
        P1Frag[P1Frag == 1.] = 0.9999
        # print np.sum(P1Frag[P1Frag!=1.])
        # print 'P1Frag, ', P1Frag
        if np.any(P1Frag == 1.):
            raise ValueError('P1Frag cannot be 1')

        idFreqFrag = int(np.argmax(frequencies) + 1)
        normFreqFrag = np.true_divide(frequencies,fragLen)

        P1Frags.append(P1Frag)
        P1FragsForMat.append(np.matlib.repmat(P1Frag,fragLen,1))
        freqFrags.append(np.matlib.repmat(frequencies,fragLen,1))
        normFreqFrags.append(normFreqFrag)
        normFreqFragsForMat.append(np.matlib.repmat(normFreqFrag,fragLen,1))
        idFreqFragForMat.append(np.multiply(np.ones(fragLen),idFreqFrag).astype('int'))

    return P1Frags, P1FragsForMat, freqFrags, normFreqFrags, idFreqFragForMat, normFreqFragsForMat

def computeLogP2Complete(oneIndivFragIntervals, P1FragsAll, indivFragmentsIntervals, P1Frags, lenFragments, blobsIndices):

    def getOverlap(a, b):
        overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] == b[0] or a[0] == b[1]:
            overlap = 1
        if a[1] == a[0] and (b[0]<=a[0] and b[1]>=a[1]):
            overlap = 1
        if b[1] == b[0] and (a[0]<=b[0] and a[1]>=b[1]):
            overlap = 1
        return overlap

    def computeP2(P1Frag,P1CoexistingFrags):
        numerator = P1Frag * np.prod(1.-P1CoexistingFrags,axis=0)
        # if numerator == 0.:
        #     raise ValueError('numerator of P2 is 0')
        denominator = np.sum(numerator)
        if denominator == 0:
            raise ValueError('denominator of P2 is 0')

        P2 = numerator / denominator
        return P2

    def computeLogP2(P1Frag,P1CoexistingFrags):
        logProb1Fragment = np.log(P1Frag)
        # print 'logProb1Fragment, ', logProb1Fragment
        # print 'logCoexisting, ',  np.log(1.-P1CoexistingFrags)
        logCoexisting = np.log(1.-P1CoexistingFrags)
        sumLogProbs1CF = np.sum(logCoexisting,axis=0)

        logP2Frag = logProb1Fragment + sumLogProbs1CF
        return logP2Frag

    logP2FragsForMat = []
    logP2FragIdForMat = []
    P2FragsForMat = []
    P2Frags = []
    for j, (P1Frag,indivFragmentInterval) in enumerate(zip(P1Frags,indivFragmentsIntervals)):
        # print '\nIndividual fragment, ', j, ' ----------------'
        # print 'Interval, ', indivFragmentInterval
        # print 'Individual fragment P1, ', P1Frag
        P1CoexistingFrags = []

        for k in blobsIndices:
            indivFragmentsIntervals2 = oneIndivFragIntervals[k]
            # print 'coexistence in fragments list ', k
            P1FragsK = np.asarray(P1FragsAll[k])
            overlaps = np.asarray([getOverlap(indivFragmentInterval,otherInterval) for otherInterval in indivFragmentsIntervals2])
            coexistingFragments = np.where(overlaps != 0)[0]
            # print 'coexisting Fragments, ', coexistingFragments
            P1CoexistingFrags.append(P1FragsK[coexistingFragments])

        P1CoexistingFrags = np.vstack(P1CoexistingFrags)
        # print 'coexisting Probs, ', P1CoexistingFrags

        P2Frag = computeP2(P1Frag,P1CoexistingFrags)
        logP2Frag = computeLogP2(P1Frag,P1CoexistingFrags)

        # print 'logP2Frag', logP2Frag
        idFrag = np.argmax(logP2Frag)+1
        # print idFrag
        fragLen = lenFragments[j]
        logP2FragsForMat.append(np.matlib.repmat(logP2Frag,fragLen,1))
        P2FragsForMat.append(np.matlib.repmat(P2Frag,fragLen,1))
        logP2FragIdForMat.append(np.multiply(np.ones(fragLen),idFrag).astype('int'))
        P2Frags.append(P2Frag)

    return logP2FragsForMat, logP2FragIdForMat, P2FragsForMat, P2Frags

def computeOverallP2(P2FragsAll,oneIndivFragLens):
    numFrames = 0
    weightedP2 = []
    for P2Frags, oneIndivLens in zip(P2FragsAll,oneIndivFragLens):
        P2Frags = np.asarray(P2Frags)
        P2Frags = np.max(P2Frags,axis=1)
        oneIndivLens = np.asarray(oneIndivLens)
        weightedP2.append(P2Frags*oneIndivLens)
        numFrames += np.sum(oneIndivLens)
    overallP2 = np.true_divide(np.sum(np.asarray(flatten(weightedP2))),numFrames)
    return overallP2

# fragments =  pickle.load(open('/home/lab/Desktop/TF_models/IdTracker/Medaka/20161129174648_/fragments.pkl','rb'))
# oneIndivFragLens = fragments['oneIndivFragLens']



def idUpdater(ids,indivFragments,numFrames,maxNumBlobs):
    IdsArray = np.zeros((numFrames,maxNumBlobs))

    for (Id,indivFragment) in zip(ids,indivFragments):
        frames = np.asarray(indivFragment)[:,0]
        columns = np.asarray(indivFragment)[:,1]
        IdsArray[frames,columns] = Id[0]
    return IdsArray

def probsUptader(vectorPerFrame,indivFragments,numFrames,maxNumBlobs,numAnimals):
    ProbsArray = np.zeros((numFrames,maxNumBlobs,numAnimals))
    for (vectorPerFrame,indivFragment) in zip(vectorPerFrame,indivFragments):
        frames = np.asarray(indivFragment)[:,0]
        columns = np.asarray(indivFragment)[:,1]
        ProbsArray[frames,columns,:] = vectorPerFrame[0]

    return ProbsArray

def getCkptvideoPath(videoPath, accumCounter, train=0,ckptTime=0):
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

    if train == 0 or train == 2:
        sessionPath = subFolder + '/Session_' + str(lastIndex)
        ckptvideoPath = sessionPath + '/AccumulationStep_' + str(accumCounter)
        if train == 0:
            print 'you will assign identities from the last checkpoint in ', ckptvideoPath
        else:
            print 'you will keep training from the last checkpoint in ', ckptvideoPath

    elif train == 1:
        sessionPath = subFolder + '/Session_' + str(lastIndex + 1)
        os.makedirs(sessionPath)
        print 'You just created ', sessionPath
        figurePath = sessionPath + '/figures'
        os.makedirs(figurePath)
        print 'You just created ', figurePath
        ckptvideoPath = sessionPath + '/AccumulationStep_' + str(accumCounter)
        print 'model checkpoints will be saved in ', ckptvideoPath

    figurePath = sessionPath + '/figures'
    return ckptvideoPath, figurePath

def fineTuner(videoPath, trainDict, fragsForTrain, accumCounter, lossAccDict, fragmentsDict = [], portraits = [], videoInfo = []):
    """
    videoPath: path to the video that we want to tracker
    trainDict: includes
        ckptName: name of the checkpoint folder where we will save the model used to infer the identities
        loadCkpt_folder: folder where to load the master model that we will fine tune with the new references
        batchSize: bath size used for the training of the fine tuning
        num_epochs: number of epochs for the fine tuning
        lr: learning rate for the optimization algorithim for the fine tuning
        train:  train = 0 (id assignation)
                train = 1 (first fine-tuning)
                train = 2 (further tuning from previons checkpoint with more references) no implemented yet, now it keeps training from the previous checkpoint with the same references
    fragsForTrain: list of fragments indices that should be used for the fine-tunning
    fragmentsDict:
    portraits:
    """
    print '--------------------------------'
    print 'Loading fragments, portraits and videoInfo for the tracking...'
    if fragmentsDict == []:
        fragmentsDict = loadFile(videoPath, 'fragments')
        fragmentsDict = fragmentsDict.to_dict()[0]
    if len(portraits) == 0:
        portraits = loadFile(videoPath, 'portraits')
    if videoInfo == []:
        videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')

    numFrames =  len(portraits)
    numAnimals = int(videoInfo['numAnimals'])
    maxNumBlobs = videoInfo['maxNumBlobs']

    loadCkpt_folder = trainDict['loadCkpt_folder']
    ckptName = trainDict['ckptName']
    batchSize = trainDict['batchSize']
    numEpochs =  trainDict['numEpochs']
    lr = trainDict['lr']
    train = trainDict['train']

    ''' ************************************************************************
    Fine tuning with the longest fragment
    ************************************************************************ '''
    print 'Fine tuning with the first longest fragment...'
    ckpt_dir, fig_dir = getCkptvideoPath(videoPath, accumCounter, train)
    imsize,\
    X_train, Y_train,\
    X_val, Y_val = DataFineTuning(fragsForTrain, fragmentsDict, portraits,numAnimals)

    print '\n fine tune train size:    images  labels'
    print X_train.shape, Y_train.shape
    print 'val fine tune size:    images  labels'
    print X_val.shape, Y_val.shape

    channels, width, height = imsize
    resolution = np.prod(imsize)
    classes = numAnimals

    numImagesT = Y_train.shape[0]
    numImagesV = Y_val.shape[0]
    Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batchSize)
    Vindices, Viter_per_epoch = get_batch_indices(numImagesV,batchSize)

    print 'running with the devil'
    print 'ckpt_dir', ckpt_dir
    print 'loadCkpt_folder', loadCkpt_folder
    lossAccDict = run_training(X_train, Y_train, X_val, Y_val,
                    width, height, channels, classes, resolution,
                    ckpt_dir, fig_dir, loadCkpt_folder, accumCounter,
                    batchSize,numEpochs, lossAccDict,
                    Tindices, Titer_per_epoch,
                    Vindices, Viter_per_epoch,
                    1.,lr) #dropout
    return lossAccDict

def idAssigner(videoPath,trainDict,accumCounter, ckpt_dir, fragmentsDict = [],portraits = [], videoInfo = []):
    '''
    videoPath: path to the video to which we want ot assign identities
    ckptName: name of the checkpoint folder with the model for the assignation of identities
    '''
    if len(portraits) == 0:
        portraits = loadFile(videoPath, 'portraits')

    if len(videoInfo) == 0:
        videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')
    print videoInfo
    numFrames =  len(portraits)
    numAnimals = videoInfo['numAnimals']
    maxNumBlobs = videoInfo['maxNumBlobs']
    meanIndivArea = videoInfo['meanIndivArea']
    stdIndivArea = videoInfo['stdIndivArea']

    if len(fragmentsDict) == 0:
        fragmentsDict = loadFile(videoPath, 'fragments', hdfpkl='pkl')
    oneIndivFragFrames = fragmentsDict['oneIndivFragFrames']
    oneIndivFragIntervals = fragmentsDict['oneIndivFragIntervals']
    oneIndivFragSumLens = fragmentsDict['oneIndivFragSumLens']
    oneIndivFragLens = fragmentsDict['oneIndivFragLens']

    lensIntervalsLists = [len(oneIndivFragInterval) for oneIndivFragInterval in oneIndivFragIntervals]
    numGoodLists = np.sum(np.asarray(lensIntervalsLists) != 0)

    ckptName = trainDict['ckptName']
    batchSize = 1000
    ckpt_dir, fig_dir = getCkptvideoPath(videoPath,accumCounter,train=0,ckptTime=0)

    '''
    Loop to IndivFragments, Ids in frames of IndivFragments, P1 given the Ids
    '''
    softMaxProbsAll = []
    softMaxIdAll = []
    P1FragsAll = []
    freqFragsAll = []
    normFreqFragsAll = []

    idSoftMaxAllVideo = -np.ones((numFrames,maxNumBlobs)) # softMax predicted ids per frame
    PSoftMaxAllVIdeo = np.zeros((numFrames,maxNumBlobs,numAnimals)) # softMax probabilities per frame
    freqFragAllVideo = np.zeros((numFrames,maxNumBlobs,numAnimals)) # frequencies for each individual fragment
    normFreqFragAllVideo = np.zeros((numFrames,maxNumBlobs,numAnimals))
    idFreqFragAllVideo= -np.ones((numFrames,maxNumBlobs))
    P1FragAllVideo = np.zeros((numFrames,maxNumBlobs,numAnimals)) # P1 for each individual fragment
    for i, (indivFragments, sumFragIndices) in enumerate(zip(oneIndivFragFrames,oneIndivFragSumLens)):
        print '******************************************************'
        print 'Computing softMax probabilities, id-frequencies, and P1 for list of fragments ', i
        # Load data for the assignment (### TODO this can be done in portraits so that we only need to do it once)
        if len(indivFragments) != 0:
            imsize, portsFragments =  DataIdAssignation(portraits, indivFragments)

            images_max = np.max(portsFragments)
            if images_max > 1:
                print 'I am normalizing the images since their maximum is ', images_max
                portsFragments = portsFragments/255.

            print '\n values of the images during identity assigation: max min'
            print np.max(portsFragments), np.min(portsFragments)
            # Set variables for the forward pass
            loadCkpt_folder = ckpt_dir
            channels, width, height = imsize
            resolution = np.prod(imsize)
            classes = numAnimals
            numImagesT = len(portsFragments) #FIXME
            # Get batch indices
            Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batchSize)
            print 'indices test ', Tindices
            print 'iteration per epoch ', Titer_per_epoch
            # Run forward pass,
            softMaxProbs, softMaxId = fragmentProbId(portsFragments, width, height, channels,
                classes, resolution, loadCkpt_folder, batchSize, Tindices, Titer_per_epoch)

            # list of softmax probabilities for each one-individual fragment
            softMaxProbs = np.split(softMaxProbs,sumFragIndices)
            softMaxId = np.split(softMaxId,sumFragIndices)
            softMaxProbsAll.append(softMaxProbs)
            softMaxIdAll.append(softMaxId)

            P1Frags, P1FragsForMat, freqFrags, normFreqFrags, idFreqFragForMat, normFreqFragsForMat = computeP1(softMaxProbs)
            P1FragsAll.append(P1Frags)
            freqFragsAll.append(freqFrags)
            normFreqFragsAll.append(normFreqFrags)

            # Probabilities from the softMax for the whole video
            ProbsUpdated = probsUptader(softMaxProbs,indivFragments,numFrames,maxNumBlobs,numAnimals)
            PSoftMaxAllVIdeo += ProbsUpdated

            # Identities from the softMax for the whole video
            IdsUpdated = idUpdater(softMaxId,indivFragments,numFrames,maxNumBlobs)
            idSoftMaxAllVideo += IdsUpdated

            # Frequencies per fragment for the whole video (considering the threshold in the probabilities of the softmax)
            FreqFragUpdated = probsUptader(freqFrags,indivFragments,numFrames,maxNumBlobs,numAnimals)
            freqFragAllVideo += FreqFragUpdated

            # Normalized frequencies per fragment for the whole video (considering the threshold in the probabilities of the softmax)
            normFreqFragUpdated = probsUptader(normFreqFragsForMat,indivFragments,numFrames,maxNumBlobs,numAnimals)
            normFreqFragAllVideo += normFreqFragUpdated

            IdsFragUpdated = idUpdater(idFreqFragForMat,indivFragments,numFrames,maxNumBlobs)
            idFreqFragAllVideo += IdsFragUpdated

            # P1 probabilities
            P1FragUpdated =probsUptader(P1FragsForMat,indivFragments,numFrames,maxNumBlobs,numAnimals)
            P1FragAllVideo += P1FragUpdated

    idFreqFragAllVideo = idFreqFragAllVideo.astype('int')
    idSoftMaxAllVideo = idSoftMaxAllVideo.astype('int')
    freqFragAllVideo = freqFragAllVideo.astype('int')

    '''
    Loop to Compute P2 for each individual fragment
    This loop can be parallelized
    '''
    idLogP2FragAllVideo= -np.ones((numFrames,maxNumBlobs)) # fragments identities assigned from P2 to each individual fragment
    logP2FragAllVideo = np.zeros((numFrames,maxNumBlobs,numAnimals)) # logP2 for each individual fragment
    P2FragAllVideo = np.zeros((numFrames,maxNumBlobs,numAnimals))
    P2FragsAll = []
    print '******************************************************'
    for i in range(numGoodLists):
        # print '******************************************************'
        print 'Computing logP2 for list of fragments, ', i
        indivFragmentsIntervals = oneIndivFragIntervals[i]
        P1Frags = P1FragsAll[i]
        indivFragments = oneIndivFragFrames[i]
        lenFragments = oneIndivFragLens[i]
        freqFrags = freqFragsAll[i]
        softMaxProbs = softMaxProbsAll[i]
        softMaxId = softMaxIdAll[i]

        logP2FragsForMat = []
        P1FragsForMat = []
        logP2FragIdForMat = []

        blobsIndices = list(range(numGoodLists))
        blobsIndices.pop(i)

        logP2FragsForMat, logP2FragIdForMat, P2FragsForMat, P2Frags = computeLogP2Complete(oneIndivFragIntervals, P1FragsAll, indivFragmentsIntervals, P1Frags, lenFragments, blobsIndices)
        P2FragsAll.append(P2Frags)
        # logP2
        LogProbsFragUpdated = probsUptader(logP2FragsForMat,indivFragments,numFrames,maxNumBlobs,numAnimals)
        logP2FragAllVideo += LogProbsFragUpdated

        # P2
        ProbsFragUpdated = probsUptader(P2FragsForMat,indivFragments,numFrames,maxNumBlobs,numAnimals)
        P2FragAllVideo += ProbsFragUpdated

        # identities from logP2
        IdsFragUpdated = idUpdater(logP2FragIdForMat,indivFragments,numFrames,maxNumBlobs)
        idLogP2FragAllVideo += IdsFragUpdated

    SaveAllP2s = []
    sessionPath = '/'.join(ckpt_dir.split('/')[:-1])
    overallP2 = computeOverallP2(P2FragsAll,oneIndivFragLens)
    SaveAllP2s.append(overallP2)
    pickle.dump( SaveAllP2s , open( sessionPath + "/P2s.pkl", "wb" ) )
    print '**** overallP2, ', overallP2
    idLogP2FragAllVideo = idLogP2FragAllVideo.astype('int')
    IdsStatistics = {'blobIds':idSoftMaxAllVideo,
        'probBlobIds':PSoftMaxAllVIdeo,
        'FreqFrag': freqFragAllVideo,
        'normFreqFragAllVideo': normFreqFragAllVideo,
        'idFreqFragAllVideo': idFreqFragAllVideo,
        'P1Frag': P1FragAllVideo,
        'fragmentIds':idLogP2FragAllVideo,
        'probFragmentIds':logP2FragAllVideo,
        'P2FragAllVideo':P2FragAllVideo,
        'overallP2': overallP2}

    portraits['identities'] = idFreqFragAllVideo.tolist()
    # e(videoPath,portraits,'portraits',time=0)

    pickle.dump( IdsStatistics , open( ckpt_dir + "/statistics.pkl", "wb" ) )
    pickle.dump( IdsStatistics , open( sessionPath + "/statistics.pkl", "wb" ) )
    # saveFile(ckpt_dir, IdsStatistics, 'statistics',hdfpkl='pkl')
    return normFreqFragsAll, portraits

def bestFragmentFinder(fragsForTrain,normFreqFragsAll,fragmentsDict,numAnimals,minDistTrav,badFragments,portraits,thVels):
    print ' ****************** Finding next best fragment for references\n'
    # Load data needed and pass it to arrays
    fragments = np.asarray(fragmentsDict['fragments'])
    framesAndBlobColumns = fragmentsDict['framesAndBlobColumns']
    minLenIndivCompleteFragments = fragmentsDict['minLenIndivCompleteFragments']
    minDistIndivCompleteFragments = fragmentsDict['minDistIndivCompleteFragments']
    lens = np.asarray(minLenIndivCompleteFragments)
    distsTrav = np.asarray(minDistIndivCompleteFragments)
    intervalsFragments = fragmentsDict['intervals']

    # Compute distances to the identity matrix for each complete set of individual fragments
    mat = []
    distI = []
    identity = np.identity(numAnimals)
    for i, intervals in enumerate(intervalsFragments): # loop in complete set of fragments
        # print 'fragment, ', i
        for j, interval in enumerate(intervals): # loop in individual fragments of the complete set of fragments
            # print 'individual fragment, ', j
            mat.append(normFreqFragsAll[interval[0]][interval[1]])
        matFragment = np.vstack(mat)
        mat = []
        perm = np.argmax(matFragment,axis=1)

        matFragment = matFragment[:,perm]
        # print matFragment
        # print numpy.linalg.norm(matFragment - identity)
        # print lens[i]
        distI.append(numpy.linalg.norm(matFragment - identity)) #TODO when optimizing the code one should compute the matrix distance only for fragments above 100 length
    distI = np.asarray(distI)
    distInorm = distI/np.max(distI)
    # lensnorm = np.true_divide(lens,np.max(lens))
    distsTravNorm = np.true_divide(distsTrav,np.max(distsTrav))

    # Get best values of the parameters length and distance to identity
    distI0norm = np.min(distInorm[fragsForTrain])
    distI0norm = np.ones(len(distI))*distI0norm
    # len0norm = np.max(lensnorm[fragsForTrain])
    # len0norm = np.ones(len(lens))*len0norm
    distTrav0norm = np.max(distsTravNorm[fragsForTrain])
    distTrav0norm = np.ones(len(distsTrav))*distTrav0norm

    # Compute score of every global fragment with respect to the optimal value of the parameters
    # score = np.sqrt((distI0norm-distInorm)**2 + ((len0norm-lensnorm))**2)
    score = np.sqrt((distI0norm-distInorm)**2 + ((distTrav0norm-distsTravNorm))**2)

    # Get indicies of the best fragments according to its score
    fragIndexesSorted = np.argsort(score).tolist()

    # Remove short fragments
    print '(fragIndexesSorted, distTrav, lens) before eliminating the short ones, ', zip(fragIndexesSorted,distsTrav[fragIndexesSorted],lens[fragIndexesSorted])
    # print 'Current minimum length, ', minLen
    print 'Current minimum distance travelled, ', minDistTrav
    fragIndexesSortedLong = [x for x in fragIndexesSorted if distsTrav[x] > minDistTrav]
    print '(fragIndexesSorted, distTrav, lens) after eliminating the short ones, ', zip(fragIndexesSortedLong,distsTrav[fragIndexesSortedLong],lens[fragIndexesSortedLong])

    # We only consider fragments that have not been already picked for fine-tuning
    print 'Current fragsForTrain, ', fragsForTrain
    print 'Bad fragments, ', badFragments
    nextPossibleFragments = [frag for frag in fragIndexesSortedLong if frag not in fragsForTrain and frag not in badFragments]

    # Check whether the animals are moving enough so that the images are going to be different enough
    realNextPossibleFragments = []
    for frag in nextPossibleFragments:
        framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][frag]
        avVels = []
        print 'Checking whether the fragmentNumber ', frag, ' is good for training'
        for framesAndColumnsInterval in framesAndColumnsGlobalFrag:
            avVels.append(getAvVelFragment(portraits,framesAndColumnsInterval))
        print 'The average velocities for each blob are (pixels/frame), ', avVels
        if any(np.asarray(avVels)<=thVels):
            badFragments.append(frag)
            print 'There is some animal that does not move enough'
            print 'Bad fragments, ', badFragments
        else:
            realNextPossibleFragments.append(frag)

    while len(realNextPossibleFragments)==0 and minDistTrav >= 0:
        print 'The list of possible fragments is the same as the list of fragments used previously for finetuning'
        print 'We reduce the minLen in 50 units'
        if minDistTrav > 50:
            step = 50
        else:
            step = 10
        minDistTrav -= step
        print 'Current minimum distTrav, ', minDistTrav
        fragIndexesSortedLong = [x for x in fragIndexesSorted if distsTrav[x] > minDistTrav]

        # We only consider fragments that have not been already picked for fine-tuning
        print 'Current fragsForTrain, ', fragsForTrain
        print 'Bad fragments, ', badFragments
        nextPossibleFragments = [frag for frag in fragIndexesSortedLong if frag not in fragsForTrain and frag not in badFragments]
        nextPossibleFragments = np.asarray(nextPossibleFragments)
        print 'Next possible fragments for train', nextPossibleFragments

        realNextPossibleFragments = []
        for frag in nextPossibleFragments:
            framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][frag]
            avVels = []
            print 'Checking whether the fragmentNumber ', frag, ' is good for training'
            for framesAndColumnsInterval in framesAndColumnsGlobalFrag:
                avVels.append(getAvVelFragment(portraits,framesAndColumnsInterval))
            print 'The average velocities for each blob are (pixels/frame), ', avVels
            if any(np.asarray(avVels)<=thVels):
                badFragments.append(frag)
                print 'There is some animal that does not move enough'
                print 'Bad fragments, ', badFragments
            else:
                realNextPossibleFragments.append(frag)

    nextPossibleFragments = np.asarray(realNextPossibleFragments)
    print 'Next possible fragments for train', nextPossibleFragments


    if len(nextPossibleFragments) != 0:

        lensND = np.asarray([lens[frag] for frag in nextPossibleFragments])
        distsTravND = np.asarray([distsTrav[frag] for frag in nextPossibleFragments])
        distIND = np.asarray([distI[frag] for frag in nextPossibleFragments])
        print '(len, distTrav, dist) of the nextPossibleFragments', zip(lensND,distsTravND,distIND)

        bestFragInd = nextPossibleFragments[0]
        bestFragDist = distI[bestFragInd]
        bestFragLen = lens[bestFragInd]
        bestFragDistTrav = distsTrav[bestFragInd]
        print 'BestFragInd, bestFragDist, bestFragLen, bestFragDistTrav'
        print bestFragInd, bestFragDist, bestFragLen, bestFragDistTrav

        # acceptableFragIndices = np.where((lensND > 100) & (distIND <= bestFragDist))[0]
        acceptable = np.where((distIND <= bestFragDist))[0]
        acceptableFragIndices = nextPossibleFragments[acceptable]

        fragsForTrain = np.asarray(fragsForTrain)
        print 'Old Frags for train, ', fragsForTrain
        print 'acceptableFragIndices, ', acceptableFragIndices
        fragsForTrain = np.unique(np.hstack([fragsForTrain,acceptableFragIndices])).tolist()
        print 'Fragments for training, ', fragsForTrain

        continueFlag = True
    else:
        print 'There are no more good fragments'
        continueFlag = False

    return fragsForTrain, continueFlag, minDistTrav, badFragments
