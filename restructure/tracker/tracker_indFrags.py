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
from itertools import groupby
from pprint import pprint

def takeImagesLabels(images,identity,minNumRef): # this function runs in parallel
    imagesList = np.asarray(images)
    indexes = np.linspace(0,len(imagesList)-1,minNumRef).astype('int')
    return imagesList[indexes], np.ones(minNumRef)*identity

def takeRefFromDict(refDict,indivFragsForTrain): ### TODO this function could be moved to input_data_cnn.py
    ''' I compute the minimum number of references I can take '''
    print 'lengths of refDict, ', [len(refDict[iD]) for iD in refDict.keys()]
    minNumRef = np.min([len(refDict[iD]) for iD in refDict.keys()])
    print 'minNumRef, ', minNumRef
    numAnimals = len(refDict.keys())

    num_cores = multiprocessing.cpu_count()
    num_cores = 1
    output = Parallel(n_jobs=num_cores)(delayed(takeImagesLabels)(refDict[identity],identity,minNumRef) for identity in range(numAnimals))
    images = [t[0] for t in output]
    labels = [t[1] for t in output]

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

def getImagesRef(indivFrags,oneIndivFragFrames,identity,portraits): # this function runs in parallel
    images = []
    for indivFrag in indivFrags:
        blobIndex = indivFrag[2]
        indivFragNumber = indivFrag[3]
        frames = oneIndivFragFrames[blobIndex][indivFragNumber][:,0]
        columns = oneIndivFragFrames[blobIndex][indivFragNumber][:,1]
        for frame,column in zip(frames,columns): # I loop in all the frames of the individual fragment to add them to the dictionary of references
            images.append(portraits.loc[frame,'images'][column])

    return images, identity

def DataFineTuning(indivFragsForTrain, fragmentsDict, portraits,numAnimals):
    print '--------------------------------------'
    print 'fragments for training, '
    pprint(indivFragsForTrain)
    print '--------------------------------------'

    print '\n Getting images for fine-tuning'
    oneIndivFragFrames = fragmentsDict['oneIndivFragFrames']

    refDict = {}
    identities = range(numAnimals)
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    print '\n Preparing refDict'
    output = Parallel(n_jobs=num_cores)(delayed(getImagesRef)(indivFragsForTrain[identity],oneIndivFragFrames,identity,portraits) for identity in identities)
    for (images,identity) in output:
        print 'identity, ', identity, 'num images,',len(images)
        refDict[identity] = images

    print '\n Preparing data for CNN'
    imsize, X_train, Y_train, X_val, Y_val = takeRefFromDict(refDict,indivFragsForTrain)



    return imsize, X_train, Y_train, X_val, Y_val

def DataIdAssignation(portraits, indivFragments):
    portraitsFrag = np.asarray(portraits.loc[:,'images'].tolist())
    height,width = portraits.loc[0,'images'][0].shape
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
        if np.any(denominator == np.inf):
            denominator = maxFloat
        # print 'denominator, ', denominator
        P1Frag = numerator / denominator
        # print 'P1Frag, ', P1Frag
        if np.any(P1Frag == 0.):
            print 'frequencies, ', frequencies
            print 'numerator, ', numerator
            print 'denominator, ', denominator
            print 'P1Frag, ', P1Frag
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

def computeLogP2Complete(oneIndivFragIntervals, P1FragsAll, indivFragmentsIntervals, P1Frags, lenFragments, velFragments, distFragments, blobsIndices,blobIndex):

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
    for j, (P1Frag,indivFragmentInterval) in enumerate(zip(P1Frags,indivFragmentsIntervals)): # loop for every interval in the current blob index
        # print '\nIndividual fragment, ', j, ' ----------------'
        # print 'Interval, ', indivFragmentInterval
        # print 'Individual fragment P1, ', P1Frag
        lenFragment = lenFragments[j]
        velFragment = velFragments[j]
        distFragment = distFragments[j]
        P1CoexistingFrags = []
        # indivFragmentInterval2 = indivFragmentInterval
        for k in blobsIndices: # looping on the blob indices to compute the coexistence of the individual fragments
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
        flatIndinvFragmentInterval = (blobIndex,j,) + tuple(flatten(indivFragmentInterval)) + (lenFragment,) + (velFragment,) + (distFragment,)
        # fragIdAndP2 = (np.argmax(P2Frag),np.max(P2Frag))
        fragIdAndP2 = (np.argmax(P1Frag),np.max(P1Frag))
        P2Frags.append(fragIdAndP2 + flatIndinvFragmentInterval)
        # print 'the fucking things we are putting in this fucking list ', fragIdAndP2 + flatIndinvFragmentInterval
    return logP2FragsForMat, logP2FragIdForMat, P2FragsForMat, P2Frags

def computeOverallP2(P2FragsAll,oneIndivFragLens):
    numFrames = 0
    weightedP2 = []
    for P2Frags, oneIndivLens in zip(P2FragsAll,oneIndivFragLens):
        P2Frags = np.asarray(P2Frags)
        P2Frags = P2Frags[:,1] #np.max(P2Frags,axis=1)
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

def getCkptvideoPath(videoPath,ckptName,train=0,time=0,ckptTime=0,):
    """
    train = 0 (id assignation)
    train = 1 (first fine-tuning)
    train = 2 (further tuning from previons checkpoint with more references)
    """

    video = os.path.basename(videoPath)
    folder = os.path.dirname(videoPath)
    filename, extension = os.path.splitext(video)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
    subFolder = subFolders[time]

    if train == 0:
        ckptSubFolders = natural_sort(glob.glob(subFolder +'/ckpt_' + ckptName + '_'+  '*/'))[::-1]
        ckptvideoPath = ckptSubFolders[ckptTime]

        print 'you will assign identities from the last checkpoint in ', ckptvideoPath
    elif train == 1:
        ts = '_{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        ckptName = ckptName + ts
        ckptvideoPath = subFolder + '/ckpt_' + ckptName
        print 'model checkpoints will be saved in ', ckptvideoPath

    elif train == 2:
        print subFolder +'ckpt_' + ckptName + '_'+ '*/'
        ckptSubFolders = natural_sort(glob.glob(subFolder +'/ckpt_' + ckptName + '_' + '*/'))[::-1]
        print ckptSubFolders
        ckptvideoPath = ckptSubFolders[ckptTime]
        print 'you will keep training from the last checkpoint in ', ckptvideoPath

    return ckptvideoPath

def fineTuner(videoPath, trainDict, indivFragsForTrain, fragmentsDict = [], portraits = [], videoInfo = []):
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
    indivFragsForTrain: list of fragments indices that should be used for the fine-tunning
    fragmentsDict:
    portraits:
    """
    print '--------------------------------'
    print 'Loading fragments, portraits and videoInfo for the tracking...'
    if fragmentsDict == []:
        fragmentsDict = loadFile(videoPath, 'fragments', time=0, hdfpkl = 'pkl')
        fragmentsDict = fragmentsDict.to_dict()[0]
    if len(portraits) == 0:
        portraits = loadFile(videoPath, 'portraits', time=0)
    if videoInfo == []:
        videoInfo = loadFile(videoPath, 'videoInfo', time=0)

    numFrames =  len(portraits)
    numAnimals = int(videoInfo['numAnimals'])
    maxNumBlobs = videoInfo['maxNumBlobs']

    loadCkpt_folder = trainDict['loadCkpt_folder']
    ckptName = trainDict['ckptName']
    batchSize = trainDict['batchSize']
    numEpochs =  trainDict['numEpochs']
    lr = trainDict['lr']
    train = trainDict['train']

    if train == 1 or train == 2:
        ckpt_dir = getCkptvideoPath(videoPath,ckptName,train,time=0,ckptTime=0)
        imsize,\
        X_train, Y_train,\
        X_val, Y_val = DataFineTuning(indivFragsForTrain, fragmentsDict, portraits,numAnimals)

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
                        ckpt_dir, loadCkpt_folder, batchSize,numEpochs,
                        Tindices, Titer_per_epoch,
                        Vindices, Viter_per_epoch,
                        1.,lr) #dropout
    # return lossAccDict
# def idAssigner(videoPath,ckptName,batchSize,indivValAcc):
def idAssigner(videoPath,trainDict,fragmentsDict = [],portraits = [], videoInfo = []):
    '''
    videoPath: path to the video to which we want ot assign identities
    ckptName: name of the checkpoint folder with the model for the assignation of identities
    '''
    if len(portraits) == 0:
        portraits = loadFile(videoPath, 'portraits', time=0)

    if len(videoInfo) == 0:
        videoInfo = loadFile(videoPath, 'videoInfo', time=0)
    numFrames =  len(portraits)
    numAnimals = videoInfo['numAnimals']
    maxNumBlobs = videoInfo['maxNumBlobs']
    meanIndivArea = videoInfo['meanIndivArea']
    stdIndivArea = videoInfo['stdIndivArea']

    if len(fragmentsDict) == 0:
        fragmentsDict = loadFile(videoPath, 'fragments', time=0, hdfpkl='pkl')
    oneIndivFragFrames = fragmentsDict['oneIndivFragFrames']
    oneIndivFragIntervals = fragmentsDict['oneIndivFragIntervals']
    oneIndivFragSumLens = fragmentsDict['oneIndivFragSumLens']
    oneIndivFragLens = fragmentsDict['oneIndivFragLens']
    oneIndivFragVels = fragmentsDict['oneIndivFragVels']
    oneIndivFragDists = fragmentsDict['oneIndivFragDists']

    lensIntervalsLists = [len(oneIndivFragInterval) for oneIndivFragInterval in oneIndivFragIntervals]
    numGoodLists = np.sum(np.asarray(lensIntervalsLists) != 0)

    ckptName = trainDict['ckptName']
    batchSize = 1000
    ckpt_dir = getCkptvideoPath(videoPath,ckptName,train=0,time=0,ckptTime=0)

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
        print 'Computing softMax probabilities, id-frequencies, and P1 for blob index ', i
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
        velFragment = oneIndivFragVels[i]
        distFragment = oneIndivFragDists[i]
        freqFrags = freqFragsAll[i]
        softMaxProbs = softMaxProbsAll[i]
        softMaxId = softMaxIdAll[i]

        logP2FragsForMat = []
        P1FragsForMat = []
        logP2FragIdForMat = []

        blobsIndices = list(range(numGoodLists))
        blobsIndices.pop(i)

        logP2FragsForMat, logP2FragIdForMat, P2FragsForMat, P2Frags = computeLogP2Complete(oneIndivFragIntervals, P1FragsAll, indivFragmentsIntervals, P1Frags, lenFragments, velFragment, distFragment, blobsIndices,i)
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

    overallP2 = computeOverallP2(P2FragsAll,oneIndivFragLens)
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
        'P2FragAllVideo':P2FragAllVideo, # by single frames
        'P2FragsAll': P2FragsAll, # organized by individual fragments
        'overallP2': overallP2}

    portraits['identities'] = idFreqFragAllVideo.tolist()
    # saveFile(videoPath,portraits,'portraits',time=0)
    saveFile(videoPath, IdsStatistics, 'statistics', time = 0,hdfpkl='pkl')
    return normFreqFragsAll, portraits, P2FragsAll


def popOut(indivFragsForTrainList, idP2Frags, identity):
    """
    This functions pop-out fragments already used during training to avoid to repeat
    the accumulation of the same fragment twice.
    REMARK: we are using blobindex and frag number to pop out since it could be that
            an identity has changed during assignation...
    """

    candidates = []
    print 'number of fragments for identity ', identity, ', ', len(idP2Frags)
    for idfrag in idP2Frags: # all the fragments of an identity
        for frag in indivFragsForTrainList: # all
            if frag[2] != idfrag[2] or frag[3] != idfrag[3]:
                canAppend = True
            else:
                canAppend = False
                print '----------------------------------------------'
                print 'I am pooping (out)', idfrag
                break
        if canAppend:
            candidates.append(idfrag)
    print 'number of fragments for identity ', identity, ' after popping out', len(candidates)

    return candidates, identity

def chooser(candidates, minDist, identity, chosens = 1):
    sortedCandidates = sorted(candidates, key=lambda x: (x[1], x[-1]), reverse=True)
    counter = 0
    trainFrags = []
    continueFlag = True
    while len(trainFrags) != chosens and minDist > 0:
        for cur_cand in sortedCandidates:
            if cur_cand[-1] >= minDist:
                trainFrags.append(cur_cand)
                if len(trainFrags) == chosens:
                    break

        if len(trainFrags) < chosens:
            minDist -= 100

    if len(trainFrags) < chosens and minDist <= 0:
        continueFlag = False

    return trainFrags, identity, minDist, continueFlag

def bestFragmentFinder(indivFragsForTrain,normFreqFragsAll,fragmentsDict,numAnimals,P2FragsAll,minDists, badFragments):

    """
    1. Create a dictionary (idDict) organised as follow
        i. keys are the identities
        ii. values are lists of the form [(blobIndex, fragmentNumber, length, P2, start, end ), (...),... ]
            where each tuple belong to one individual fragment. blobIndex and
            fragmentNumber are used to access the images belonging to the fragment
            this information is stored in oneIndivFragFrames (in fragmentsDict); length, P2, start and end
            are used to choose the best candidate for propagation among the individual fragments assigned
            to a certain identity (key of the dictionary)
    2. Create candidate dict. Always organised by identity it contains all the individual fragments that
       will be evaluated in the iterative accumulation process.
    """
    print '----------------------------'
    print 'Entering the bestFragmentFinder'
    ids = range(numAnimals)
    P2FragsAll = flatten(P2FragsAll)
    indivFragsForTrainList = flatten([v for v in indivFragsForTrain.values()])
    print '----------------------->8'
    print 'indivFragsForTrainList',
    pprint(indivFragsForTrainList)
    idDict = groupByCustom(P2FragsAll, ids, 0)
    print '----------------------->8'
    for i in ids:
        print len(idDict[i]), 'fragments for identity ', i
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    output = Parallel(n_jobs=num_cores)(delayed(popOut)(indivFragsForTrainList, idDict[identity], identity) for identity in range(numAnimals))
    candidateDict = {}

    for (frags,identity) in output:
        candidateDict[identity] = frags

    # print '----------------------->8'
    # print 'candidates', len(candidateDict[0])

    output = Parallel(n_jobs=num_cores)(delayed(chooser)(candidateDict[identity], minDists[identity], identity) for identity in range(numAnimals))
    for (trainFrags,identity,minDist,continueFlag) in output:
        if not continueFlag:
            break
        indivFragsForTrain[identity].append(tuple(flatten(trainFrags)))
        minDists[identity] = minDist
    print '-------------------------------------'
    print 'indivFragsForTrain, ', indivFragsForTrain
    print '-------------------------------------'
    return indivFragsForTrain, continueFlag, minDists
