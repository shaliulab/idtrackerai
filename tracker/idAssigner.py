import sys
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/CNN')

from py_utils import *
from video_utils import *
from idTrainerTracker import *
from cnn_utils import *

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

def DataIdAssignation(portraits, indivFragments):
    portraitsFrag = np.asarray(portraits.loc[:,'images'].tolist())
    portsFragments = []
    # print 'indivFragments', indivFragments
    for indivFragment in indivFragments:
        # print 'indiv fragment, ', indivFragment
        portsFragment = []
        # print 'portsFragment, ', portsFragment
        for (frame, column) in indivFragment:
            portsFragment.append(portraitsFrag[frame][column])

        portsFragments.append(np.asarray(portsFragment))
    # print portsFragments
    images = np.vstack(portsFragments)
    images = np.expand_dims(images,axis=3)
    images = cropImages(images,32)
    imsize = (32,32,1)
    images = standarizeImages(images)
    return imsize, images


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

def fragmentProbId(X_t, width, height, channels, classes, loadCkpt_folder, batchSize, Tindices, Titer_per_epoch , keep_prob = 1.0, printFlag = True):
    with tf.Graph().as_default():

        images_pl = tf.placeholder(tf.float32, [None, width, height, channels], name = 'images')
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
                if printFlag:
                    print '********************************************************'
                    print 'We are also loading the softmax'
                    print '********************************************************'
                    print 'loading weigths from ' + loadCkpt_folder + '/model'
                    print 'loading softmax from ' + loadCkpt_folder + '/softmax'

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
    minFloat = float_info[3]
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
            P1Frag[P1Frag == 0.] = minFloat

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
        logCoexisting = np.log(1. - P1CoexistingFrags)
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

def idUpdater(ids,indivFragments,numFrames,maxNumBlobs):
    IdsArray = np.zeros((numFrames,maxNumBlobs))

    for (Id,indivFragment) in zip(ids,indivFragments):
        frames = np.asarray(indivFragment)[:,0]
        columns = np.asarray(indivFragment)[:,1]
        IdsArray[frames,columns] = Id[0]

    return IdsArray

def probsUptader(vectorPerFrame,indivFragments,numFrames,maxNumBlobs,numAnimals):
    ProbsArray = np.zeros((numFrames,maxNumBlobs,numAnimals))

    for (vectorPerFram,indivFragment) in zip(vectorPerFrame,indivFragments):
        frames = np.asarray(indivFragment)[:,0]
        columns = np.asarray(indivFragment)[:,1]
        ProbsArray[frames,columns,:] = vectorPerFram

    return ProbsArray

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

def idAssigner(videoPath, trainDict, accumCounter, fragmentsDict = [],portraits = [], videoInfo = [], plotFlag = True, printFlag = True):
    '''
    videoPath: path to the video to which we want ot assign identities
    '''
    if len(portraits) == 0:
        portraits = loadFile(videoPath, 'portraits')

    if len(videoInfo) == 0:
        videoInfo = loadFile(videoPath, 'videoInfo', hdfpkl='pkl')

    if printFlag:
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

    batchSize = 1000
    ckpt_dir = getCkptvideoPath(videoPath,accumCounter,train=0)


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
        if printFlag:
            print '\n******************************************************'
            print 'Computing softMax probabilities, id-frequencies, and P1 for list of fragments ', i
        # Load data for the assignment (### TODO this can be done in portraits so that we only need to do it once)
        if len(indivFragments) != 0:
            imsize, portsFragments =  DataIdAssignation(portraits, indivFragments)

            if printFlag:
                print '\n values of the images during identity assigation: max min'
                print np.max(portsFragments), np.min(portsFragments)

            # Set variables for the forward pass
            loadCkpt_folder = ckpt_dir
            width, height, channels = imsize
            classes = numAnimals
            numImagesT = len(portsFragments)
            # Get batch indices
            Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batchSize)

            if printFlag:
                print 'indices test ', Tindices
                print 'iteration per epoch ', Titer_per_epoch

            # Run forward pass,
            softMaxProbs, softMaxId = fragmentProbId(portsFragments, width, height, channels,
                classes, loadCkpt_folder, batchSize, Tindices, Titer_per_epoch)

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

    if printFlag:
        print '******************************************************'

    for i in range(numGoodLists):
        # print '******************************************************'
        if printFlag:
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

    sessionPath = '/'.join(ckpt_dir.split('/')[:-1])
    overallP2 = computeOverallP2(P2FragsAll,oneIndivFragLens)

    if printFlag:
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
        'normFreqFragsAll':normFreqFragsAll,
        'P2FragsAll': P2FragsAll,
        'overallP2': overallP2}

    # portraits['identities'] = idLogP2FragAllVideo.tolist()
    # saveFile(videoPath,portraits,'portraits')

    # pickle.dump( IdsStatistics , open( ckpt_dir + "/statistics.pkl", "wb" ) )
    pickle.dump( IdsStatistics , open( sessionPath + "/statistics.pkl", "wb" ) )

    return IdsStatistics #, portraits
