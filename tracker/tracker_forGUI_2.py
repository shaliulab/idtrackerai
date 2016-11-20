import sys
sys.path.append('../utils')
sys.path.append('../CNN')

from py_utils import *
from video_utils import *
from idTrainer import *
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

def DataFirstFineTuning(longestFrag, portraits, numAnimals):
    portraitsFrag = np.asarray(flatten(portraits.loc[longestFrag[0]:longestFrag[1],'images'].tolist()))
    portraitsFrag = np.expand_dims(portraitsFrag,axis=1)
    identities = portraits.loc[longestFrag[0]:longestFrag[1],'permutations'].tolist()
    # print longestFrag[0], longestFrag[1]

    identities = [np.delete(identity, np.where(identity == -1)) for identity in identities]
    # print 'numLabels, ', len(identities)
    # for identity in identities:
    #     print len(identity) == 5
    portDims = portraitsFrag.shape
    imsize = (portDims[1],portDims[2], portDims[3])
    # portraitsFrag = np.reshape(portraitsFrag, [portDims[0]*portDims[1],1,portDims[2], portDims[3]])
    labels = flatten(identities)
    labels = dense_to_one_hot(labels, numAnimals)
    numImages = len(labels)

    # print 'numLabels', len(labels)
    # print 'numImages', portraitsFrag.shape

    perm = np.random.permutation(numImages)
    portraitsFrag = portraitsFrag[perm]
    labels = labels[perm]
    #split train and validation
    numTrain = np.ceil(np.true_divide(numImages,10)*8).astype('int')
    X_train = portraitsFrag[:numTrain]
    Y_train = labels[:numTrain]
    X_val = portraitsFrag[numTrain:]
    Y_val = labels[numTrain:]

    resolution = np.prod(imsize)
    X_train = np.reshape(X_train, [numTrain, resolution])
    X_val = np.reshape(X_val, [numImages - numTrain, resolution])

    return imsize, X_train, Y_train, X_val, Y_val

def DataIdAssignation(portraits, animalInd,meanIndivArea,stdIndivArea):
    portraitsFrag = np.asarray(portraits.loc[:,'images'].tolist())
    newPortraitsDf = portraits.copy()
    areasFrag = np.asarray(portraits.loc[:,'areas'].tolist())
    identities = np.asarray(portraits.loc[:,'permutations'].tolist())
    identitiesInd = np.where(identities==animalInd)
    frames = identitiesInd[0]
    portraitInds = identitiesInd[1]
    potentialCrossings = []
    if frames.size != 0: # there are frames assigned to this individualIndex
        # _, height, width =  portraitsFrag[0].shape
        height = 32
        width = 32
        imsize = (1, height, width)
        indivFragments = []
        indivFragment = []
        indivFragmentInterval = ()
        indivFragmentsIntervals = []
        portsFragments = []
        portsFragment = []
        lenFragments = []
        sumFragIndices = []
        for i, (frame, portraitInd) in enumerate(zip(frames, portraitInds)):

            if i != len(frames)-1: # if it is not the last frame
                if frames[i+1] - frame == 1 : # if the next frame is a consecutive frame
                    currentArea = areasFrag[frame][portraitInd]
                    if currentArea < meanIndivArea + 5*stdIndivArea: # if the area is accepted by the model area
                        indivFragment.append((frame, portraitInd))
                        portsFragment.append(portraitsFrag[frame][portraitInd])

                        if len(indivFragmentInterval) == 0: # is the first frame we append to the interval
                            indivFragmentInterval = indivFragmentInterval + (frame,) # we are using tuples

                    else: # if the area is too big, I close the individual fragment and I add the indices to the list of potentialCrossings
                        potentialCrossings.append((frame,portraitInd)) # save to list of potential crossings
                        newIdentitiesFrame = portraits.loc[frame,'permutations']
                        newIdentitiesFrame[portraitInd] = -1
                        newPortraitsDf.set_value(frame, 'permutations', newIdentitiesFrame)

                        if len(indivFragment) != 0:
                            # I close the individual fragment and I open a new one
                            indivFragment = np.asarray(indivFragment)
                            indivFragments.append(indivFragment)

                            indivFragmentInterval = indivFragmentInterval + (frame-1,)
                            indivFragmentsIntervals.append(indivFragmentInterval)


                            # print len(indivFragment)
                            portsFragments.append(np.reshape(np.asarray(portsFragment), [len(indivFragment),height*width]))
                            lenFragments.append(len(indivFragment))
                            sumFragIndices.append(sum(lenFragments))
                            indivFragment = []
                            indivFragmentInterval = ()
                            portsFragment = []
                else: #the next frame is not a consecutive frame, I close the individual fragment
                    # print 'they are not consecutive frames, I close the fragment'
                    currentArea = areasFrag[frame][portraitInd]
                    if currentArea < meanIndivArea + 5*stdIndivArea: # if the area is accepted by the model area
                        indivFragment.append((frame, portraitInd))
                        portsFragment.append(portraitsFrag[frame][portraitInd])

                        if len(indivFragmentInterval) == 0: # is the first frame we append to the interval
                            indivFragmentInterval = indivFragmentInterval + (frame,) # we are using tuples

                    else: # if the area is too big, I close the individual fragment and I add the indices to the list of potentialCrossings
                        potentialCrossings.append((frame,portraitInd)) # save to list of potential crossings
                        newIdentitiesFrame = portraits.loc[frame,'permutations']
                        newIdentitiesFrame[portraitInd] = -1
                        newPortraitsDf.set_value(frame, 'permutations', newIdentitiesFrame)

                    if len(indivFragment) != 0:
                        indivFragment = np.asarray(indivFragment)
                        indivFragments.append(indivFragment)
                        indivFragmentInterval = indivFragmentInterval + (frame,)
                        indivFragmentsIntervals.append(indivFragmentInterval)
                        portsFragments.append(np.reshape(np.asarray(portsFragment), [len(indivFragment),height*width]))
                        lenFragments.append(len(indivFragment))
                        sumFragIndices.append(sum(lenFragments))
                        indivFragment = []
                        indivFragmentInterval = ()
                        portsFragment = []
            else:
                # print 'it is the last frame, I close the fragments '
                currentArea = areasFrag[frame][portraitInd]
                if currentArea < meanIndivArea + 5*stdIndivArea: # if the area is accepted by the model area
                    indivFragment.append((frame, portraitInd))
                    portsFragment.append(portraitsFrag[frame][portraitInd])

                    if len(indivFragmentInterval) == 0: # is the first frame we append to the interval
                        indivFragmentInterval = indivFragmentInterval + (frame,) # we are using tuples

                else: # if the area is too big, I close the individual fragment and I add the indices to the list of potentialCrossings
                    potentialCrossings.append((frame,portraitInd)) # save to list of potential crossings
                    newIdentitiesFrame = portraits.loc[frame,'permutations']
                    newIdentitiesFrame[portraitInd] = -1
                    newPortraitsDf.set_value(frame, 'permutations', newIdentitiesFrame)


                # I close the individual fragment and I open a new one
                if len(indivFragment) != 0:
                    indivFragment = np.asarray(indivFragment)
                    indivFragments.append(indivFragment)
                    indivFragmentInterval = indivFragmentInterval + (frame,)
                    indivFragmentsIntervals.append(indivFragmentInterval)
                    portsFragments.append(np.reshape(np.asarray(portsFragment), [len(indivFragment),height*width]))
                    lenFragments.append(len(indivFragment))

        return imsize, np.vstack(portsFragments), indivFragments,indivFragmentsIntervals, lenFragments,sumFragIndices, newPortraitsDf
    else:
        return None, None, None, None, None, None,  portraits



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

def fragmentProbId(X_t, width, height, channels, classes, resolution, loadCkpt_folder, batch_size, Tindices, Titer_per_epoch , keep_prob = 1.0):
    with tf.Graph().as_default():

        images_pl = tf.placeholder(tf.float32, [None, resolution], name = 'images')
        keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')

        logits, relu = inference1(images_pl, width, height, channels, classes, keep_prob_pl)
        predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)

        saver_model = createSaver('soft', False, 'saver_model')
        saver_softmax = createSaver('soft', True, 'saver_softmax')
        with tf.Session() as sess:
            # you need to initialize all variables
            tf.initialize_all_variables().run()

            print "\n****** Starting analysing fragment ******\n"

            # Load weights from a pretrained model if there is not any model saved
            # in the ckpt folder of the test
            if loadCkpt_folder:
                print 'loading weigths from ' + loadCkpt_folder
                restoreFromFolder(loadCkpt_folder + '/model', saver_model, sess)
                restoreFromFolder(loadCkpt_folder + '/softmax', saver_softmax, sess)

            else:
                warnings.warn('It is not possible to perform knowledge transfer, give a folder containing a trained model')

            # print "Start from:", start
            opListProb = [predictions, logits, relu]

            ''' Getting idProb fragment '''
            probsFrames = []
            reluFrames = []
            identities = []
            logitsFrames = []
            allPredictions = []
            allProbs = []
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

                allProbs.append(probs)
                allPredictions.append(predictions)
                # print '--------------------------------------'
                # print predictions
                # print '--------------------------------------'
            allPredictions = flatten(allPredictions)
            allProbs = flatten(allProbs)
            # allPredictions  = countRateSet(allPredictions)
            # allPredictions = sorted(allPredictions, key=lambda x: x[1],reverse=True)
            # allPredictions = np.asarray(allPredictions).astype(int)
            # allPredictions = allPredictions[:,0]
                # exp = np.exp(logits)
                # sumsExps = np.sum(exp,axis=1)
                # probs = []
                # for i, sumExps in enumerate(sumsExps):
                #     probs.append(np.true_divide(exp[i,:],sumExps))
                #
                # probs = np.asarray(probs)
                # probs[probs < 0.] = 0
                # # probs = np.around(probs,decimals=2)
                # ids = np.argmax(probs,axis=1)
                #
                # zeros = np.all(probs ==0, axis=1)
                # ids[zeros] = -1
                # allPredictions.append(predictions)
                # probsFrames.append(probs)
                # reluFrames.append(relu)
                # logitsFrames.append(logits)
                # identities.append(ids)
    return np.asarray(allProbs), np.asarray(allPredictions).astype('int')
    # return np.asarray(allPredictions), probsFrames, reluFrames, identities, logitsFrames

def computeP1(IdProbs):
    """
    IdProbs: list of IdProb, [IdProb1, IdProb2]
    IdProb: are the probabilities of assignment for each frame of the individual fragment (the output of the softmax for each portrait of the individual fragment)
    IdProb.shape: (num frames in fragment, maxNumBlobs, numAnimals)
    """
    probs1Fragments = []
    frequenciesFragments = []
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
        print '---------------------------'
        print 'idsFragment, ', idsFragment
        fragLen = IdProb.shape[0]
        print 'numAnimals, ', numAnimals
        frequencies = np.asarray([np.sum(idsFragment == i) for i in range(numAnimals)]) # counting the frequency of every assignment
        print 'frequencies, ', frequencies
        numerator = 2.**frequencies
        denominator = np.sum(numerator)
        probs1Fragment = numerator / denominator
        print 'probs1Fragment, ', probs1Fragment
        if np.any(probs1Fragment == 0.):
            raise ValueError('probs1Fragment cannot be 0')
        # probs1Fragment[probs1Fragment == 1.] = 1. - np.sum(probs1Fragment[probs1Fragment!=1.])
        probs1Fragment[probs1Fragment == 1.] = 0.9999
        # print np.sum(probs1Fragment[probs1Fragment!=1.])
        print 'probs1Fragment, ', probs1Fragment
        if np.any(probs1Fragment == 1.):
            raise ValueError('probs1Fragment cannot be 1')
        probs1Fragments.append(probs1Fragment)
        frequenciesFragments.append(np.matlib.repmat(frequencies,fragLen,1))
        # frequenciesFragments.append(frequencies)

    return probs1Fragments, frequenciesFragments

# def idFragment(IdProbs,indivValAcc):
#     """
#     IdProbs: list of IdProb, [IdProb1, IdProb2]
#     IdProb: are the probabilities of assignment for each frame of the individual fragment (the output of the softmax for each portrait of the individual fragment)
#     IdProb.shape: (num frames in fragment, maxNumBlobs, numAnimals)
#     """
#     fragmentsIds = []
#     probFragmentsIds = []
#
#     for IdProb in IdProbs: #looping on the individual fragments
#         print '******* Computing fragment ID and Prob *******'
#         print 'shape of IdProb, ', IdProb.shape
#         epsilon = 0.01
#         epMat = np.zeros_like(IdProb)
#         epMat[IdProb >= .99] = epsilon
#
#         # # Sum of the log of the probability of assignment
#         # # log(Q_j) = Sum_frames(log(p_j)) where p_j is the probability of assignment for individual j in a single frame
#         # # j_assigned = argmax(Q_j)
#         # sumLogIdProb = np.sum(np.log(IdProb),axis=0)
#         # fragmentsId = np.argmax(sumLogIdProb)+1
#         # probFragmentsId = sumLogIdProb
#
#         # IdTracker way assuming that the indivValAcc is the probability of good assignment for each identity
#         idsFragment = np.argmax(IdProb,axis=1) # assignment of identities for single frames according to the output of the softmax
#         # print 'idsFragment, ', idsFragment
#         numAnimals = IdProb.shape[1]
#         indivValAcc = np.ones(numAnimals)*.66
#         # print 'numAnimals, ', numAnimals
#         numFramesFragment = IdProb.shape[0]
#         frequencies = np.asarray([np.sum(idsFragment == i) for i in range(numAnimals)]) # counting the frequency of every assignment
#         # print 'frequencies, ', frequencies
#         idFound = np.where(frequencies!=0)[0]
#         # print 'idFound', idFound
#         P_frag_k = indivValAcc[idFound]**frequencies[idFound] * (1-indivValAcc[idFound])**(numFramesFragment-frequencies[idFound])
#         # print 'P_frag_k', P_frag_k
#         probId = P_frag_k / np.sum(P_frag_k)
#         probFragmentsId = np.zeros(numAnimals)
#         probFragmentsId[idFound] = probId
#         # print 'probFragmentsId, ', probFragmentsId
#         fragmentsId = np.argmax(probFragmentsId)+1
#         # print 'fragmentsId, ', fragmentsId
#
#
#
#
#
#         # Average probability
#         # <P_j> = Mean_frames(p_j) where p_j is the probability of assignment for individual j in a single frame
#         # probFragmentsId = np.mean(IdProb, axis=0)
#         # fragmentsId = np.argmax(probFragmentsId)+1
#
#         fragLen = IdProb.shape[0]
#         probFragmentsIds.append(np.matlib.repmat(probFragmentsId,fragLen,1))
#         fragmentsIds.append(np.multiply(np.ones(fragLen),fragmentsId).astype('int'))
#
#     return fragmentsIds, probFragmentsIds

def idUpdater(Ids,IdsArray):

    for Id in Ids:
        currId = Id[0]
        frames = np.asarray(Id[1])[:,0]
        indices = np.asarray(Id[1])[:,1]
        IdsArray[frames,indices] = currId

    return IdsArray

def idProbsUpdated(idProbs,ProbsArray):

    for idProb in idProbs:
        frames = np.asarray(idProb[1])[:,0]
        indices = np.asarray(idProb[1])[:,1]
        ProbsArray[frames,indices,:] = idProb[0]

    return ProbsArray

# def createFolder(videoPath, name = '', timestamp = False):
#
#     ts = '_{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
#     name = name + ts
#
#     folder = os.videoPath.dirname(videoPath)
#     folderName = folder +'/'+ name + '/segmentation'
#     os.makedirs(folderName) # create a folder
#
#     # folderName = folderName
#     # os.makedirs(folderName) # create a folder
#
#     print folderName + ' has been created'
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

def fineTuner(videoPath,ckptName,loadCkpt_folder,batch_size,num_epochs,lr,train):
    """
    videoPath: path to the video that we want to tracker
    ckptName: name of the checkpoint folder where we will save the model used to infer the identities
    loadCkpt_folder: folder where to load the master model that we will fine tune with the new references
    batch_size: bath size used for the training of the fine tuning
    num_epochs: number of epochs for the fine tuning
    lr: learning rate for the optimization algorithim for the fine tuning
    train:  train = 0 (id assignation)
            train = 1 (first fine-tuning)
            train = 2 (further tuning from previons checkpoint with more references) no implemented yet, now it keeps training from the previous checkpoint with the same references
    """
    print '--------------------------------'
    print 'Loading fragments, portraits and videoInfo for the tracking...'
    fragments = loadFile(videoPath, 'fragments', time=0)
    fragments = np.asarray(fragments)
    portraits = loadFile(videoPath, 'portraits', time=0)
    videoInfo = loadFile(videoPath, 'videoInfo', time=0)
    info = videoInfo.to_dict()[0]
    numFrames =  len(portraits)
    numAnimals = info['numAnimals']
    maxNumBlobs = info['maxNumBlobs']

    ''' ************************************************************************
    Fine tuning with the longest fragment
    ************************************************************************ '''
    if train == 1 or train == 2:
        print 'Fine tuning with the first longest fragment...'
        ckpt_dir = getCkptvideoPath(videoPath,ckptName,train,time=0,ckptTime=0)
        imsize,\
        X_train, Y_train,\
        X_val, Y_val = DataFirstFineTuning(fragments[0], portraits,numAnimals)

        print '\n fine tune train size:    images  labels'
        print X_train.shape, Y_train.shape
        print 'val fine tune size:    images  labels'
        print X_val.shape, Y_val.shape

        channels, width, height = imsize
        resolution = np.prod(imsize)
        classes = numAnimals

        numImagesT = Y_train.shape[0]
        numImagesV = Y_val.shape[0]
        Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batch_size)
        Vindices, Viter_per_epoch = get_batch_indices(numImagesV,batch_size)

        print 'running with the devil'
        lossAccDict = run_training(X_train, Y_train, X_val, Y_val,
                        width, height, channels, classes, resolution,
                        ckpt_dir, loadCkpt_folder, batch_size,num_epochs,
                        Tindices, Titer_per_epoch,
                        Vindices, Viter_per_epoch,
                        1.,lr) #dropout
    return lossAccDict
def idAssigner(videoPath,ckptName,batch_size,indivValAcc):
    '''
    videoPath: path to the video to which we want ot assign identities
    ckptName: name of the checkpoint folder with the model for the assignation of identities
    '''
    print '--------------------------------'
    print 'Loading portraits and videoInfo for the assignation of identities...'
    portraits = loadFile(videoPath, 'portraits', time=0)
    videoInfo = loadFile(videoPath, 'videoInfo', time=0)
    globalFragments = loadFile(videoPath, 'fragments', time=0)
    videoInfo = videoInfo.to_dict()[0]
    numFrames =  len(portraits)
    numAnimals = videoInfo['numAnimals']
    maxNumBlobs = videoInfo['maxNumBlobs']
    meanIndivArea = videoInfo['meanIndivArea']
    stdIndivArea = videoInfo['stdIndivArea']

    ckpt_dir = getCkptvideoPath(videoPath,ckptName,train=0,time=0,ckptTime=0)
    Ids = []
    AllIds = []
    idProbs = []
    AllIdProbs = []
    AllFragmentsIds = []
    AllProbFragmentsIds = []
    '''
    Loop to IndivFragments, Ids in frames of IndivFragments, P1 given the Ids
    '''
    indivFragmentsBlobId = []
    idProbsBlobId = []
    indivFragmentsIntervalsBlobId = []
    IdsBlobId = []
    probs1FragmentsBlobId = []
    allProbsBlobId = []
    allPredictionsBlobId = []
    lenFragmentsBlobId = []
    frequenciesFragmentsBlobId = []


    for i in range(maxNumBlobs):
        imsize, portsFragments, indivFragments,indivFragmentsIntervals,lenFragments, sumFragIndices, portraits =  DataIdAssignation(portraits, i,meanIndivArea,stdIndivArea)
        # print 'shape portsFragments, ', portsFragments.shape
        # print 'length Fragments, ', lenFragments
        # print 'length indivFragments, ', len(indivFragments)
        # print 'sumFragIndices', sumFragIndices
        if imsize != None:
            loadCkpt_folder = ckpt_dir
            channels, width, height = imsize
            resolution = np.prod(imsize)
            classes = numAnimals
            numImagesT = batch_size
            Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batch_size)

            # allPredictions, probsFrames, reluFrames, identities, logits = fragmentProbId(
            #     X_fragment, width, height, channels, classes, resolution, loadCkpt_folder, batch_size, Tindices, Titer_per_epoch)
            allProbs, allPredictions = fragmentProbId(portsFragments, width, height, channels,
                classes, resolution, loadCkpt_folder, batch_size, Tindices, Titer_per_epoch)
            # print allProbs.shape, allPredictions.shape
            allProbs = np.split(allProbs,sumFragIndices)
            # print 'lentgh allProbs', len(allProbs)
            # print 'length indivFragments', len(indivFragments)

            allPredictions = np.split(allPredictions,sumFragIndices)
            idProbs = zip(allProbs, indivFragments)
            # print idProbs
            Ids = zip(allPredictions, indivFragments)

            probs1Fragments, frequenciesFragments = computeP1(allProbs)
            # print 'probs1Fragments, ', probs1Fragments
            indivFragmentsIntervalsBlobId.append(indivFragmentsIntervals)
            probs1FragmentsBlobId.append(probs1Fragments)
            indivFragmentsBlobId.append(indivFragments)
            idProbsBlobId.append(idProbs)
            IdsBlobId.append(Ids)
            allProbsBlobId.append(allProbs)
            allPredictionsBlobId.append(allPredictions)
            lenFragmentsBlobId.append(lenFragments)
            frequenciesFragmentsBlobId.append(frequenciesFragments)


    '''
    Loop to Computer P2 for each individual fragment
    This loop can be parallelized
    '''
    def getOverlap(a, b):
        overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] == b[0] or a[0] == b[1]:
            overlap = 1
        if a[1] == a[0] and (b[0]<=a[0] and b[1]>=a[1]):
            overlap = 1
        if b[1] == b[0] and (a[0]<=b[0] and a[1]>=b[1]):
            overlap = 1
        return overlap

    IdsAssigned = -np.ones((numFrames,maxNumBlobs))
    ProbsAssigned = np.zeros((numFrames,maxNumBlobs,numAnimals))
    IdsFragAssigned= -np.ones((numFrames,maxNumBlobs))
    ProbsFragAssigned = np.zeros((numFrames,maxNumBlobs,numAnimals))
    FreqFrag = np.zeros((numFrames,maxNumBlobs,numAnimals))
    P1Frag = np.zeros((numFrames,maxNumBlobs,numAnimals))

    for i in range(len(indivFragmentsIntervalsBlobId)):
        print '******************************************************'
        print '\n Computing P2 for list of fragments, ', i
        indivFragmentsIntervals1 = indivFragmentsIntervalsBlobId[i]
        probs1Fragments = probs1FragmentsBlobId[i]
        indivFragments = indivFragmentsBlobId[i]
        idProbs = idProbsBlobId[i]
        Ids = IdsBlobId[i]
        allProbs = allProbsBlobId[i]
        allPredictions = allPredictionsBlobId[i]
        lenFragments = lenFragmentsBlobId[i]
        frequenciesFragments = frequenciesFragmentsBlobId[i]

        # print 'lisf of intervals, ', indivFragmentsIntervals1
        logP2Fragments = []
        P1FragmentsIds = []
        fragmentsIds = []
        blobsIndices = list(range(len(indivFragmentsIntervalsBlobId)))
        blobsIndices.pop(i)
        # print 'blobsIndices, ', blobsIndices

        ''' computing p2 for a fragment '''
        for j, indivFragmentInterval in enumerate(indivFragmentsIntervals1):
            print '\n fragment, ', j, ' ----------------'
            print 'Interval, ', indivFragmentInterval
            # print 'interval, ', indivFragmentInterval
            prob1Fragment = probs1Fragments[j]
            print 'current Probs, ', prob1Fragment
            probs1CoexistingFragments = []

            for k in blobsIndices:
                indivFragmentsIntervals2 = indivFragmentsIntervalsBlobId[k]
                print 'coexistence in fragments list ', k
                # print 'intervals of the list, ', indivFragmentsIntervals2
                probs1fragmentk = np.asarray(probs1FragmentsBlobId[k])
                overlaps = np.asarray([getOverlap(indivFragmentInterval,otherInterval) for otherInterval in indivFragmentsIntervals2])
                # print overlaps
                coexistingFragments = np.where(overlaps != 0)[0]
                print 'coexisting Fragments, ', coexistingFragments
                probs1CoexistingFragments.append(probs1fragmentk[coexistingFragments])

            probs1CoexistingFragments = np.vstack(probs1CoexistingFragments)
            print 'coexisting Probs, ', probs1CoexistingFragments

            def computeP2(prob1Fragment,probs1CoexistingFragments):
                # print 'prob1Fragment, ', prob1Fragment
                logProb1Fragment = np.log(prob1Fragment)
                print 'logProb1Fragment, ', logProb1Fragment
                # print 'probs1CoexistingFragments, ', probs1CoexistingFragments
                print 'logCoexisting, ',  np.log(1.-probs1CoexistingFragments)
                logCoexisting = np.log(1.-probs1CoexistingFragments)
                sumLogProbs1CF = np.sum(logCoexisting,axis=0)

                # if all(sumLogP1 == -np.inf for sumLogP1 in sumLogProbs1CF):
                #     print 'sumLogProbs1CF, ', sumLogProbs1CF
                #     sumLogProbs1CF = np.zeros_like(sumLogProbs1CF)
                # else:
                #     minLogCoexisting = np.min(logCoexisting[logCoexisting!=-np.inf])
                #     print 'min logCoexisting, ', minLogCoexisting
                #     logCoexisting[logCoexisting==-np.inf] = minLogCoexisting-1000
                #     print 'logCoexisting, ', logCoexisting
                #     sumLogProbs1CF = np.sum(logCoexisting,axis=0)
                #     print 'sumLogProbs1CF, ', sumLogProbs1CF

                LogP2Fragment = logProb1Fragment + sumLogProbs1CF
                # if np.any(LogP2Fragment == -np.inf):
                #     raise ValueError('is -inf')

                # a = prob1Fragment*np.prod((1-probs1CoexistingFragments),axis=0)
                # print 'a, ', a
                # b = np.sum(prob1Fragment*np.prod((1-probs1CoexistingFragments),axis=0))
                # print 'b, ', b
                # prob2Fragment = a/b
                # if np.isnan(prob2Fragment).any():
                #     raise ValueError('is nan')

                return LogP2Fragment


            logP2Fragment = computeP2(prob1Fragment,probs1CoexistingFragments)
            print 'logP2Fragment', logP2Fragment
            idFrag = np.argmax(logP2Fragment)+1
            print idFrag

            fragLen = lenFragments[j]
            logP2Fragments.append(np.matlib.repmat(logP2Fragment,fragLen,1))
            P1FragmentsIds.append(np.matlib.repmat(prob1Fragment,fragLen,1))
            fragmentsIds.append(np.multiply(np.ones(fragLen),idFrag).astype('int'))

            ''' finished computing the probabilities of assignment for a fragment '''

        # print fragmentsIds
        # print logP2Fragments
        # fragmentsIds, logP2Fragments = idFragment(allProbs,indivValAcc)
        # print fragmentsIds
        # print logP2Fragments
        fragmentsIds = zip(fragmentsIds, indivFragments)
        logP2Fragments = zip(logP2Fragments, indivFragments)
        P1FragmentsIds = zip(P1FragmentsIds, indivFragments)
        frequenciesFragments = zip(frequenciesFragments,indivFragments)

        ProbsArray = np.zeros((numFrames,maxNumBlobs,numAnimals))
        ProbsUpdated = idProbsUpdated(idProbs,ProbsArray)
        ProbsAssigned += ProbsUpdated

        IdsArray = np.zeros((numFrames,maxNumBlobs))
        IdsUpdated = idUpdater(Ids,IdsArray)
        IdsAssigned += IdsUpdated
        # print IdsAssigned[80:90]

        ProbsFragUpdated = idProbsUpdated(logP2Fragments,ProbsArray)
        ProbsFragAssigned += ProbsFragUpdated

        IdsFragUpdated = idUpdater(fragmentsIds,IdsArray)
        IdsFragAssigned += IdsFragUpdated

        FreqFragUpdated = idProbsUpdated(frequenciesFragments,ProbsArray) # it works the same for frequencies
        FreqFrag += FreqFragUpdated

        P1FragUpdated = idProbsUpdated(P1FragmentsIds,ProbsArray) # it works the same for frequencies
        P1Frag += P1FragUpdated


        #     # print 'new fragmentsIds, ', fragmentsIds
        #     # print 'new probFragmentsIds', probFragmentsIds
        #
        # # fragmentsIds, probFragmentsIds = idFragment(allProbs,indivValAcc)
        # # print 'old fragmentsIds, ', fragmentsIds
        # # print 'old probFragmentsIds', probFragmentsIds
        # fragmentsIds = zip(fragmentsIds, indivFragments)
        # probFragmentsIds = zip(probFragmentsIds, indivFragments)
        #
        # ProbsArray = np.zeros((numFrames,maxNumBlobs,numAnimals))
        # ProbsUpdated = idProbsUpdated(idProbs,ProbsArray)
        # ProbsAssigned += ProbsUpdated
        #
        # IdsArray = np.zeros((numFrames,maxNumBlobs))
        # IdsUpdated = idUpdater(Ids,IdsArray)
        # IdsAssigned += IdsUpdated
        # # print IdsAssigned[80:90]
        #
        # ProbsFragUpdated = idProbsUpdated(probFragmentsIds,ProbsArray)
        # ProbsFragAssigned += ProbsFragUpdated
        #
        # IdsFragUpdated = idUpdater(fragmentsIds,IdsArray)
        # IdsFragAssigned += IdsFragUpdated
        # print IdsFragAssigned[80:90]
    # print '********************************************************************'
    # print portraits.loc[80:90,'permutations']
    IdsAssigned = IdsAssigned.astype('int')
    # print IdsAssigned[80:90]
    IdsFragAssigned = IdsFragAssigned.astype('int')
    # print IdsFragAssigned[80:90]
    IdsStatistics = {'blobIds':IdsAssigned,
        'probBlobIds':ProbsAssigned,
        'fragmentIds':IdsFragAssigned,
        'probFragmentIds':ProbsFragAssigned,
        'FreqFrag': FreqFrag,
        'P1Frag': P1Frag}

    # bestNextFragment = computeNextFragment(globalFragments,ProbsFragAssigned)

    saveFile(videoPath, IdsStatistics, 'statistics', time = 0)


















# def computeNextFragment(globalFragments,ProbsFragAssigned):
#
#     for globalFragment in globalFragments:
#         globalFragmentLogProb = ProbsFragAssigned[globalFragment[0]+1,:5,:]
#         print globalFragmentLogProb
#
# videoPath = '../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi'
# globalFragments = loadFile(videoPath, 'fragments', time=0)
# globalFragments = np.asarray(globalFragments)
# IdsStatistics = loadFile(videoPath, 'statistics', time=0)
# IdsStatistics = IdsStatistics.to_dict()[0]
# ProbsFragAssigned = IdsStatistics['probFragmentIds']
#
# computeNextFragment(globalFragments,ProbsFragAssigned)
