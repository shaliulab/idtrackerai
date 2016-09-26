import sys
sys.path.append('../utils')
sys.path.append('../CNN')

from py_utils import *
from video_utils import *
from cnn_model_summaries import *
import time
import numpy as np
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

def DataFirstFineTuning(longestFrag, portraits, numAnimals):
    portraitsFrag = np.asarray(portraits.loc[longestFrag[0]:longestFrag[1],'images'].tolist())
    identities = portraits.loc[longestFrag[0]:longestFrag[1],'permutations'].tolist()
    # print identities
    identities = [np.delete(identity, np.where(identity == -1)) for identity in identities]

    portDims = portraitsFrag.shape
    imsize = (1,portDims[2], portDims[3])
    portraitsFrag = np.reshape(portraitsFrag, [portDims[0]*portDims[1],1,portDims[2], portDims[3]])
    labels = flatten(identities)
    labels = dense_to_one_hot(labels, numAnimals)
    numImages = len(labels)
    perm = np.random.permutation(numImages)
    portraitsFrag = portraitsFrag[perm]
    labels = labels[perm]
    #split train and validation
    numTrain = np.ceil(np.true_divide(numImages,10)*9).astype('int')
    X_train = portraitsFrag
    Y_train = labels
    X_val = portraitsFrag[numTrain:]
    Y_val = labels[numTrain:]

    resolution = np.prod(imsize)
    X_train = np.reshape(X_train, [numImages, resolution])
    X_val = np.reshape(X_val, [numImages - numTrain, resolution])

    return imsize, X_train, Y_train, X_val, Y_val


def DataNextFragment(portraits, animalInd):

    portraitsFrag = np.asarray(portraits.loc[:,'images'].tolist())
    identities = np.asarray(portraits.loc[:,'permutations'].tolist())
    identitiesInd = np.where(identities==animalInd)
    frames = identitiesInd[0]
    portraitInds = identitiesInd[1]
    # print portraitsFrag[0]
    # print portraitsFrag[0].shape

    # _, height, width =  portraitsFrag[0].shape
    height = 32
    width = 32
    imsize = (1, height, width)
    indivFragments = []
    indivFragment = []
    portsFragments = []
    portsFragment = []
    lenFragments = []
    sumFragIndices = []

    for i, (frame, portraitInd) in enumerate(zip(frames, portraitInds)):
        indivFragment.append((frame, portraitInd))
        portsFragment.append(portraitsFrag[frame][portraitInd])
        # print indivFragment

        if i != len(frames)-1:
            if frames[i+1] - frame != 1:
                indivFragment = np.asarray(indivFragment)
                indivFragments.append(indivFragment)
                # print np.asarray(portsFragment).shape
                # print height, width
                # print len(indivFragment)
                portsFragments.append(np.reshape(np.asarray(portsFragment), [len(indivFragment),height*width]))
                lenFragments.append(len(indivFragment))
                sumFragIndices.append(sum(lenFragments))
                indivFragment = []
                portsFragment = []
        else:
            indivFragments.append(indivFragment)
            portsFragments.append(np.reshape(np.asarray(portsFragment), [len(indivFragment),height*width]))
            lenFragments.append(len(indivFragment))
            # sumFragIndices.append(sum(lenFragments))
    return imsize, np.vstack(portsFragments), indivFragments, lenFragments,sumFragIndices


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

def idFragment(IdProbs):
    fragmentsIds = []
    probFragmentsIds = []

    for IdProb in IdProbs:
        epsilon = 0.01
        epMat = np.zeros_like(IdProb)
        # print epMat.shape
        epMat[IdProb >= .99] = epsilon
        # print epMat.shape
        # print IdProb-epMat
        # print np.prod(np.subtract(1., IdProb-epMat),axis = 0)
        # avProbFragmentsId = np.power(np.subtract(1.,np.mean(IdProb, axis=0)),IdProb.shape[0])
        # probFragmentsId = np.subtract(1., np.power(np.subtract(1.,np.mean(IdProb, axis=0)),IdProb.shape[0]))
        # probFragmentsId = np.subtract(1., np.prod(np.subtract(1., IdProb-epMat), axis = 0))
        probFragmentsId = np.mean(IdProb, axis=0)
        print 'probFragmentsId', probFragmentsId
        fragmentsId = np.argmax(probFragmentsId)
        # print 'fragments Id', fragmentsId

        fragLen = IdProb.shape[0]
        # print 'fragment lengths', fragLen
        probFragmentsIds.append(np.matlib.repmat(probFragmentsId,fragLen,1))
        fragmentsIds.append(np.multiply(np.ones(fragLen),fragmentsId).astype('int'))

    return fragmentsIds, probFragmentsIds

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

# def createFolder(path, name = '', timestamp = False):
#
#     ts = '_{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
#     name = name + ts
#
#     folder = os.path.dirname(path)
#     folderName = folder +'/'+ name + '/segmentation'
#     os.makedirs(folderName) # create a folder
#
#     # folderName = folderName
#     # os.makedirs(folderName) # create a folder
#
#     print folderName + ' has been created'
def getCkptPath(path,ckptName,train=0,time=0,ckptTime=0,):
    """
    train = 0 (id assignation)
    train = 1 (first fine-tuning)
    train = 2 (further tuning from previons checkpoint with more references)
    """

    video = os.path.basename(path)
    folder = os.path.dirname(path)
    filename, extension = os.path.splitext(video)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    subFolder = subFolders[time]

    if train == 0:
        ckptSubFolders = natural_sort(glob.glob(subFolder +'/ckpt_' + ckptName + '_'+  '*/'))[::-1]
        ckptPath = ckptSubFolders[ckptTime]

        print 'you will assign identities from the last checkpoint in ', ckptPath
    elif train == 1:
        ts = '_{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        ckptName = ckptName + ts
        ckptPath = subFolder + '/ckpt_' + ckptName
        print 'model checkpoints will be saved in ', ckptPath

    elif train == 2:
        print subFolder +'ckpt_' + ckptName + '_'+ '*/'
        ckptSubFolders = natural_sort(glob.glob(subFolder +'/ckpt_' + ckptName + '_' + '*/'))[::-1]
        print ckptSubFolders
        ckptPath = ckptSubFolders[ckptTime]
        print 'you will keep training from the last checkpoint in ', ckptPath

    return ckptPath

# if False:
if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = '', type = str)
    parser.add_argument('--ckpt_folder_name', default = "", type= str)
    parser.add_argument('--loadCkpt_folder', default = "../CNN/ckpt_Train_30indiv_36dpf_22000_transfer", type = str)
    parser.add_argument('--num_epochs', default = 50, type = int)
    parser.add_argument('--batch_size', default = 50, type = int)
    parser.add_argument('--learning_rate', default = 0.001, type= float)
    parser.add_argument('--train', default = 1, type = int)
    args = parser.parse_args()

    np.set_printoptions(precision=2)
    # read args
    path = args.path
    ckptName = args.ckpt_folder_name
    loadCkpt_folder = args.loadCkpt_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.learning_rate
    train = args.train
    print 'Loading stuff'
    fragments = loadFile(path, 'fragments', time=0)
    portraits = loadFile(path, 'portraits', time=0)
    videoInfo = loadFile(path, 'videoInfo', time=0)
    numFrames =  len(portraits)
    numAnimals = videoInfo['numAnimals']
    maxNumBlobs = videoInfo['maxNumBlobs']

    ''' Fine tuning with the longest fragment '''
    if train == 1 or train == 2:

        ckpt_dir = getCkptPath(path,ckptName,train,time=0,ckptTime=0)
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
        run_training(X_train, Y_train, X_val, Y_val,
            width, height, channels, classes, resolution,
            ckpt_dir, loadCkpt_folder, batch_size,num_epochs,
            Tindices, Titer_per_epoch,
            Vindices, Viter_per_epoch,
            1.,lr) #dropout

    ''' Loop to assign identities to '''
    if train == 0:
        ckpt_dir = getCkptPath(path,ckptName,train,time=0,ckptTime=0)
        Ids = []
        AllIds = []
        idProbs = []
        AllIdProbs = []
        AllFragmentsIds = []
        AllProbFragmentsIds = []
        IdsAssigned = np.zeros((numFrames,maxNumBlobs))
        ProbsAssigned = np.zeros((numFrames,maxNumBlobs,numAnimals))
        IdsFragAssigned= np.zeros((numFrames,maxNumBlobs))
        ProbsFragAssigned = np.zeros((numFrames,maxNumBlobs,numAnimals))
        for i in range(maxNumBlobs):
            imsize, portsFragments, indivFragments,lenFragments, sumFragIndices =  DataNextFragment(portraits, i)
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
            # print len(allProbs)
            allPredictions = np.split(allPredictions,sumFragIndices)
            idProbs = zip(allProbs, indivFragments)
            Ids = zip(allPredictions, indivFragments)

            fragmentsIds, probFragmentsIds = idFragment(allProbs)
            fragmentsIds = zip(fragmentsIds, indivFragments)
            probFragmentsIds = zip(probFragmentsIds, indivFragments)

            ProbsArray = np.zeros((numFrames,maxNumBlobs,numAnimals))
            ProbsUpdated = idProbsUpdated(idProbs,ProbsArray)
            ProbsAssigned += ProbsUpdated

            IdsArray = np.zeros((numFrames,maxNumBlobs))
            IdsUpdated = idUpdater(Ids,IdsArray)
            IdsAssigned += IdsUpdated

            ProbsFragUpdated = idProbsUpdated(probFragmentsIds,ProbsArray)
            ProbsFragAssigned += ProbsFragUpdated

            IdsFragUpdated = idUpdater(fragmentsIds,IdsArray)
            IdsFragAssigned += IdsFragUpdated

        IdsAssigned = IdsAssigned.astype('int')
        IdsFragAssigned = IdsFragAssigned.astype('int')
        IdsStatistics = {'blobIds':IdsAssigned,
            'probBlobIds':ProbsAssigned,
            'fragmentIds':IdsFragAssigned,
            'probFragmentIds':ProbsFragAssigned}

        saveFile(path, IdsStatistics, 'statistics', time = 0)
        # filename = filename.split('_')[0]
        # pickle.dump(IdsStatistics, open(folder +'/'+ filename + '_statistics' + '.pkl',"wb"))

        # allIdentities.loc[:] = IdsAssigned
        # for
        # allProbabilities
        # allIdentities.to_pickle(folder +'/'+ filename.split('_')[0] + '_identities_new2.pkl')

        #     ##############checker###################
        #     # np.set_printoptions(threshold = np.inf)
        #     # pprint(np.subtract(np.subtract(allPredictions,1).astype('int'),fragmentFramesId))
        #     #########################################
        #     predictions = []
        #     probsFramePerm = []
        #     identitiesPerm = []
        #     for i, probsFrame in enumerate(probsFrames):
        #         predictions.append(list(allPredictions[i][np.argsort(fragmentFramesId[i])]))
        #         probsFramePerm.append(probsFrame[np.argsort(fragmentFramesId[i]),:])
        #         identitiesPerm.append(list(identities[i][np.argsort(fragmentFramesId[i])]))
        #
        #     #silly count: how many frame for each animal (columns in the original identitiesPerm)
        #     # identitiesPerm = list(np.asarray(identitiesPerm).T)
        #     # print predictions
        #     predictionsT = list(np.asarray(predictions).T)
        #     counts = [Counter(traj) for traj in predictionsT]
        #     # pprint(counts)
        #     #
        #     probsArr = np.asarray(probsFramePerm)
        #     prob = np.true_divide(np.sum(probsArr,axis=0),fragment[1]-fragment[0])
        #     # pprint(np.around(prob, decimals = 2))
        #     newFragmentId = idAssigner(prob,numAnimals)
        #     print newFragmentId
        #     fragmentFramesId = np.asarray(fragmentFramesId)
        #     newFragmentFramesId = idFragmentUpdater(fragmentFramesId,newFragmentId,numAnimals)
        #     Ids.append(newFragmentFramesId)
        #     allIdentities.loc[fragment[0]:fragment[1]] = Ids[n]
        #     allIdentities.to_pickle(folder +'/'+ filename.split('_')[0] + '_identities.pkl')
