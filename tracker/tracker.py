import sys
sys.path.append('../utils')
sys.path.append('../CNN')

from py_utils import *
from video_utils import *
from cnn_model_summaries import *
import time
import numpy as np
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

def DataFirstFineTuning(longestFrag, portraits):
    portraitsFrag = np.asarray(portraits.loc[longestFrag[0]:longestFrag[1],'images'].tolist())
    identities = portraits.loc[longestFrag[0]:longestFrag[1],'permutations'].tolist()
    numAnimals = len(identities[0])
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

    return numAnimals, imsize, X_train, Y_train, X_val, Y_val

def DataNextFragment(frag, portraits):
    portraitsFrag = np.asarray(portraits.loc[frag[0]:frag[1],'images'].tolist())
    identities = portraits.loc[frag[0]:frag[1],'permutations'].tolist()
    numAnimals = len(identities[0])
    portDims = portraitsFrag.shape
    imsize = (1,portDims[2], portDims[3])
    portraitsFrag = np.reshape(portraitsFrag, [portDims[0]*portDims[1],1,portDims[2], portDims[3]])
    numImages = len(identities)*numAnimals

    X_train = portraitsFrag

    resolution = np.prod(imsize)
    X_train = np.reshape(X_train, [numImages, resolution])

    return numAnimals, imsize, X_train, identities

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
            for iter_i in range(Titer_per_epoch):

                predictions, logits, relu, feed_dict = run_batch(
                    sess, opListProb, Tindices, iter_i, Titer_per_epoch,
                    images_pl, keep_prob_pl,
                    X_t, keep_prob = keep_prob)
                # print '--------------------------------------'
                # print predictions
                # print '--------------------------------------'
                exp = np.exp(logits)
                sumsExps = np.sum(exp,axis=1)
                probs = []
                for i, sumExps in enumerate(sumsExps):
                    probs.append(np.true_divide(exp[i,:],sumExps))

                probs = np.asarray(probs)
                probs[probs < 0.] = 0
                # probs = np.around(probs,decimals=2)
                ids = np.argmax(probs,axis=1)

                zeros = np.all(probs ==0, axis=1)
                ids[zeros] = -1
                allPredictions.append(predictions)
                probsFrames.append(probs)
                reluFrames.append(relu)
                logitsFrames.append(logits)
                identities.append(ids)

    return np.asarray(allPredictions), probsFrames, reluFrames, identities, logitsFrames

def idAssigner(prob,numAnimals):
    """
    prob: matrix with the probabilities that each animal has each identity in the fragment
    """
    # print 'first prob ', prob
    ids = np.ones(numAnimals)*np.nan
    idsAssigned = 0
    while idsAssigned < numAnimals:
        (previousId,newId) = np.unravel_index(np.argmax(prob),prob.shape)
        # print previousId, newId
        ids[previousId] = newId
        prob[:,newId] = 0
        prob[previousId,:] = 0
        idsAssigned += 1
        # print idsAssigned
        # print prob
        # print 'array of identities ',ids
    return ids.astype('int')

def idFragmentUpdater(fragmentFramesId,newFragmentId,numAnimals):
    newFragmentFramesId = np.zeros_like(fragmentFramesId)
    for i in range(numAnimals):
        newFragmentFramesId[fragmentFramesId==i]=newFragmentId[i]

    return newFragmentFramesId

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = '', type = str)
    parser.add_argument('--ckpt_folder', default = "", type= str)
    parser.add_argument('--loadCkpt_folder', default = "../CNN/ckpt_Train_30indiv_36dpf_22000", type = str)
    parser.add_argument('--num_epochs', default = 50, type = int)
    parser.add_argument('--batch_size', default = 250, type = int)
    parser.add_argument('--train', default = 1, type = int)
    args = parser.parse_args()

    np.set_printoptions(precision=2)
    # read args
    path = args.path
    ckpt_dir = args.ckpt_folder
    loadCkpt_folder = args.loadCkpt_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train = args.train
    print 'Loading stuff'
    #load fragments and images
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    filename = folder +'/'+ filename.split('_')[0] + '_segments.pkl'
    #fragments are sorted according to their length
    fragments = pickle.load(open(filename, 'rb'))
    filename = filename.split('_')[0] + '_portraits.pkl'
    portraits = pd.read_pickle(filename)

    ''' Fine tuning with the longest fragment '''
    if train == 1:
        numAnimals, imsize,\
        X_train, Y_train,\
        X_val, Y_val = DataFirstFineTuning(fragments[0], portraits)

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
            .8) #dropout

    ''' Loop to assign identities to '''
    if train == 0:
        Ids = []
        allIdentities = pd.DataFrame(index=portraits.index,columns=range(5))
        for n, fragment in enumerate(fragments):
            # if n ==1 :
            print 'testing fragment',n

            loadCkpt_folder = ckpt_dir
            numAnimals, imsize, X_fragment, fragmentFramesId = DataNextFragment(fragment, portraits)

            channels, width, height = imsize
            resolution = np.prod(imsize)
            classes = numAnimals
            batch_size = classes

            numImagesT = len(fragmentFramesId)*numAnimals
            Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batch_size)


            allPredictions, probsFrames, reluFrames, identities, logits = fragmentProbId(
                X_fragment, width, height, channels, classes, resolution, loadCkpt_folder, batch_size, Tindices, Titer_per_epoch)

            ##############checker###################
            # np.set_printoptions(threshold = np.inf)
            # pprint(np.subtract(np.subtract(allPredictions,1).astype('int'),fragmentFramesId))
            #########################################
            predictions = []
            probsFramePerm = []
            identitiesPerm = []
            for i, probsFrame in enumerate(probsFrames):
                predictions.append(list(allPredictions[i][np.argsort(fragmentFramesId[i])]))
                probsFramePerm.append(probsFrame[np.argsort(fragmentFramesId[i]),:])
                identitiesPerm.append(list(identities[i][np.argsort(fragmentFramesId[i])]))

            #silly count: how many frame for each animal (columns in the original identitiesPerm)
            # identitiesPerm = list(np.asarray(identitiesPerm).T)
            # print predictions
            predictionsT = list(np.asarray(predictions).T)
            counts = [Counter(traj) for traj in predictionsT]
            # pprint(counts)
            #
            probsArr = np.asarray(probsFramePerm)
            prob = np.true_divide(np.sum(probsArr,axis=0),fragment[1]-fragment[0])
            # pprint(np.around(prob, decimals = 2))
            newFragmentId = idAssigner(prob,numAnimals)
            print newFragmentId
            fragmentFramesId = np.asarray(fragmentFramesId)
            newFragmentFramesId = idFragmentUpdater(fragmentFramesId,newFragmentId,numAnimals)
            Ids.append(newFragmentFramesId)
            allIdentities.loc[fragment[0]:fragment[1]] = Ids[n]
            allIdentities.to_pickle(folder +'/'+ filename.split('_')[0] + '_identities.pkl')
