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

def DataFirstFineTuning(fragments, portraits):
    longerFrag = fragments[0]
    portraitsFrag = np.asarray(portraits.loc[longerFrag[0]:longerFrag[1],'images'].tolist())
    identities = portraits.loc[longerFrag[0]:longerFrag[1],'permutations'].tolist()
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
    X_train = portraitsFrag[:numTrain]
    Y_train = labels[:numTrain]
    X_val = portraitsFrag[numTrain:]
    Y_val = labels[numTrain:]

    resolution = np.prod(imsize)
    X_train = np.reshape(X_train, [numTrain, resolution])
    X_val = np.reshape(X_val, [numImages - numTrain, resolution])

    return numAnimals, imsize, X_train, Y_train, X_val, Y_val

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default = '../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', type = str)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--loadCkpt_folder', default = "", type = str)
    parser.add_argument('--num_epochs', default = 50, type = int)
    parser.add_argument('--batch_size', default = 250, type = int)
    args = parser.parse_args()

    # read args
    path = args.path
    ckpt_dir = args.ckpt_folder
    loadCkpt_folder = args.loadCkpt_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    print 'Loading stuff'
    #load fragments and images
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    filename = folder +'/'+ filename.split('_')[0] + '_segments.pkl'
    #fragments are sorted according to their length
    fragments = pickle.load(open(filename, 'rb'))
    filename = folder +'/'+ filename.split('_')[0] + '_portraits.pkl'
    portraits = pd.read_pickle(filename)

    numAnimals, imsize,\
    X_train, Y_train,\
    X_val, Y_val = DataFirstFineTuning(fragments, portraits)

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
        Vindices, Viter_per_epoch)
