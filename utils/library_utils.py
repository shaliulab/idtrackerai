import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from fragmentation import *
from get_portraits import *

from py_utils import *
import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import time
import re
from joblib import Parallel, delayed
import multiprocessing
import itertools
import cPickle as pickle
import math
from natsort import natsorted, ns
from os.path import isdir, isfile
import scipy.spatial.distance as scisd

def orderCenters(centers,camera): ### TODO check if this function does what you were thinking about Paco, I go back to the f****** bug...
    """
    orders points counterclockwise
    Assuming that (0,0) is in the top-left corner because the centers come from the image
    """
    #select the circle with minimal x (at most the arena can be 44deg rotated wrt its center)
    cents = np.asarray(centers)
    #centroid of the four points
    centroid = np.divide(np.sum(cents, axis=0), cents.shape[0])
    #put the centroid in the origin
    trCenters = np.subtract(centers, centroid)
    #take the signum
    signum = np.sign(trCenters)
    #compute the arctan to order
    arctans = [np.arctan2(s[0],s[1]) for s in signum]
    # just to check
    #signum = signum[np.argsort(arctans)]
    cents = cents[np.argsort(arctans)]
    cents = list(tuple(map(tuple,cents)))
    def shift(seq, n):
        n = n % len(seq)
        return seq[n:] + seq[:n]
    # print 'ordered centers, ', cents
    if camera == 2: ### NOTE We assume only 4 arenas in a square so that a rotation of 180 degrees is shifting 2 positions the centers
        cents = shift(cents, 2)
    # print 'ordered centers after rotation, ', cents
    return cents

# centers = [(-1,-1),(1,1),(1,-1),(-1,1)]
# print 'centers, ', centers
# centers = orderCenters(centers,2)

def assignCenterFrame(centers,centroids,camera = 1):
    """
    centers: centers of the arenas
    centroids: centroids of the fish for a frame
    """
    centers = orderCenters(centers,camera) # Order the centers counterclockwise
    # print 'centers ordered,', centers
    d = scisd.cdist(centers,centroids) # Compute the distance of each fish to each centroid
    identities = np.argmin(d,axis=0) # Assign identity by the centroid to which they are closer

    return identities

## Unit test
# centers = [(1,1),(-1,1),(-1,-1),(1,-1)]
# print 'centers, ', centers
# centroids = [(-1,1),(1,1),(-1,-1),(1,-1)]
# print 'centroids, ', centroids
# identities = assignCenterFrame(centers,centroids,2)
# print identities


def assignCenterAndSave(path,centers, camera):
    df, _ = loadFile(path, 'segmentation', time=0)
    dfPermutations = pd.DataFrame(index=df.index,columns={'permutation'})
    for centroids, index in zip(df.centroids,df.index):
        dfPermutations.loc[index,'permutation'] = assignCenterFrame(centers,centroids,camera)

    df['permutation'] = dfPermutations
    saveFile(path, df, 'segment', time = 0)

def assignCenters(paths,centers,camera = 1):
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    Parallel(n_jobs=num_cores)(delayed(assignCenterAndSave)(path, centers, camera) for path in paths)

def portraitsToIMDB(portraits, numAnimalsInGroup, groupNum):
    images = np.asarray(flatten([port for port in portraits.loc[:,'images'] if len(port) == numAnimalsInGroup]))
    images = np.expand_dims(images, axis=1) #this is because the images are in gray scale and we need the channels to be a dimension explicitely
    imsize = (images.shape[1],images.shape[2], images.shape[3])
    labels = np.asarray(flatten([perm for perm in portraits.loc[:,'permutations'] if len(perm) == numAnimalsInGroup])) + numAnimalsInGroup*groupNum
    if len(images) != len(labels):
        raise ValueError('The number of images and labels should match.')
    print 'Group, ', groupNum
    print 'Labels, ', labels
    labels = np.expand_dims(labels, axis=1)

    return imsize, images, labels

def retrieveInfoLib(libPath, preprocessing = "curvature_portrait"):
    #the folder's name is the age of the individuals
    ageInDpf = os.path.split(libPath)[-1]

    # get the list of subfolders
    subDirs = [d for d in os.listdir(libPath) if isdir(libPath +'/'+ d)]
    subDirs = [subDir for subDir in subDirs if 'group' in subDir]
    subDirs = natsorted(subDirs, alg=ns.IGNORECASE)

    return ageInDpf, preprocessing, subDirs
