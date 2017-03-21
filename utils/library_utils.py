import cv2
import sys
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/preprocessing')

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

# def orderCenters(centers,video, transform):
#     """
#     orders points counterclockwise
#     Assuming that (0,0) is in the top-left corner because the centers come from the image
#     """
#     #select the circle with minimal x (at most the arena can be 44deg rotated wrt its center)
#     cents = np.asarray(centers)
#     #centroid of the four points
#     centroid = np.divide(np.sum(cents, axis=0), cents.shape[0])
#     #put the centroid in the origin
#     trCenters = np.subtract(centers, centroid)
#     #take the signum
#     signum = np.sign(trCenters)
#     #compute the arctan to order
#     arctans = [np.arctan2(s[0],s[1]) for s in signum]
#     # just to check
#     #signum = signum[np.argsort(arctans)]
#     cents = cents[np.argsort(arctans)]
#     cents = list(tuple(map(tuple,cents)))
#     def shift(seq, n):
#         n = n % len(seq)
#         return seq[n:] + seq[:n]
#     # print 'ordered centers, ', cents
#     if video == 2: ### NOTE We assume only 4 arenas in a square so that a rotation of 180 degrees is shifting 2 positions the centers
#         cents = shift(cents, 2)
#     # print 'ordered centers after rotation, ', cents
#     return cents

def orderCenters(centers, video, transform):
    """
    orders points according to the video number and the transformation ('rotation', 'translation' or 'none')
    this is supposed to work for the following libraries in the following way:
    TU20160413 and TU20160428:
        transform = 'rotation' (180 degrees)
        video1:             video2:
            0 3                 2 1
            1 2                 3 0
    TU20160920
        video1:
            1 0
            2 3
    TU20170131, TU20170201 and TU20170202:
        video1:             video2:
            1 0     7 6         7 6     1 0
            2 3     4 5         4 5     2 3
    Assuming that (0,0) is in the top-left corner because the centers come from the image
    """
    #select the circle with minimal x (at most the arena can be 44deg rotated wrt its center)
    cents = np.asarray(centers)
    # print 'cents, ', cents
    #centroid of the centers
    centroid = np.true_divide(np.sum(cents, axis=0), cents.shape[0])
    # print 'centroid, ', centroid
    #put the centroid in the origin
    trCenters = np.subtract(cents, centroid)
    # print 'trCenters, ', trCenters
    #take the signum
    # signum = np.sign(trCenters)
    # print 'signum, ', signum
    #compute the arctan to order
    arctans = [np.arctan2(s[0],s[1]) for s in trCenters]
    # print 'arctans, ', arctans
    cents = cents[np.argsort(arctans)]
    cents = list(tuple(map(tuple,cents)))
    # print 'cents'
    def shift(seq, n):
        n = n % len(seq)
        return seq[n:] + seq[:n]
    def rearrange(cents):
        newCents = []
        newCents.append(cents[6])
        newCents.append(cents[7])
        newCents.append(cents[4])
        newCents.append(cents[5])
        newCents.append(cents[2])
        newCents.append(cents[3])
        newCents.append(cents[0])
        newCents.append(cents[1])
        return np.asarray(newCents)

    if video == 2 and transform == 'rotation':
        # print 'video, ', video
        # print 'applying ', transform
        cents = shift(cents, 2)
    if video == 2 and transform == 'translation':
        # print 'video, ', video
        # print 'applying ', transform
        cents = rearrange(cents)

    return cents

def assignCenterFrame(centers,centroids,video, transform):
    """
    centers: centers of the arenas
    centroids: centroids of the fish for a frame
    """
    centers = orderCenters(centers,video,transform) # Order the centers counterclockwise
    # print 'centers ordered,', centers
    d = scisd.cdist(centers,centroids) # Compute the distance of each fish to each centroid
    identities = np.argmin(d,axis=0) # Assign identity by the centroid to which they are closer

    return identities #, centers

# Unit test
# centers = [(1,1),(2,2),(2,1),(1,2)] # centers of the petri dishes
# print 'centers, ', centers
# centroids = [(1,1),(2,1),(1,2),(2,2)] # centroids of the animals in the frame
# print 'centroids, ', centroids
# identities, orderedCenters = assignCenterFrame(centers,centroids,video = 2, transform = 'rotation')
# print identities
#
# centers = [(1,1),(2,2),(2,1),(1,2),(4,1),(4,2),(5,1),(5,2)] # centers of the petri dishes
# print 'centers, ', centers
# centroids = [(1,1),(2,1),(1,2),(2,2),(4,1),(4,2),(5,2),(5,1)] # centroids of the animals in the frame
# print 'centroids, ', centroids
# identities, orderedCenters = assignCenterFrame(centers,centroids,video = 2, transform = 'translation')
# print identities
#
# plt.figure()
# ax = plt.subplot(111)
# ax.axis('equal')
# for i, (x,y) in enumerate(centers):
#     petridish = plt.Circle((x, y), 0.4, color='k', fill=False)
#     ax.add_patch(petridish)
#     ax.text(x-0.1,y-0.1,'c' + str(i), fontsize = 14, fontweight = 'bold', ha = 'center' , va = 'center')
#     ax.text(orderedCenters[i][0]+0.1,orderedCenters[i][1]-0.1,'p' + str(i),fontsize = 14, fontweight = 'bold', color = 'r', ha = 'center' , va = 'center')
#     ax.text(centroids[i][0]-0.1,centroids[i][1]+0.1,'b' + str(i),fontsize = 14, fontweight = 'bold', color = 'b', ha = 'center' , va = 'center')
#     ax.text(centroids[i][0]+0.1,centroids[i][1]+0.1,'i' + str(identities[i]),fontsize = 14, fontweight = 'bold', color = 'g', ha = 'center' , va = 'center')
# ax.set_xlim((0,6))
# ax.set_ylim((0,3))
# ax.invert_yaxis()
# plt.show()


def assignCenterAndSave(path,centers, video, transform):
    df, numSegment = loadFile(path, 'segmentation')
    dfPermutations = pd.DataFrame(index=df.index,columns={'permutations'})
    frameIndices = loadFile(path, 'frameIndices')
    segmentIndices = frameIndices.loc[frameIndices.loc[:,'segment']==int(numSegment)]
    segmentIndices = segmentIndices.index.tolist()
    dfGlobal = pd.DataFrame(index = segmentIndices,columns={'identities','permutations','centroids','areas'})
    for i, (centroids, index) in enumerate(zip(df.centroids,df.index)):
        dfPermutations.loc[index,'permutations'] = assignCenterFrame(centers,centroids,video,transform)
        dfGlobal.loc[segmentIndices[i],'identities'] = dfPermutations.loc[index,'permutations']
        dfGlobal.loc[segmentIndices[i],'permutations'] = dfPermutations.loc[index,'permutations']
        dfGlobal.loc[segmentIndices[i],'centroids'] = centroids
        dfGlobal.loc[segmentIndices[i],'areas'] = df.loc[index,'areas']

    df['permutations'] = dfPermutations['permutations']
    saveFile(path, df, 'segment')
    return dfGlobal

def assignCenters(paths,centers,video = 1,transform = 'none'):
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1

    outDf = Parallel(n_jobs=num_cores)(delayed(assignCenterAndSave)(path, centers, video, transform) for path in paths)

    dfGlobal = pd.concat(outDf)
    dfGlobal = dfGlobal.sort_index(axis=0,ascending=True)
    return dfGlobal

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

def newMiniframesToIMDB(newMiniframes):
    images = np.asarray(flatten([port for port in newMiniframes.loc[:,'images']]))
    images = np.expand_dims(images, axis=1) #this is because the images are in gray scale and we need the channels to be a dimension explicitely
    imsize = (images.shape[1],images.shape[2], images.shape[3])
    labels = np.reshape(np.asarray(zip(flatten(newMiniframes.loc[:,'noses'].tolist()),flatten(newMiniframes.loc[:,'middleP'].tolist()))),[-1,4])
    # labels = np.asarray(flatten([zip(noses,middlePs) for (noses,middlePs) in zip(flatten(newMiniframes.loc[:,'noses'].tolist()),newMiniframes.loc[:,'middleP'].tolist())))
    if len(images) != len(labels):
        raise ValueError('The number of images and labels should match.')
    # labels = np.expand_dims(labels, axis=1)

    return imsize, images, labels

def retrieveInfoLib(libPath, preprocessing = "curvature_portrait"):
    #the folder's name is the age of the individuals
    ageInDpf = os.path.split(libPath)[-1]
    strain = libPath.split('/')[-2]

    # get the list of subfolders
    subDirs = [d for d in os.listdir(libPath) if isdir(libPath +'/'+ d)]
    subDirs = [subDir for subDir in subDirs if 'group' in subDir or 'Group' in subDir and 'prep' not in subDir]
    subDirs = natsorted(subDirs, alg=ns.IGNORECASE)

    return ageInDpf, preprocessing, subDirs, strain
