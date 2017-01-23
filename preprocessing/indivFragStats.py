import sys
sys.path.append('../utils')

from py_utils import *
from video_utils import *

import time
import numpy as np
from matplotlib import pyplot as plt
from Tkinter import *
import tkMessageBox
import argparse
import os
import glob
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import cPickle as pickle
import math

# videoPaths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
videoPaths = scanFolder('../nofragsError/_1.avi')
# videoPaths = scanFolder('../larvae1/trial_1_1.avi')
portraits = loadFile(videoPaths[0], 'portraits', time=0)
fragmentsDict = loadFile(videoPaths[0],'fragments',time=0,hdfpkl='pkl')
frameIndices = loadFile(videoPaths[0], 'frameIndices', time=0)

def getCentroidsFragment(portraits,framesAndColumns):
    centroids = []
    for (frame,column) in framesAndColumns:
        centroids.append(portraits.loc[frame,'noses'][column])
    centroids = np.asarray(centroids).astype('float32')
    return centroids

def getVelocities(centroids):
    '''
    instant velocity in pixels/frame
    '''
    return np.sqrt(np.sum(np.diff(centroids,axis=0)**2,axis=1))

def getDistanceTraveled(centroids):
    '''
    in pixels
    '''
    return np.sum(np.sqrt(np.sum(np.diff(centroids,axis=0)**2,axis=1)),axis=0)

def getBodyLength(videoPaths,framesAndColumns,frameIndices):
    '''
    Estimation of the body length using the diagonal of the boundingBoxes (((x,y),(x+w,y+h)))
    '''
    diags = []
    numSeg = 0
    for (frame,column) in framesAndColumns:
        numSegNew = frameIndices.loc[frame,'segment']
        frameSeg = frameIndices.loc[frame,'frame']
        print numSegNew, frameSeg, frame, column
        print numSeg != numSegNew
        print numSeg, numSegNew
        if numSeg != numSegNew:
            print 'new segment different than old one'
            df, _ = loadFile(videoPaths[numSegNew-1], 'segmentation', time=0)
            numSeg = numSegNew
            print 'numSeg, ', numSeg
        try:
            bb = df.loc[frameSeg,'boundingBoxes'][column]
        except:
            print df.loc[frameSeg,'boundingBoxes']
        diag = np.sqrt((bb[1][0] - bb[0][0])**2 + (bb[1][1] - bb[0][1])**2)
        diags.append(diag)
    return np.mean(diags)



framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][0]
StdGlobalFrag = []
for i, framesAndColumnsInterval in enumerate(framesAndColumnsGlobalFrag):
    print '\nIndividual fragment ', i
    BL = getBodyLength(videoPaths,framesAndColumnsInterval,frameIndices)
    centroids = getCentroidsFragment(portraits,framesAndColumnsInterval)
    distanceTraveled = getDistanceTraveled(centroids)
    velocities = getVelocities(centroids)
    print 'body length, ', BL
    print 'numFrames, ', len(framesAndColumnsInterval)
    print 'std position, ', centroids.std(axis=0)
    print 'distanceTraveled, ', distanceTraveled
    print 'averageVelocity', np.mean(velocities)

    plt.ion()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(centroids[:,0],centroids[:,1])
    plt.xlim((0,1500))
    plt.ylim((0,1500))


    plt.subplot(2,1,2)
    hist, edges = np.histogram(velocities, bins = 50, range=(0,10))
    centers = edges[:-1] + np.diff(edges)/2.
    plt.plot(centers, hist)
