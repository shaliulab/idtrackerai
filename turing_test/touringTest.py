import cv2
import sys
sys.path.append('../utils')

from py_utils import *

import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
import seaborn as sns
from matplotlib.image import AxesImage

numSegment = 0
# paths = scanFolder('../Cafeina5pecesLarge/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi')
# paths = scanFolder('../Medaka/20fish_20130909T191651_1.avi')
# paths = scanFolder('../Cafeina5pecesSmall/Caffeine5fish_20140206T122428_1.avi')
# paths = scanFolder('../BigGroup/manyFish_26dpf_20161110_1.avi')
# paths = scanFolder('../38fish_adult_splitted/adult1darkenes_1.avi')
paths = scanFolder('/home/lab/Desktop/aggr/video_4/4.avi')

frameIndices = loadFile(paths[0], 'frameIndices', time=0)
videoInfo = loadFile(paths[0], 'videoInfo', time=0)
videoInfo = videoInfo.to_dict()[0]
stats = loadFile(paths[0], 'statistics', time=0)
stats = stats.to_dict()[0]
numAnimals = videoInfo['numAnimals']
allFragIds = stats['fragmentIds']
dfGlobal = loadFile(paths[0], 'portraits', time=0)

def idTrajectories(allFragIds, numAnimals, test=True):
    path = paths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)

    frameCounter = 0

    portraitsTrajectories = []

    for i, path in enumerate(paths):
        # capture video to get frame information relative to the segment and then release it
        cap = cv2.VideoCapture(path)
        numFrameSegment = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.release()
        # take the identities relative to the segment, load dataframe to have centroids
        allIdsSegment = allFragIds[frameCounter : frameCounter + numFrameSegment]
        df, sNumber = loadFile(path, 'segmentation', time=0)
        sNumber = int(sNumber)

        # centroids = df['centroids'].tolist()
        # noses = dfGlobal['noses'].tolist()
        portraits = dfGlobal['images'].tolist()
        portraits = portraits[frameCounter : frameCounter + numFrameSegment]
        # noses = noses[frameCounter : frameCounter + numFrameSegment]

        frameCounter += numFrameSegment
        segmentPortraits = []

        for i, IDs in enumerate(allIdsSegment):
            # print 'IDs ', IDs
            curPortraits = portraits[i]

            ordPortraits = []
            for ID in IDs:
                if ID != -1:
                    ordPortraits.append(curPortraits[ID])



            # print 'ord cent ', ordCentroids
            # print 'ord noses ', ordNoses
            segmentPortraits.append(ordPortraits)

        portraitsTrajectories.append(segmentPortraits)

    portraitsTrajectories = flatten(portraitsTrajectories)
    portDict = {'portraits': portraitsTrajectories}
    portraitsIDs = pd.DataFrame(data = portDict)

    # saveFile(path, portraitsIDs, 'portraitsOrdered', time = 0)

    if test == True:
        global counter2
        plt.ion()
        fig, axs = plt.subplots(1,numAnimals, figsize=(18, 6), facecolor='w', edgecolor='k')
        fig.suptitle('Can you recognize who is who?')
        axs = axs.ravel()
        counter = 0
        guesses = []
        permutedIdentities = []
        # permute frame in time
        permFrames = np.random.permutation(len(portraitsTrajectories))
        portraitsTrajectories = np.asarray(portraitsTrajectories)[permFrames]
        numReferences = 10
        for portraits in portraitsTrajectories:
            print 'rendering fish for the first time'
            if counter < numReferences and len(portraits) == numAnimals:

                for j,port in enumerate(portraits):
                    fishName = 'fish ' + str(j)
                    axs[j].set_title(fishName)
                    axs[j].imshow(port, cmap = 'gray', interpolation = 'none')
                plt.draw()
                counter += 1
                time.sleep(10)

            elif len(portraits) == numAnimals:

                counter2 = 0
                perm = np.random.permutation(numAnimals)
                permutedIdentities.append(perm)
                print perm
                print 'the test started'
                for i,j in enumerate(perm):

                    fishName = 'fish ?'
                    axs[i].set_title(fishName)
                    axs[i].imshow(portraits[j], cmap = 'gray', interpolation = 'none')


                plt.draw()
                # fig.canvas.mpl_connect('button_press_event', pass)
                # fig.canvas.mpl_connect('pick_event', onpick4)


                ids = ''
                while ids == '' or len(ids) != numAnimals:
                    ids = getInput('ids','type ids in order')
                ids = [int(c) for c in ids]
                guesses.append(ids)
                counter += 1

            if counter == numReferences + 10:
                plt.close()
                plt.ioff()
                guesses = np.asarray(guesses)
                permutedIdentities = np.asarray(permutedIdentities)
                print guesses
                print permutedIdentities
                subtract = np.subtract(guesses,permutedIdentities)
                print subtract
                correct = len(np.where(subtract==0)[0])
                perc = np.true_divide(correct, np.prod(subtract.shape)) * 100
                print perc
                # summing = np.sum(subtract, axis=0)
                # perc = [np.true_divide(abs(s), len(guesses))*100 for s in summing]
                # print perc
                # avPerc = np.mean(perc)
                # print avPerc
                break
    return portraitsIDs

t = idTrajectories(allFragIds, numAnimals)
