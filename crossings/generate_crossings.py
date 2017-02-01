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
height = videoInfo['height']
width = videoInfo['width']
stats = loadFile(paths[0], 'statistics', time=0)
stats = stats.to_dict()[0]
numAnimals = videoInfo['numAnimals']
allFragIds = stats['fragmentIds']
dfGlobal = loadFile(paths[0], 'portraits', time=0)

def idTrajectories(allFragIds, numAnimals, height, width, show=True):
    path = paths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)

    frameCounter = 0

    centroidTrajectories = []
    nosesTrajectories = []
    miniframeTrajectories = []
    bbsTrajectories = []

    for i, path in enumerate(paths):
        # capture video to get frame information relative to the segment and then release it
        cap = cv2.VideoCapture(path)
        numFrameSegment = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.release()
        # take the identities relative to the segment, load dataframe to have centroids
        allIdsSegment = allFragIds[frameCounter : frameCounter + numFrameSegment]
        df, sNumber = loadFile(path, 'segmentation', time=0)
        sNumber = int(sNumber)
        print 'data we have', df.keys()
        miniframes = df['miniFrames'].tolist()
        centroids = df['centroids'].tolist()
        boundingBoxes = df['boundingBoxes'].tolist()
        noses = dfGlobal['noses'].tolist()
        noses = noses[frameCounter : frameCounter + numFrameSegment]

        frameCounter += numFrameSegment
        segmentCentroids = []
        segmentNoses = []
        segmentMiniframes = []
        segmentBbs = []

        for i, IDs in enumerate(allIdsSegment):
            # print 'IDs ', IDs
            curCentroids = centroids[i]
            curNoses = noses[i]
            curMiniframes = miniframes[i]
            curBbs = boundingBoxes[i]

            ordCentroids = []
            ordNoses = []
            ordMiniframes = []
            ordBbs = []
            # print 'centroids ', curCentroids
            # print 'noses ', curNoses
            for ID in IDs:
                if ID != -1:
                    ordCentroids.append(curCentroids[ID])
                    ordNoses.append(curNoses[ID])
                    ordMiniframes.append(curMiniframes[ID])
                    ordBbs.append(curBbs[ID])
                elif ID == -1 and len(ordCentroids) < numAnimals:
                    ordCentroids.append((np.nan, np.nan))
                    ordNoses.append((np.nan, np.nan))
                    ordMiniframes.append(np.nan)
                    ordBbs.append(np.nan)


            # print 'ord cent ', ordCentroids
            # print 'ord noses ', ordNoses
            segmentCentroids.append(ordCentroids)
            segmentNoses.append(ordNoses)
            segmentMiniframes.append(ordMiniframes)
            segmentBbs.append(ordBbs)

        centroidTrajectories.append(segmentCentroids)
        nosesTrajectories.append(segmentNoses)
        miniframeTrajectories.append(segmentMiniframes)
        bbsTrajectories.append(segmentBbs)


    centroidTrajectories = flatten(centroidTrajectories)
    nosesTrajectories = flatten(nosesTrajectories)
    miniframeTrajectories = flatten(miniframeTrajectories)
    bbsTrajectories = flatten(bbsTrajectories)

    trajDict = {'centroids': centroidTrajectories, 'noses': nosesTrajectories, 'miniframes': miniframeTrajectories, 'bbs': bbsTrajectories}
    trajectories = pd.DataFrame(data = trajDict)


    # saveFile(path, trajectories, 'trajectories_with_miniframes', time = 0)
    #
    # if show == True:
    #     sns.set_style("darkgrid")
    #
    #     centroidTrajectories = np.asarray(centroidTrajectories)
    #     # print centroidTrajectories[centroidTrajectories == (-1,-1)]
    #     nosesTrajectories = np.asarray(nosesTrajectories)
    #     # nosesTrajectories[nosesTrajectories == (-1,-1)] =  (np.nan, np.nan)
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1,3,1, projection='3d')
    #     ax2 = fig.add_subplot(1,3,2, projection='3d')
    #     ax3 = fig.add_subplot(1,3,3, projection='3d')
    #
    #     ax1.spines["top"].set_visible(False)
    #     ax1.spines["right"].set_visible(False)
    #     ax1.get_xaxis().tick_bottom()
    #     ax1.set_axis_bgcolor('none')
    #
    #
    #     ax2.spines["top"].set_visible(False)
    #     ax2.spines["right"].set_visible(False)
    #     ax2.get_xaxis().tick_bottom()
    #     ax2.set_axis_bgcolor('none')
    #
    #     ax3.spines["top"].set_visible(False)
    #     ax3.spines["right"].set_visible(False)
    #     ax3.get_xaxis().tick_bottom()
    #     ax3.set_axis_bgcolor('none')
    #
    #     for ID in range(numAnimals):
    #         print ID
    #         centID = centroidTrajectories[:,ID,:]
    #         noseID = nosesTrajectories[:,ID,:]
    #         xcentID = centID[:,0]
    #         ycentID = centID[:,1]
    #         xnoseID = noseID[:,0]
    #         ynoseID = noseID[:,1]
    #         zs = range(len(xcentID))
    #
    #         label = 'Animal ' + str(ID)
    #         ax1.plot(xcentID,ycentID,zs, label=label)
    #         ax1.legend(fancybox=True, framealpha=0.05)
    #         ax2.plot(xnoseID,ynoseID,zs, label=label)
    #         ax2.legend(fancybox=True, framealpha=0.05)
    #         ax3.plot(xcentID,ycentID,zs, label=label + ' centroid')
    #         ax3.plot(xnoseID,ynoseID,zs, label = label + ' nose')
    #         ax3.legend(fancybox=True, framealpha=0.05)
    #
    #     plt.show()
    return trajectories

t = idTrajectories(allFragIds, numAnimals, height, width)
