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

''' select statistics file '''
statisticsPath = selectFile()
print 'The trajectories will be build from the statiscits file ', statisticsPath
sessionPath = os.path.dirname(statisticsPath)
print 'The trajectories will be build from the session ', sessionPath
CNN_modelsPath = os.path.dirname(sessionPath)
print 'The CNN_models folder is ', CNN_modelsPath
pathToVideos = os.path.dirname(CNN_modelsPath)
print 'The video folder is ', pathToVideos
''' get videoPath from statistics file '''

extensions = ['.avi', '.mp4']
videoPath = natural_sort([v for v in os.listdir(pathToVideos) if os.path.isfile(pathToVideos +'/'+ v) if any( ext in v for ext in extensions)])[0]
videoPath = pathToVideos + '/' + videoPath
videoPaths = scanFolder(videoPath)

videoInfo = loadFile(videoPaths[0], 'videoInfo', hdfpkl='pkl')
stats = loadFile(videoPaths[0], 'statistics', hdfpkl = 'pkl',sessionPath = sessionPath)
numAnimals = videoInfo['numAnimals']
allFragIds = stats['fragmentIds']
dfGlobal = loadFile(videoPaths[0], 'portraits')

def idTrajectories(allFragIds, numAnimals, show=True):
    path = videoPaths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)

    frameCounter = 0

    centroidTrajectories = []
    nosesTrajectories = []

    centroids = dfGlobal['centroids'].tolist()
    noses = dfGlobal['noses'].tolist()

    for j, IDs in enumerate(allFragIds):
        curCentroids = centroids[j]
        curNoses = noses[j]

        ordCentroids = [(np.nan, np.nan) for k in range(numAnimals)]
        ordNoses = [(np.nan, np.nan) for k in range(numAnimals)]

        for l,ID in enumerate(IDs):
            if ID != -1:
                ordCentroids[ID] = curCentroids[l]
                ordNoses[ID] = curNoses[l]

        centroidTrajectories.append(ordCentroids)
        nosesTrajectories.append(ordNoses)

    trajDict = {'centroids': centroidTrajectories, 'noses': nosesTrajectories}
    trajectories = pd.DataFrame(data = trajDict)
    saveFile(videoPaths[0], trajectories, 'trajectories',hdfpkl = 'pkl',sessionPath = sessionPath)

    if show == True:
        sns.set_style("darkgrid")

        centroidTrajectories = np.asarray(centroidTrajectories)
        nosesTrajectories = np.asarray(nosesTrajectories)

        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1, projection='3d')
        ax2 = fig.add_subplot(1,3,2, projection='3d')
        ax3 = fig.add_subplot(1,3,3, projection='3d')

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.get_xaxis().tick_bottom()
        ax1.set_axis_bgcolor('none')


        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.get_xaxis().tick_bottom()
        ax2.set_axis_bgcolor('none')

        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.get_xaxis().tick_bottom()
        ax3.set_axis_bgcolor('none')
        plots1 = []
        plots2 = []
        plots3 = []
        for ID in range(numAnimals):
            print ID
            centID = centroidTrajectories[:,ID,:]
            noseID = nosesTrajectories[:,ID,:]
            xcentID = centID[:,0]
            ycentID = centID[:,1]
            xnoseID = noseID[:,0]
            ynoseID = noseID[:,1]
            zs = range(len(xcentID))

            label = 'Animal ' + str(ID)
            p1 = ax1.plot(xcentID,ycentID,zs, label=label)
            l1 = ax1.legend(fancybox=True, framealpha=0.05)
            p2 = ax2.plot(xnoseID,ynoseID,zs, label=label)
            l2 = ax2.legend(fancybox=True, framealpha=0.05)
            p31 = ax3.plot(xcentID,ycentID,zs, label=label + ' centroid')
            p32 = ax3.plot(xnoseID,ynoseID,zs, label = label + ' nose')
            l3 = ax3.legend(fancybox=True, framealpha=0.05)
            plots1.append(p1)
            plots2.append(p2)
            plots3.append([p31,p32])

        plots3 = flatten(plots3)
        plotted1 = {}

        for legline, origline in zip(l1.get_lines(), flatten(plots1)):
            legline.set_picker(5)
            plotted1[legline] = origline

        for legline, origline in zip(l2.get_lines(), flatten(plots2)):
            legline.set_picker(5)
            plotted1[legline] = origline

        for legline, origline in zip(l3.get_lines(), flatten(plots3)):
            legline.set_picker(5)
            plotted1[legline] = origline

        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = plotted1[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()
    return trajectories

t = idTrajectories(allFragIds, numAnimals)
