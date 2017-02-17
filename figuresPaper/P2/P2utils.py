import cv2
import sys
sys.path.append('../../utils')

from py_utils import *

import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Colormap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

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
import pyautogui

def idTrajectories(videoPath, sessionPath, allFragIds, dfGlobal, numAnimals):

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
    saveFile(videoPath, trajDict, 'trajectories',hdfpkl = 'pkl',sessionPath = sessionPath)

    return trajDict

def plotTrajectories(trajDict, numAnimals, fig, ax1, ax2, ax3, framesToPlot=[], plotBoth=False):
    sns.set_style("darkgrid")

    centroidTrajectories = np.asarray(trajDict['centroids'])
    nosesTrajectories = np.asarray(trajDict['noses'])

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.set_facecolor('none')
    plots1 = []

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.get_xaxis().tick_bottom()
    ax2.set_facecolor('none')
    plots2 = []

    if plotBoth:
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.get_xaxis().tick_bottom()
        ax3.set_facecolor('none')
        plots3 = []



    for ID in range(numAnimals):
        # print ID
        centID = centroidTrajectories[:,ID,:]
        noseID = nosesTrajectories[:,ID,:]

        if len(framesToPlot) != 0:
            start = framesToPlot[0]
            end = framesToPlot[1]
        else:
            start = 0
            end = len(centID)

        xcentID = centID[:,0][start:end]
        ycentID = centID[:,1][start:end]
        xnoseID = noseID[:,0][start:end]
        ynoseID = noseID[:,1][start:end]
        zs = range(len(xcentID))

        label = 'Animal ' + str(ID)
        p1 = ax1.plot(xcentID,ycentID,zs, label=label)
        l1 = ax1.legend(fancybox=True, framealpha=0.05)
        p2 = ax2.plot(xnoseID,ynoseID,zs, label=label)
        l2 = ax2.legend(fancybox=True, framealpha=0.05)
        plots1.append(p1)
        plots2.append(p2)

        if plotBoth:
            p31 = ax3.plot(xcentID,ycentID,zs, label=label + ' centroid')
            p32 = ax3.plot(xnoseID,ynoseID,zs, label = label + ' nose')
            l3 = ax3.legend(fancybox=True, framealpha=0.05)
            plots3.append([p31,p32])

    if plotBoth:
        plots3 = flatten(plots3)
    plotted1 = {}

    for legline, origline in zip(l1.get_lines(), flatten(plots1)):
        legline.set_picker(5)
        plotted1[legline] = origline

    for legline, origline in zip(l2.get_lines(), flatten(plots2)):
        legline.set_picker(5)
        plotted1[legline] = origline
    if plotBoth:
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

def orderVideo(matrixToOrder,permutations,maxNumBlobs):
    matrixOrdered = np.zeros_like(matrixToOrder)

    for frame in range(len(permutations)):
        for i in range(maxNumBlobs):
            index = list(np.where(permutations[frame]==i)[0])
            # print index
            if len(index) == 1:
                matrixOrdered[frame,i] = matrixToOrder[frame,index]
            else:
                matrixOrdered[frame,i] = -1

    return matrixOrdered

def plotFragments(accumDict, fragmentsDict, portraits, idUsedIndivIntervals, numAnimals, ax):
    fragsForTrain = accumDict['fragsForTrain']
    accumCounter = accumDict['counter']

    fragments = fragmentsDict['fragments']
    permutations = np.asarray(portraits.loc[:,'permutations'].tolist())
    maxNumBlobs = len(permutations[0])
    permOrdered =  orderVideo(permutations,permutations,maxNumBlobs)
    permOrdered = permOrdered.T.astype('float32')

    ax.cla()
    permOrdered[permOrdered >= 0] = 1.
    im = ax.imshow(permOrdered,cmap=plt.cm.gray,interpolation='none',vmin=0.,vmax=1.)
    im.cmap.set_under('k')

    colors = get_spaced_colors_util(numAnimals,norm=True)
    # print numAnimals
    # print colors
    for (frag,ID) in idUsedIndivIntervals:
        # print identity
        blobIndex = frag[0]
        start = frag[2][0]
        end = frag[2][1]
        ax.add_patch(
            patches.Rectangle(
                (start, blobIndex-0.5),   # (x,y)
                end-start,  # width
                1.,          # height
                fill=True,
                edgecolor=None,
                facecolor=colors[ID+1],
                alpha = 1.
            )
        )

    ax.axis('tight')
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Blob index')
    ax.set_yticks(range(0,maxNumBlobs,4))
    ax.set_yticklabels(range(1,maxNumBlobs+1,4))
    ax.invert_yaxis()

def plotAccLoss(trainPlot, valPlot, ax, xlim, ylim = 1.):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_facecolor('none')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss function')
    ax.legend(fancybox=True, framealpha=0.05)
    ax.set_xlim((0,xlim))
    ax.set_ylim((0,ylim))
    ax.plot(trainPlot,'r-', label='training')
    ax.plot(valPlot, 'b-', label='validation')
