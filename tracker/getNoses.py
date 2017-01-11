import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')
from py_utils import *
from get_miniframes import *
from library_utils import newMiniframesToIMDB

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
import h5py
from pprint import pprint 

numSegment = 0

videoPath = selectFile()
paths = scanFolder(videoPath)

frameIndices = loadFile(paths[0], 'frameIndices', time=0)
videoInfo = loadFile(paths[0], 'videoInfo', time=0)
videoInfo = videoInfo.to_dict()[0]
stats = loadFile(paths[0], 'statistics', time=0)
stats = stats.to_dict()[0]
numAnimals = videoInfo['numAnimals']
allFragIds = stats['fragmentIds']
dfGlobal = loadFile(paths[0], 'portraits', time=0)

def idNoses(allFragIds, numAnimals, frameIndices, show=True):
    path = paths[0]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)

    frameCounter = 0

    frameIndicesList = frameIndices.index.tolist()
    AllNewMiniframes = pd.DataFrame(index = frameIndicesList, columns= ['images', 'noses', 'middleP'])
    counterFrame = 0
    totalNumImages = 0
    for i, path in enumerate(paths):
        # capture video to get frame information relative to the segment and then release it
        cap = cv2.VideoCapture(path)
        numFrameSegment = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.release()
        # take the identities relative to the segment, load dataframe to have centroids
        allIdsSegment = allFragIds[frameCounter : frameCounter + numFrameSegment]
        df, sNumber = loadFile(path, 'segmentation', time=0)
        sNumber = int(sNumber)

        miniframes = df['miniFrames'].tolist()
        cnts = df['contours'].tolist()
        pixels = df['pixels'].tolist()
        bbs = df['boundingBoxes'].tolist()
        bkgSamps = df['bkgSamples'].tolist()

        frameCounter += numFrameSegment

        for j, IDs in enumerate(allIdsSegment):
            curMiniframes = miniframes[j]
            curCnts = cnts[j]
            curPixels = pixels[j]
            curBbs = bbs[j]
            curBkgSamples = bkgSamps[j]

            newMiniframes = []
            newNoses = []
            newMiddleP = []

            for l,ID in enumerate(IDs):
                if ID != -1:
                    new_frame, new_nose, new_m = getMiniframes(curPixels[l], curMiniframes[l],curCnts[l],curBbs[l],curBkgSamples[l],counter = None, path = path)

                    newMiniframes.append(new_frame)
                    newNoses.append(new_nose)
                    newMiddleP.append(new_m)
                    totalNumImages +=1
            # print 'newNoses', newNoses

            AllNewMiniframes.set_value(frameIndicesList[counterFrame], 'images', np.asarray(newMiniframes))
            AllNewMiniframes.set_value(frameIndicesList[counterFrame], 'noses', newNoses)
            AllNewMiniframes.set_value(frameIndicesList[counterFrame], 'middleP', newMiddleP)
            print counterFrame
            counterFrame += 1


    # saveFile(videoPath, allNewMiniframes, 'newMiniframes', time = 0)
    imsize, images, labels = newMiniframesToIMDB(AllNewMiniframes)
    print imsize
    print len(images)
    print len(labels)
    print images
    print labels
    preprocParams= loadFile(path, 'preprocparams',0)
    preprocParams = preprocParams.to_dict()[0]
    numAnimalsInGroup = preprocParams['numAnimals']
    numIndivIMDB =  preprocParams['numAnimals']
    # averageNumImagesPerIndiv = int(np.divide(totalNumImages,numIndivIMDB))
    minimalNumImagesPerIndiv = int(np.divide(len(labels),numAnimalsInGroup))

    ### TODO the dataset name should include the strain of the animals used
    nameDatabase =  str(numIndivIMDB) + 'indiv_' + str(int(minimalNumImagesPerIndiv)) + 'ImPerInd_' + 'curvature'
    if not os.path.exists(folder + '/IMDBs'): # Checkpoint folder does not exist
        os.makedirs(folder + '/IMDBs') # we create a checkpoint folder
    else:
        if os.path.isfile(folder + '/IMDBs/' + nameDatabase + '_0.hdf5'):
            text = 'A IMDB already exist with this name (' + nameDatabase + '). Do you want to create a new one with a different name?'
            newName = getInput('Confirm selection','The IMDB already exist. Do you want to create a new one with a different name [y/n]?')
            if newName == 'y':
                nameDatabase = getInput('Insert new name: ','The current name is "' + nameDatabase + '"')
            elif newName == 'n':
                displayMessage('Overwriting IMDB','You are going to overwrite the current IMDB (' + nameDatabase + ').')
                for filename in glob.glob(folder + '/IMDBs/' + nameDatabase + '_*.hdf5'):
                    os.remove(filename)
            else:
                raise ValueError('Invalid string, it must be "y" or "n"')

    f = h5py.File(folder + '/IMDBs/' + nameDatabase + '_%i.hdf5', driver='family')
    grp = f.create_group("database")


    dset1 = grp.create_dataset("images", images.shape, dtype='f')
    dset2 = grp.create_dataset("labels", labels.shape, dtype='i')

    dset1[...] = images
    dset2[...] = labels

    grp.attrs['originalMatPath'] = folder
    grp.attrs['numIndiv'] = numIndivIMDB
    grp.attrs['imageSize'] = imsize
    # grp.attrs['averageNumImagesPerIndiv'] = averageNumImagesPerIndiv
    grp.attrs['numImagesPerIndiv'] = minimalNumImagesPerIndiv
    grp.attrs['ageInDpf'] = 'unkwnown'
    grp.attrs['preprocessing'] = 'curvature'

    pprint([item for item in grp.attrs.iteritems()])

    f.close()

    print 'Database saved as %s ' % nameDatabase

    return imsize, images, labels

imsize, images, labels = idNoses(allFragIds, numAnimals, frameIndices)
