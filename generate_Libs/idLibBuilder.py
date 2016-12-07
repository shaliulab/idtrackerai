import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *
from library_utils import *

import time
import h5py
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
from pprint import pprint

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    ''' ************************************************************************
    Selecting library directory
    ************************************************************************ '''
    initialDir = '/media/lab/idZebLib_TU20160413_34_36dpf/idZebLib/TU20160413/36dpf'
    libPath = selectDir(initialDir)
    # libPath = '/home/lab/Desktop/TF_models/IdTracker/data/library/25dpf'
    ageInDpf, preprocessing, subDirs = retrieveInfoLib(libPath, preprocessing = "curvaturePortrait")
    group = 0
    imagesIMDB = []
    labelsIMDB = []
    numImagesList = []
    numIndivIMDB = 0
    totalNumImages = 0
    setParams = getInput('Set preprocessing parameters','Do you want to set the parameters for the preprocessing of each video? ([y]/n)')
    runPreproc = getInput('Run preprocessing','Do you want to run the preprocessing? ([y]/n)')
    buildLib = getInput('Build library','Do you want to build the library? ([y]/n)')
    for i, subDir in enumerate(subDirs):
        print '-----------------------'
        print 'preprocessing ', subDir

        ''' Path to video/s '''
        path = libPath + '/' + subDir
        extensions = ['.avi', '.mp4']
        # videoPath = natural_sort([v for v in os.listdir(path) if isfile(path +'/'+ v) if '.avi' in v])[0]
        videoPath = natural_sort([v for v in os.listdir(path) if isfile(path +'/'+ v) if any( ext in v for ext in extensions)])[0]
        videoPath = path + '/' + videoPath
        videoPaths = scanFolder(videoPath)

        ''' ************************************************************************
        Set preprocessing parameters
        ************************************************************************ '''
        if setParams == 'y' or setParams == '':
            skipSubDir = getInput('Skip subdirectory','Do you want to set parameters for this subDir ('+ subDir +')? (Y/n)')
            if skipSubDir == 'n':
                continue
            elif skipSubDir == 'y' or skipSubDir == '':
                ''' ************************************************************************
                GUI to select the preprocessing parameters
                *************************************************************************'''
                prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
                useBkg = prepOpts['bkg']
                useROI =  prepOpts['ROI']
                useBkg = 0

                #Check for preexistent files generated during a previous session. If they
                #exist and one wants to keep them they will be loaded
                processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragmentation','portraits']
                processesDict, srcSubFolder = copyExistentFiles(videoPath, processesList, time=1)
                print processesDict
                loadPreviousDict = selectOptions(processesList, processesDict, text='Already processed steps in this video \n (check to load from ' + srcSubFolder + ')')
                print loadPreviousDict

                usePreviousBkg = loadPreviousDict['bkg']
                usePreviousROI = loadPreviousDict['ROI']

                ''' ROI selection and bkg loading'''
                if i != 0 and not loadPreviousDict['preprocparams']:
                    cv2.namedWindow('Bars')
                videoPaths = scanFolder(videoPath)
                numSegment = 0
                width, height, bkg, mask, centers = playPreview(videoPaths, useBkg, usePreviousBkg, useROI, usePreviousROI)

                ''' Segmentation inspection '''
                if not loadPreviousDict['preprocparams']:
                    print 'Entering segmentation preview'
                    SegmentationPreview(videoPath, width, height, bkg, mask, useBkg)

                    cv2.waitKey(1)
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    numSegment = getInput('Segment number','Type the segment to be visualized')

                    end = False
                    while not end:
                        numSegment = getInput('Segment number','Type the segment to be visualized')
                        if numSegment == 'q' or numSegment == 'quit' or numSegment == 'exit':
                            end = True
                        else:
                            end = False
                            path = videoPaths[int(numSegment)]
                            preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
                            preprocParams = preprocParams.to_dict()[0]
                            numAnimalsInGroup = preprocParams['numAnimals']
                            minThreshold = preprocParams['minThreshold']
                            maxThreshold = preprocParams['maxThreshold']
                            minArea = preprocParams['minArea']
                            maxArea = preprocParams['maxArea']
                            mask = loadFile(videoPaths[0], 'ROI',0)
                            mask = np.asarray(mask)
                            centers= loadFile(videoPaths[0], 'centers',0)
                            centers = np.asarray(centers) ### TODO maybe we need to pass to a list of tuples
                            EQ = 0
                            bkg = checkBkg(useBkg, usePreviousBkg, videoPaths, EQ, width, height)
                            cv2.namedWindow('Bars')
                            SegmentationPreview(path, width, height, bkg, mask, useBkg, minArea, maxArea, minThreshold, maxThreshold)
                        cv2.waitKey(1)
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                else:
                    preprocParams = loadFile(videoPaths[0], 'preprocparams',0)
                    preprocParams = preprocParams.to_dict()[0]
                    numAnimalsInGroup = preprocParams['numAnimals']
                    minThreshold = preprocParams['minThreshold']
                    maxThreshold = preprocParams['maxThreshold']
                    minArea = preprocParams['minArea']
                    maxArea = preprocParams['maxArea']

        ''' ************************************************************************
        Preprocessing
        ************************************************************************ '''
        if runPreproc == 'y' or runPreproc == '':
            ''' ************************************************************************
            Segmentation
            ************************************************************************ '''
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            if not loadPreviousDict['segmentation']:
                preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
                preprocParams = preprocParams.to_dict()[0]
                numAnimalsInGroup = preprocParams['numAnimals']
                minThreshold = preprocParams['minThreshold']
                maxThreshold = preprocParams['maxThreshold']
                minArea = preprocParams['minArea']
                maxArea = preprocParams['maxArea']
                EQ = 0
                print preprocParams
                segment(videoPaths, numAnimalsInGroup,
                            mask, centers, useBkg, bkg, EQ,
                            minThreshold, maxThreshold,
                            minArea, maxArea)

            ''' ************************************************************************
            Fragmentation
            *************************************************************************'''
            ''' Group and number of individuals '''
            # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
            if 'camera' in subDir: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
                camera = int(subDir.split('_')[-1])
                if camera == 1 and i != 0:
                    group += 1
                if camera == 2:
                    numIndivIMDB += numAnimalsInGroup
            else:
                camera = 1
                group += 1
                numIndivIMDB += numAnimalsInGroup
            print 'Camera ', camera

            if not loadPreviousDict['fragmentation']:
                assignCenters(videoPaths,centers,camera)
                playFragmentation(videoPaths,False) # last parameter is to visualize or not
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            ''' ************************************************************************
            Portraying
            ************************************************************************ '''
            if not loadPreviousDict['portraits']:
                portrait(videoPaths)
            # portraits = loadFile(videoPaths[0], 'portraits', time=0)

        if buildLib == 'y' or buildLib == '':
            ''' ************************************************************************
            Build images and labels array
            ************************************************************************ '''
            preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
            preprocParams = preprocParams.to_dict()[0]
            numAnimalsInGroup = preprocParams['numAnimals']
            ''' Group and number of individuals '''
            # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
            if 'camera' in subDir: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
                camera = int(subDir.split('_')[-1])
                if camera == 1 and i != 0:
                    group += 1
                if camera == 2:
                    numIndivIMDB += numAnimalsInGroup
            else:
                camera = 1
                group += 1
                numIndivIMDB += numAnimalsInGroup
            print 'Camera ', camera
            portraits = loadFile(videoPaths[0], 'portraits', time=0)
            groupNum = i
            imsize, images, labels = portraitsToIMDB(portraits, numAnimalsInGroup, group)
            print 'images, shape, ', images.shape
            print 'labels, shape, ', labels.shape
            imagesIMDB.append(images)
            labelsIMDB.append(labels)

            totalNumImages += labels.shape[0]

            numImagesList.append(labels.shape[0])
            print numImagesList

    ''' ************************************************************************
    Save IMDB to hdf5
    ************************************************************************ '''
    if buildLib == 'y' or buildLib == '':
        preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
        preprocParams = preprocParams.to_dict()[0]
        numAnimalsInGroup = preprocParams['numAnimals']

        averageNumImagesPerIndiv = int(np.divide(totalNumImages,numIndivIMDB))
        minimalNumImagesPerIndiv = int(np.divide(np.min(numImagesList),numAnimalsInGroup))*2
        imagesIMDB = np.vstack(imagesIMDB)
        labelsIMDB = np.vstack(labelsIMDB)

        ### TODO the dataset name should include the strain of the animals used
        nameDatabase =  ageInDpf + '_' + str(numIndivIMDB) + 'indiv_' + str(int(minimalNumImagesPerIndiv)) + 'ImPerInd_' + preprocessing
        if not os.path.exists(libPath + '/IMDBs'): # Checkpoint folder does not exist
            os.makedirs(libPath + '/IMDBs') # we create a checkpoint folder
        else:
            if os.path.isfile(libPath + '/IMDBs/' + nameDatabase + '_0.hdf5'):
                text = 'A IMDB already exist with this name (' + nameDatabase + '). Do you want to create a new one with a different name?'
                newName = getInput('Confirm selection','The IMDB already exist. Do you want to create a new one with a different name [y/n]?')
                if newName == 'y':
                    nameDatabase = getInput('Insert new name: ','The current name is "' + nameDatabase + '"')
                elif newName == 'n':
                    displayMessage('Overwriting IMDB','You are going to overwrite the current IMDB (' + nameDatabase + ').')
                    for filename in glob.glob(libPath + '/IMDBs/' + nameDatabase + '_*.hdf5'):
                        os.remove(filename)
                else:
                    raise ValueError('Invalid string, it must be "y" or "n"')

        f = h5py.File(libPath + '/IMDBs/' + nameDatabase + '_%i.hdf5', driver='family')
        grp = f.create_group("database")


        dset1 = grp.create_dataset("images", imagesIMDB.shape, dtype='f')
        dset2 = grp.create_dataset("labels", labelsIMDB.shape, dtype='i')

        dset1[...] = imagesIMDB
        dset2[...] = labelsIMDB

        grp.attrs['originalMatPath'] = libPath
        grp.attrs['numIndiv'] = numIndivIMDB
        grp.attrs['imageSize'] = imsize
        # grp.attrs['averageNumImagesPerIndiv'] = averageNumImagesPerIndiv
        grp.attrs['numImagesPerIndiv'] = minimalNumImagesPerIndiv
        grp.attrs['ageInDpf'] = ageInDpf
        grp.attrs['preprocessing'] = preprocessing

        pprint([item for item in grp.attrs.iteritems()])

        f.close()

        print 'Database saved as %s ' % nameDatabase
