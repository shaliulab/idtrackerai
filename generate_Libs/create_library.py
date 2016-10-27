import cv2
import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from segmentation_ROI import *
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

def assignCenterFrame(centers,centroids): ### TODO this function can be improved to be faster by using repmats
    """
    centers: centers of the arenas
    centroids: centroids of the fish for a frame
    """
    d = scisd.cdist(centers,centroids)
    centersAssigned = np.argmin(d,axis=0)
    return centersAssigned

### Unit test
# centers = [(1,1),(-1,1),(-1,-1),(1,-1)]
# centroids = [(-1,1),(1,1),(-1,-1),(1,-1)]
# centersAssigned = assignCenterFrame(centers,centroids)
# print centersAssigned


def assignCenterAndSave(path,centers):
    df, _ = loadFile(path, 'segmentation', time=0)
    dfPermutations = pd.DataFrame(index=df.index,columns={'permutation'})
    for centroids, index in zip(df.centroids,df.index):
        dfPermutations.loc[index,'permutation'] = assignCenterFrame(centers,centroids)

    df['permutation'] = dfPermutations
    saveFile(path, df, 'segment', time = 0)

# path = '/home/lab/Desktop/TF_models/IdTracker/data/library/25dpf/group_1_camera_1/group_1_camera_2_20160508T100615_1.avi'
# info = loadFile(path, 'videoInfo', time=0)
# centers = info['ROICenters']
# assignCenterPath(path,centers)
# play([path])

def assignCenters(paths,centers):
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    Parallel(n_jobs=num_cores)(delayed(assignCenterAndSave)(path, centers) for path in paths)

def buildLibrary(libPath,numAnimals,
                bkgSubstraction, selectROI, EQ,
                minThreshold, maxThreshold,
                minArea, maxArea):

    def retrieveInfoLib(libPath, preprocessing = "curvature_portrait"):
        #the folder's name is the age of the individuals
        ageInDpf = os.path.split(libPath)[-1]

        # get the list of subfolders
        subDirs = [d for d in os.listdir(libPath) if isdir(libPath +'/'+ d)]
        subDirs = natsorted(subDirs, alg=ns.IGNORECASE)

        return ageInDpf, preprocessing, subDirs

    def libLoop(libPath, numAnimals,
                bkgSubstraction, selectROI, EQ,
                minThreshold, maxThreshold,
                minArea, maxArea):

        ageInDpf, preprocessing, subDirs = retrieveInfoLib(libPath, preprocessing = "curvature_portrait")
        for subDir in subDirs:
            print '-----------------------'
            print 'preprocessing ', subDir

            group = subDir.split('_')[1]
            ''' Path to video/s '''
            path = libPath + '/' + subDir
            videoPath = natural_sort([v for v in os.listdir(path) if isfile(path +'/'+ v)])[0]
            videoPath = path + '/' + videoPath
            paths = scanFolder(videoPath)
            name  = 'preprocessing_' + subDir
            segment(paths, name, numAnimals,
                        bkgSubstraction, selectROI, EQ,
                        minThreshold, maxThreshold,
                        minArea, maxArea)
            info = loadFile(paths[0], 'videoInfo', time=0)
            centers = info['ROICenters']
            assignCenters(paths,centers)
            # play(paths)
            portrait(paths)
            # fragment(paths)


    libLoop(libPath, numAnimals,
                bkgSubstraction, selectROI, EQ,
                minThreshold, maxThreshold,
                minArea, maxArea)

if __name__ == '__main__':

    libPath = '../data/library/25dpf'

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default = libPath, type = str)
    parser.add_argument('--folder_name', default = '', type = str)
    parser.add_argument('--bkg_subtraction', default = 0, type = int)
    parser.add_argument('--ROI_selection', default = 1, type = int)
    parser.add_argument('--mask_frame', default = 1, type= int)
    parser.add_argument('--Eq_image', default = 0, type = int)
    parser.add_argument('--min_th', default = 150, type = int)
    parser.add_argument('--max_th', default = 255, type = int)
    parser.add_argument('--min_area', default = 100, type = int)
    parser.add_argument('--max_area', default = 600, type = int)
    parser.add_argument('--num_animals', default = 4, type = int)
    args = parser.parse_args()

    ''' Parameters for the segmentation '''
    numAnimals = args.num_animals
    bkgSubstraction = args.bkg_subtraction
    selectROI = args.ROI_selection
    EQ = args.Eq_image
    minThreshold = args.min_th
    maxThreshold = args.max_th
    minArea = args.min_area # in pixels
    maxArea = args.max_area # in pixels


    buildLibrary(libPath,numAnimals,
                    bkgSubstraction, selectROI, EQ,
                    minThreshold, maxThreshold,
                    minArea, maxArea)
