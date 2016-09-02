import cv2
import sys
sys.path.append('../utils')
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


def computeIntersection(pixelsA,pixelsB):
    """
    pixels A (B): linear indices of blob A (B)
    """
    intersect = False
    if len(np.intersect1d(pixelsA,pixelsB)) > 0:
        intersect = True
    return intersect

def computeFrameIntersection(pixelsFrameA,pixelsFrameB,numAnimals):
    """
    pixelsFrameA (B): list of pixels of the blobs detected on frame A (B)
    """
    permutation = np.nan
    combinations = itertools.product(range(numAnimals),range(numAnimals))
    s = []
    intersect = False
    trueFragment = False
    for combination in combinations:
        inter = computeIntersection(pixelsFrameA[combination[0]],pixelsFrameB[combination[1]])
        intersect += inter
        if inter:
            s.append(combination)
    if intersect == numAnimals:
        trueFragment = True
        permutation = np.asarray(sorted(s, key=lambda x: x[1]))[:,0]
    return trueFragment, permutation

def computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals, numSegment):

    def storeFragmentIndices(SE, SEs, i):
        if len(SE) == 1:
            SE.append(i-1)
            SEs.append(SE)
        counter = 1
        SE = []
        return SE, SEs, counter

    SEs = [] # append SE^i for every segment
    SE = [] # store the indices of fragments in the form SE^1 = [[s_1^1,e_1^1], ..., [s_n^1, e_n^1]] for every segment
    counter = 1
    df = pd.DataFrame(columns=['permutation'])
    for i in range(1,len(columnPixels)):
        if (columnNumBlobs[i-1] == numAnimals and columnNumBlobs[i] == numAnimals):
            trueFragment, s = computeFrameIntersection(columnPixels[i-1],columnPixels[i],numAnimals)
            if trueFragment:
                if counter == 1:
                    df.loc[i-1,'permutation'] = np.arange(numAnimals)
                    if i-1 == 0:
                        SE.append(np.nan)
                    else:
                        SE.append(i-1)
                counter += 1
                df.loc[i,'permutation'] = df.loc[i-1,'permutation'][s]
            else:
                SE, SEs, counter = storeFragmentIndices(SE, SEs, i)

            if trueFragment and i == len(columnPixels)-1:
                SE, SEs, counter = storeFragmentIndices(SE, SEs, np.nan)
        else:
            SE, SEs, counter = storeFragmentIndices(SE, SEs, i)
    return df, SEs

def fragmentator(path):
    print 'Fragmenting video %s' % path
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    numSegment = int(filename.split('_')[-1])
    df = pd.read_pickle(path)
    columnNumBlobs = df.loc[:,'numberOfBlobs']
    columnPixels = df.loc[:,'pixels']
    numAnimals = 5
    dfPermutations, fragmentsIndices = computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals, numSegment)
    fragmentsIndices = (numSegment, fragmentsIndices)
    df['permutation'] = dfPermutations
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    df.to_pickle(folder +'/'+ filename + '.pkl')
    return fragmentsIndices

def segmentJoiner(paths,fragmentsIndices,numAnimals):
    # init first segment
    df = pd.read_pickle(paths[0])
    fragmentsIndicesA = fragmentsIndices[0][1]
    permutationA = df.iloc[-1]['permutation']
    pixelsA = df.iloc[-1]['pixels']
    numFramesA = len(df)
    if isinstance(fragmentsIndicesA[0][0],float):
        fragmentsIndicesA[0][0] = 0

    globalFrameCounter = numFramesA
    globalFragments = fragmentsIndicesA[:-1]

    for i in range(1,len(paths)):
        print 'Joining segment %s with %s ' % (paths[i-1], paths[i])
        # current segment
        df = pd.read_pickle(paths[i])
        fragmentsIndicesB = np.add(fragmentsIndices[i][1],globalFrameCounter).tolist()
        permutationB = df.iloc[0]['permutation']
        pixelsB = df.iloc[0]['pixels']
        numFramesB = len(df)

        if isinstance(permutationA,float): # if the last frame of the previous segment is not good (it is NaN)
            globalFragments.append(fragmentsIndicesA[-1])
            if isinstance(fragmentsIndicesB[0][0],float):
                fragmentsIndicesB[0][0] = globalFrameCounter

            globalFragments += fragmentsIndicesB[:-1]
            fragmentsIndicesA = fragmentsIndicesB
            permutationA = df.iloc[-1]['permutation'] # I save the permutation of the last frame of the current segment
            pixelsA = df.iloc[-1]['pixels']
            numFramesA = numFramesB
            globalFrameCounter += numFramesA
        else: # if the last frame of the previous segment is good
            if (len(pixelsA) == numAnimals and len(pixelsB) == numAnimals and not isinstance(df.loc[0,'permutation'],float)):
                trueFragment, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)

                if trueFragment:
                    newFragment = [fragmentsIndicesA[-1][0],fragmentsIndicesB[0][1]]
                    globalFragments.append(newFragment)
                    globalFragments += fragmentsIndicesB[1:-1]
                    # update permutations if they join
                    df.set_value(0,'permutation',permutationA[s])
                    counter = 1

                    while (not isinstance(df.loc[counter,'permutation'],float) and counter<len(df)):
                        pixelsA = df.loc[counter-1,'pixels']
                        pixelsB = df.loc[counter,'pixels']
                        indivA = df.loc[counter-1, 'permutation']
                        indivB = df.loc[counter, 'permutation']
                        trueFragment, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
                        df.set_value(counter,'permutation',indivA[s])
                        counter += 1
                else:
                    fragmentsIndicesA[-1][1] = globalFrameCounter-1
                    globalFragments.append(fragmentsIndicesA[-1])
                    fragmentsIndicesB[0][0] = globalFrameCounter + 1
                    globalFragments += fragmentsIndicesB[:-1]

            # update segment A
            fragmentsIndicesA = fragmentsIndicesB
            permutationA = df.iloc[-1]['permutation'] # I save the permutation of the last frame of the current segment
            pixelsA = df.iloc[-1]['pixels']
            numFramesA = numFramesB
            globalFrameCounter += numFramesA
            #save
            video = os.path.basename(paths[i])
            filename, extension = os.path.splitext(video)
            folder = os.path.dirname(paths[i])
            df.to_pickle(folder +'/'+ filename + '.pkl')

    if isinstance(fragmentsIndicesB[-1][1],float):
        fragmentsIndicesB[-1][1] = globalFrameCounter

    globalFragments.append(fragmentsIndicesB[-1])
    globalFragments = [map(int,globalFragment) for globalFragment in globalFragments]
    globalFragments = sorted(globalFragments, key=lambda x: x[1]-x[0],reverse=True)
    ### to be changed in the parallel version of this function
    filename = folder +'/'+ filename.split('_')[0] + '_segments.pkl'
    pickle.dump(globalFragments, open(filename, 'wb'))

if __name__ == '__main__':
    paths = scanFolder('./Cafeina5peces/Caffeine5fish_20140206T122428_1.pkl')

    # for path in paths:
    #     fragmentator(path)
    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    fragmentsIndices = Parallel(n_jobs=num_cores)(delayed(fragmentator)(path) for path in paths)
    fragmentsIndices = sorted(fragmentsIndices, key=lambda x: x[0])
    numAnimals = 5
    globalFragments = segmentJoiner(paths, fragmentsIndices, numAnimals)

    """
    IdInspector
    """
    numSegment = 0
    paths = scanFolder('./Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
    path = paths[numSegment]

    def IdPlayer(path):
        video = os.path.basename(path)
        filename, extension = os.path.splitext(video)
        sNumber = int(filename.split('_')[-1])
        folder = os.path.dirname(path)
        df = pd.read_pickle(folder +'/'+ filename + '.pkl')
        print 'Segmenting video %s' % path
        cap = cv2.VideoCapture(path)
        numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        def onChange(trackbarValue):
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
            centroids = df.loc[trackbarValue,'centroids']
            pixels = df.loc[trackbarValue,'pixels']
            permutation = df.loc[trackbarValue,'permutation']
            print 'previous frame, ', str(trackbarValue-1), ', permutation, ', df.loc[trackbarValue-1,'permutation']
            print 'current frame, ', str(trackbarValue), ', permutation, ', permutation
            # if sNumber == 1 and trackbarValue > 100:
            #     trueFragment, s = computeFrameIntersection(df.loc[trackbarValue-1,'pixels'],df.loc[trackbarValue,'pixels'],5)
            #     print trueFragment, s
            #     result = df.loc[trackbarValue-1,'permutation'][s]
            #     print 'result, ', result
            #Get frame from video file
            ret, frame = cap.read()
            #Color to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Plot segmentated blobs
            for i, pixel in enumerate(pixels):
                px = np.unravel_index(pixel,(height,width))
                frame[px[0],px[1]] = 255

            # plot numbers if not crossing
            if not isinstance(permutation,float):
                # print 'pass'
                for i, centroid in enumerate(centroids):
                    cv2.putText(frame,str(permutation[i]) + '-' + str(i),centroid, font, 1,0)

            cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))

            # Visualization of the process
            cv2.imshow('IdPlayer',frame)
            pass

        cv2.namedWindow('IdPlayer')
        cv2.createTrackbar( 'start', 'IdPlayer', 0, numFrame-1, onChange )
        # cv2.createTrackbar( 'end'  , 'IdPlayer', numFrame-1, numFrame, onChange )

        onChange(1)
        cv2.waitKey()

        start = cv2.getTrackbarPos('start','IdPlayer')

        return raw_input('Which segment do you want to inspect?')

    finish = False
    while not finish:
        print 'I am here', numSegment
        numSegment = IdPlayer(paths[int(numSegment)])
        if numSegment == 'q':
            finish = True
