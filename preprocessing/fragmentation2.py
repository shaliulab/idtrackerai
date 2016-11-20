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
import math

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
    INPUT
    pixelsFrameA (B): list of pixels of the blobs detected on frame A (B)
    numAnimals: number of animals in the video

    OUTPUT
    trueFragment: True when each animal from frame A overlap with only one animal in frame B
    s: permutation that needs to be applied to miniFrames for the identities in A and B to be conserved
    """
    # thArea = 1000
    # print '*********** NUM ANIMALS **********, ', numAnimals
    numAnimalsA = len(pixelsFrameA)
    numAnimalsB = len(pixelsFrameB)
    combinations = itertools.product(range(numAnimalsA),range(numAnimalsB)) # compute all the pairwise posibilities for two given number of animals
    s = []
    intersect = False
    trueFragment = False
    overlapMat = np.matrix(np.zeros((numAnimalsA,numAnimalsB)))
    for combination in combinations:
        inter = computeIntersection(pixelsFrameA[combination[0]],pixelsFrameB[combination[1]])
        overlapMat[combination] = inter

    rows = overlapMat.nonzero()[0]
    # print 'rows, ', rows
    cols = overlapMat.nonzero()[1]
    # print 'cols, ', cols
    possibleCombinations = zip(rows,cols)
    s = []
    for possibleCombination in possibleCombinations:
        # print 'possibleCombination, ', possibleCombination
        # print sum(possibleCombination[0]==rows) == 1
        # print sum(possibleCombination[1]==cols) == 1
        # print len(pixelsFrameB[possibleCombination[1]]) < thArea
        # print len(pixelsFrameA[possibleCombination[0]]) < thArea
        # if (sum(possibleCombination[0]==rows) == 1 and sum(possibleCombination[1]==cols) ==1) and \
        #     len(pixelsFrameB[possibleCombination[1]]) < thArea and len(pixelsFrameA[possibleCombination[0]]) < thArea:
        if (sum(possibleCombination[0]==rows) == 1 and sum(possibleCombination[1]==cols) ==1):
            s.append(possibleCombination)
    # print len(s)
    # print type(numAnimals)
    if len(s) == numAnimals:
        trueFragment = True

    return trueFragment, s, overlapMat

def applyPermutation(maxNumBlobs, permutation, old, missing):
    newIds = np.multiply(np.ones(maxNumBlobs, dtype='int'),-1)
    count = 0
    for (s,p) in permutation:
        if old[s]== -1:
            newIds[p] = missing[count]
            old[s] = missing[count] ### FIXME this assigns the correct identity to the blob of the previous frame that overlaped properly with the current one, however it creates a problem in the fragments. The first frame of the fragment sometimes does not have the number of animals that it should.
            count += 1
        else:
            newIds[p] = old[s]
    return newIds, old


def computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals,maxNumBlobs):
    """
    Given number of blobs in each frame and the pixels of each blob this function
    return a list of fragments for identification (parts of the video where the animals do not cross)
    and the permutations that need to be applied to the miniFrames to identify
    and animal by overlapping.

    INPUT
    columnNumBlobs: column with number of blobs for each frame in the segment
    columnPixes: list of numAnimals-lists each list has the pixels of each blob in the frame
    numAnimals: number of animals in the video

    OUTPUT
    df: DataFrame with the permutations for each frame
    SEs: list of fragments for the given segment. Each fragment is a 2 element list
    with the starting and ending frame of the fragment. The framecounter is with respect
    to the segment.
    """

    def storeFragmentIndices(SE, SEs, i):
        """
        Adds the last frame index (i) of the fragment (i),
        appends it to list of fragments SEs and initialize the new fragment

        INPUT
        SE: opened fragment (it only has the starting frame index)
        SEs: list of previous fragments
        """
        if len(SE) == 1: # the fragment is opened
            SE.append(i-1)
            SEs.append(SE)
        SE = [] # we initialize the next fragment
        return SE, SEs

    SEs = [] # list the indices of fragments in the form SEs = [[s_1^1,e_1^1], ..., [s_n^1, e_n^1]] for every segment
    SE = []
    counter = 1
    df = pd.DataFrame(index = range(len(columnPixels)), columns=['permutation'])

    # print np.multiply(np.ones(maxBlobs, dtype='int'),-1)
    init = np.multiply(np.ones(maxNumBlobs, dtype='int'),-1)
    # print columnNumBlobs[0]
    # print '------------------------------------'
    init[:int(columnNumBlobs[0])] = np.arange(int(columnNumBlobs[0]))
    df.loc[0,'permutation'] = init

    for i in range(1,len(columnPixels)): # for every frame
        # print 'Frame ---------------------------------------, ', i
        trueFragment, permutation, overlapMat = computeFrameIntersection(columnPixels[i-1],columnPixels[i],numAnimals) # compute overlapping between blobs
        # print 'trueFragment, ', trueFragment
        # print 'overlapMat, '
        # print overlapMat
        # print 'permutation, ', permutation
        old = df.loc[i-1,'permutation']

        cur_ind = set(old)
        all_ind = set(range(maxNumBlobs))
        missing = sorted(list(all_ind - cur_ind))
        # print 'missing individuals, ', missing

        df.loc[i, 'permutation'], df.loc[i-1,'permutation'] = applyPermutation(maxNumBlobs, permutation, old, missing)
        # print 'newID, ', df.loc[i, 'permutation']

        if trueFragment and len(SE)==0: # There is a good overlapping and it is a new fragment
            # print 'There is a good overlapping and it is a new fragment'
            # if missing and missing[-1] == numAnimals: # If all individuals are missing
            #     missing[-1] = -1

            # print 'ID (previous frame), ', df.loc[i-1,'permutation']
            # tofill = np.where(df.loc[i-1,'permutation'] == -1)[0][:len(missing)] #which individuals where not good in the previous frame
            # df.loc[i-1,'permutation'][tofill] = missing # We give new index to the individuals that where not in the previous frame (because they were crossin or dissapeared)
            # print 'newID (previous frame), ', df.loc[i-1,'permutation']
            if i-1 == 0: # If it is the first frame of the segment we set a Nan because we do not know what happend between this one and the previous segment
                SE.append(np.nan)
            else:
                # print '******* Appending first frame to the fragment *******'
                SE.append(i-1)
        elif not trueFragment:
            # print 'There is not a good overlapping, we close the fragment and initialize a new one'
            # print 'Fragment, ', SE
            # print 'All fragments so far, ', SEs
            SE, SEs = storeFragmentIndices(SE, SEs, i)
        # print trueFragment, i, len(columnPixels)-1
        if trueFragment and i == len(columnPixels)-1:
            # print 'This is the last frame of the segment, we close the fragment and put a NaN at the end'
            SE, SEs = storeFragmentIndices(SE, SEs, np.nan)
    # print '1111111111111111111111111111111111111111111111111111111'
    return df, SEs

def fragmentator(path, numAnimals, maxNumBlobs):
    print '-----------------------------------'
    print 'Fragmenting video %s' % path
    df, numSegment = loadFile(path, 'segmentation', time=0)
    numSegment = int(numSegment)
    columnNumBlobs = df.loc[:,'numberOfBlobs']
    columnPixels = df.loc[:,'pixels']

    dfPermutations, fragmentsIndices = computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals,maxNumBlobs)
    print 'fragmentsIndices (in fragmentator),', fragmentsIndices
    fragmentsIndices = (numSegment, fragmentsIndices)
    df['permutation'] = dfPermutations

    saveFile(path, df, 'segment', time = 0)
    return fragmentsIndices

def segmentJoiner(paths,fragmentsIndices,numAnimals, maxNumBlobs):
    """
    paths = videoPaths
    fragmentsIndices: list of tuples (numSegment, List_of_fragments_indices). We refer to complete fragments where all individuals are separated numBlobs = numAnimals
    List_of_fragments_indices: list of pairs of numbers [first frame fragment, last frame fragment]
    """
    # init first segment
    maxBlobs = maxNumBlobs
    df,_ = loadFile(paths[0], 'segmentation', time=0)
    fragmentsIndicesA = fragmentsIndices[0][1]
    permutationA = df.iloc[-1]['permutation'] # permutation in last frame of segmA
    pixelsA = df.iloc[-1]['pixels'] # list of pixels of the blobs in last frame of segmA
    numFramesA = len(df) # numbers of frames in segmA
    numBlobsA = len(pixelsA) # number of blobs in last frame of segmA
    if isinstance(fragmentsIndicesA[0][0],float):
        # The first fragment of the video might have a NaN if the first frame is good, otherwise it will ahve the number of the first good frame
        fragmentsIndicesA[0][0] = 0

    globalFrameCounter = numFramesA # We set a counter for the total number of frames in the video
    # We cannot assume that every segment will have the same amout of frames. If the segments come directly from the camera
    # it might have lose frames
    globalFragments = fragmentsIndicesA[:-1] # All the fragments in segmA but the last ones form the initialization of the list of fragments for the whole video globalFragments

    for i in range(1,len(paths)): # the loop starts from the second segment
        print '\n Joining segment %s with %s ' % (paths[i-1], paths[i])

        # current segment
        df,_ = loadFile(paths[i], 'segmentation', time=0)
        fragmentsIndicesB = np.add(fragmentsIndices[i][1],globalFrameCounter).tolist()
        permutationB = df.iloc[0]['permutation']
        pixelsB = df.iloc[0]['pixels']
        numFramesB = len(df)
        numBlobsB = len(pixelsB)

        # permutation if a list of indices from 0 to numAminals-1 and -1.
        # if there are not as many positive numbers as numAnimals, it means that in the last frame
        # there was a crossing or there was an extra blob. This means that the last frame of segmA is not included
        # in any fragment and it does not make sense trying to join the last complete fragment of segmA with the first
        # complete fragment of segmB. However it still make sense propagating the permutation for the individual fragments.
        if sum(permutationA >= 0) != numAnimals: # if the last frame of the previous segment is not good (the permutation is NaN)
            print 'they cannot join'
            globalFragments.append(fragmentsIndicesA[-1])
            if math.isnan(fragmentsIndicesB[0][0]):
                fragmentsIndicesB[0][0] = globalFrameCounter

            globalFragments += fragmentsIndicesB[:-1]

        else: # if the last frame of the previous segment is good ()
            print permutationB, numBlobsB, sum(permutationB >= 0)
            # print '===================================================='
            if ((numBlobsA == numAnimals and numBlobsB == numAnimals) and  sum(permutationB >= 0) == numAnimals):
                trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
                print 'they could join'
                if trueFragment:
                    print 'they join'
                    newFragment = [fragmentsIndicesA[-1][0],fragmentsIndicesB[0][1]]

                    globalFragments.append(newFragment)
                    globalFragments += fragmentsIndicesB[1:-1]

                else:
                    print 'they do not join'
                    fragmentsIndicesB[0][0] = globalFrameCounter + 1
                    globalFragments += fragmentsIndicesB[:-1]
                    fragmentsIndicesA[-1][1] = globalFrameCounter-1
                    globalFragments.append(fragmentsIndicesA[-1])

        # update permutations
        print 'Updating permutations...'
        trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
        cur_ind = set(permutationA)
        all_ind = set(range(maxBlobs))
        missing = sorted(list(all_ind - cur_ind))

        df.set_value(0,'permutation',applyPermutation(maxBlobs, permutation, permutationA, missing)[0])
        counter = 1
        # while (sum(df.loc[counter,'permutation'] >= 0) == df.loc[counter,'numberOfBlobs'] and counter<len(df)):
        while counter<len(df):
            # print counter
            # print 'Progapagating permutation...'
            pixelsA = df.loc[counter-1,'pixels']
            pixelsB = df.loc[counter,'pixels']
            indivA = df.loc[counter-1, 'permutation']
            indivB = df.loc[counter, 'permutation']

            trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
            cur_ind = set(indivA)
            all_ind = set(range(maxBlobs))
            missing = sorted(list(all_ind - cur_ind))

            df.set_value(counter,'permutation',applyPermutation(maxBlobs, permutation, indivA, missing)[0])
            counter += 1

        fragmentsIndicesA = fragmentsIndicesB
        permutationA = df.iloc[-1]['permutation'] # I save the permutation of the last frame of the current segment
        pixelsA = df.iloc[-1]['pixels']
        numFramesA = numFramesB
        globalFrameCounter += numFramesA
        saveFile(paths[i], df, 'segment', time = 0)

    if isinstance(fragmentsIndicesB[-1][1],float):
        fragmentsIndicesB[-1][1] = globalFrameCounter

    globalFragments.append(fragmentsIndicesB[-1])
    print 'Global fragments, ', globalFragments
    globalFragments = [map(int,globalFragment) for globalFragment in globalFragments]
    globalFragments = sorted(globalFragments, key=lambda x: x[1]-x[0],reverse=True)
    saveFile(paths[0], globalFragments, 'fragments', time = 0)
    return globalFragments

def fragment(paths):
    info = loadFile(paths[0], 'videoInfo', time=0)
    info = info.to_dict()[0]
    width = info['width']
    height = info['height']
    numAnimals = int(info['numAnimals'])
    maxNumBlobs = info['maxNumBlobs']

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    fragmentsIndices = Parallel(n_jobs=num_cores)(delayed(fragmentator)(path, numAnimals, maxNumBlobs) for path in paths)
    # print 'fragmentIndices (before sorting), ', fragmentsIndices
    fragmentsIndices = sorted(fragmentsIndices, key=lambda x: x[0])
    # print fragmentsIndices
    print 'fragmentsIndices (before joiningSegments), ', fragmentsIndices
    globalFragments = segmentJoiner(paths, fragmentsIndices, numAnimals, maxNumBlobs)
    print globalFragments
