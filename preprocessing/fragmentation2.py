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
    permutation: permutation that needs to be applied to miniFrames for the identities in A and B to be conserved
    """
    thArea = 750
    # thArea = 5
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
    cols = overlapMat.nonzero()[1]
    possibleCombinations = zip(rows,cols)
    s = []
    for possibleCombination in possibleCombinations:
        if (sum(possibleCombination[0]==rows) == 1 and sum(possibleCombination[1]==cols) ==1) and \
            len(pixelsFrameB[possibleCombination[1]]) < thArea and len(pixelsFrameA[possibleCombination[0]]) < thArea:
                s.append(possibleCombination)
    if len(s) == numAnimals:
        trueFragment = True

    return trueFragment, s, overlapMat

# numAnimals = 3
# # optimal overlapping
# print '---Optimal overlapping---'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[2,3,4],[5,6,7],[10,11,12]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # bad overlapping 1
# print '---Bad overlapping 1: blob from B overlaps with two from A---'
# pixelsA = [[2,3,4],[6,7,8],[11,12,13]]
# pixelsB = [[1,2,3],[5,6,7],[8,9,10]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # bad overlapping 2
# print '---Bad overlapping 2: blob from A overlaps with two from B---'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[2,3,4],[6,7,8],[11,12,13]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # Cross in
# print '---Cross in---'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[2,3,4],[5,6,7,10,11,12]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # Cross out
# print '---Cross out--'
# pixelsA = [[2,3,4],[5,6,7,10,11,12]]
# pixelsB = [[1,2,3],[5,6,7],[8,9,10]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # After crossing
# print '---After crossing---'
# pixelsA = [[2,3,4],[5,6,7,10,11,12]]
# pixelsB = [[1,2,3],[5,6,7,10,11,12]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # Jump
# print '---Jump---'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[2,3,4],[5,6,7],[11,12,13]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# # Cross with shadow
# print '---Cross with shadow---'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[2,3,4],[6,7,8],[10,11,12]]
# trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print 'trueFragment, ', trueFragment
# print 'overlapMat ,'
# print overlapMat
# print 'permutation, ', permutation
# print '*********************************'
# # optimal overlapping
# print 'Optimal overlapping'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[5,6,7],[2,3,4],[10,11,12]]
# trueFragment, permutation, overlapMat1, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat1
# print s
# print permutation
# # bad overlapping 1
# print 'Bad overlapping 1: blob from B overlaps with two from A'
# pixelsA = [[2,3,4],[6,7,8],[11,12,13]]
# pixelsB = [[5,6,7],[1,2,3],[8,9,10]]
# trueFragment, permutation, overlapMat2, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat2
# print s
# print permutation
# # bad overlapping 2
# print 'Bad overlapping 2: blob from A overlaps with two from B'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[6,7,8],[2,3,4],[11,12,13]]
# trueFragment, permutation, overlapMat3, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat3
# print s
# print permutation
# # Cross in
# print 'Cross in'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[5,6,7,10,11,12],[2,3,4]]
# trueFragment, permutation, overlapMat4, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat4
# print s
# print permutation
# # Cross out
# print 'Cross out'
# pixelsA = [[2,3,4],[5,6,7,10,11,12]]
# pixelsB = [[5,6,7],[1,2,3],[8,9,10]]
# trueFragment, permutation, overlapMat5, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat5
# print s
# print permutation
# # Jump
# print 'Jump'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# pixelsB = [[5,6,7],[2,3,4],[11,12,13]]
# trueFragment, permutation, overlapMat6, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat6
# print s
# print permutation
# # Cross with shadow
# print 'Cross with shadow'
# pixelsA = [[1,2,3],[5,6,7],[8,9,10]]
# # pixelsB = [[6,7,8],[2,3,4],[10,11,12]]
# pixelsB = [[1,2,3],[10,11,12],[6,7,8]]
# trueFragment, permutation, overlapMat7, s = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
# print overlapMat7
# print s
# print permutation

def applyPermutation(maxNumBlobs, permutation, old, missing):
    newIds = np.multiply(np.ones(maxNumBlobs, dtype='int'),-1)
    count = 0
    for (s,p) in permutation:
        if old[s]== -1:
            newIds[p] = missing[count]
            count += 1
        else:
            newIds[p] = old[s]
    return newIds


def computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals, numSegment,maxNumBlobs):
    """
    Given number of blobs in each frame and the pixels of each blob this function
    return a list of fragments for identification (parts of the video where the animals do not cross)
    and the permutations that need to be applied to the miniFrames to identify
    and animal by overlapping.

    INPUT
    columnNumBlobs: column with number of blobs for each frame in the segment
    columnPixes: list of numAnimals-lists each list has the pixels of each blob in the frame
    numAnimals: number of animals in the video
    numSegment: segment number

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
        trueFragment, permutation, overlapMat = computeFrameIntersection(columnPixels[i-1],columnPixels[i],numAnimals) # compute overlapping between blobs

        old = df.loc[i-1,'permutation']

        cur_ind = set(old)
        all_ind = set(range(maxNumBlobs))
        missing = list(all_ind - cur_ind)

        df.loc[i, 'permutation'] = applyPermutation(maxNumBlobs, permutation, old, missing)

        if trueFragment and len(SE)==0:

            if missing and missing[-1] == numAnimals:
                missing[-1] = -1

            tofill = np.where(df.loc[i-1,'permutation'] == -1)[0][:len(missing)]
            df.loc[i-1,'permutation'][tofill] = missing

            if i-1 == 0:
                SE.append(np.nan)
            else:
                SE.append(i-1)
        elif not trueFragment:
            SE, SEs = storeFragmentIndices(SE, SEs, i)
        if trueFragment and i == len(columnPixels)-1:
            SE, SEs = storeFragmentIndices(SE, SEs, np.nan)

    return df, SEs

def fragmentator(path, numAnimals):
    print 'Fragmenting video %s' % path
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    numSegment = int(filename.split('_')[-1])
    df = pd.read_pickle(path)
    columnNumBlobs = df.loc[:,'numberOfBlobs']
    columnPixels = df.loc[:,'pixels']
    dfPermutations, fragmentsIndices = computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals, numSegment,maxNumBlobs)
    # print dfPermutations,fragmentsIndices
    fragmentsIndices = (numSegment, fragmentsIndices)
    df['permutation'] = dfPermutations
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    df.to_pickle(folder +'/'+ filename + '.pkl')
    return fragmentsIndices

def segmentJoiner(paths,fragmentsIndices,numAnimals, maxNumBlobs):
    # init first segment
    maxBlobs = maxNumBlobs
    df = pd.read_pickle(paths[0])
    fragmentsIndicesA = fragmentsIndices[0][1]
    permutationA = df.iloc[-1]['permutation']
    pixelsA = df.iloc[-1]['pixels']
    numFramesA = len(df)
    numBlobsA = len(pixelsA)
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
        numBlobsB = len(pixelsB)
        # print sum(permutationA >= 0), permutationA
        if sum(permutationA >= 0) != numAnimals: # if the last frame of the previous segment is not good (the permutation is NaN)
            # print 'i dont give a fuck of joining'
            globalFragments.append(fragmentsIndicesA[-1])
            if math.isnan(fragmentsIndicesB[0][0]):
                fragmentsIndicesB[0][0] = globalFrameCounter

            globalFragments += fragmentsIndicesB[:-1]
            fragmentsIndicesA = fragmentsIndicesB
            permutationA = df.iloc[-1]['permutation'] # I save the permutation of the last frame of the current segment
            pixelsA = df.iloc[-1]['pixels']
            numFramesA = numFramesB
            globalFrameCounter += numFramesA
        else: # if the last frame of the previous segment is good ()
            # print permutationB, numBlobsB, sum(permutationB >= 0)
            # print '===================================================='
            if ((numBlobsA == numAnimals and numBlobsB == numAnimals) and  sum(permutationB >= 0) == numAnimals):
                trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
                # print 'they can join'
                if trueFragment:
                    # print 'they join'
                    newFragment = [fragmentsIndicesA[-1][0],fragmentsIndicesB[0][1]]

                    globalFragments.append(newFragment)
                    globalFragments += fragmentsIndicesB[1:-1]
                    # update permutations if they join
                    cur_ind = set(permutationA)
                    all_ind = set(range(maxBlobs))
                    missing = list(all_ind - cur_ind)

                    df.set_value(0,'permutation',applyPermutation(maxBlobs, s, permutationA, missing))
                    counter = 1

                    # while (sum(df.loc[counter,'permutation'] >= 0) == df.loc[counter,'numberOfBlobs'] and counter<len(df)):
                    while counter<len(df):
                        # print counter
                        # print 'Progapagating permutation...'
                        pixelsA = df.loc[counter-1,'pixels']
                        pixelsB = df.loc[counter,'pixels']
                        indivA = df.loc[counter-1, 'permutation']
                        indivB = df.loc[counter, 'permutation']
                        # print numAnimals
                        # print len(pixelsA)
                        # print len(pixelsB)
                        trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
                        # if counter == 47:
                        #     print 'permutation, ', s
                        #     print 'overlapMat, ',
                        #     print overlapMat
                        # if len(s) != numAnimals:
                        #     break
                        cur_ind = set(indivA)
                        all_ind = set(range(maxBlobs))
                        missing = list(all_ind - cur_ind)

                        # df.loc[counter, 'permutation'] = applyPermutation(maxBlobs, permutation, indivA, missing)
                        df.set_value(counter,'permutation',applyPermutation(maxBlobs, permutation, indivA, missing))

                        if trueFragment and sum(indivA>=0) <= len(pixelsA):

                            if missing and missing[-1] == numAnimals:
                                missing[-1] = -1

                            tofill = np.where(indivA == -1)[0][:len(missing)]
                            df.loc[counter-1,'permutation'][tofill] = missing

                        # if counter == 47:
                        #     print df.loc[counter-1:counter+1]
                        # df.set_value(counter,'permutation',indivA[s])
                        counter += 1
                else:
                    fragmentsIndicesB[0][0] = globalFrameCounter + 1
                    globalFragments += fragmentsIndicesB[:-1]
                    fragmentsIndicesA[-1][1] = globalFrameCounter-1
                    globalFragments.append(fragmentsIndicesA[-1])

            # update segment A
            fragmentsIndicesA = fragmentsIndicesB
            permutationA = df.iloc[-1]['permutation'] # I save the permutation of the last frame of the current segment
            pixelsA = df.iloc[-1]['pixels']
            numFramesA = numFramesB
            numBlobsA = numBlobsB
            globalFrameCounter += numFramesA
            #save
            video = os.path.basename(paths[i])
            filename, extension = os.path.splitext(video)
            folder = os.path.dirname(paths[i])
            df.to_pickle(folder +'/'+ filename + '.pkl')

    if isinstance(fragmentsIndicesB[-1][1],float):
        print 'pass last condition'
        fragmentsIndicesB[-1][1] = globalFrameCounter

    globalFragments.append(fragmentsIndicesB[-1])
    globalFragments = [map(int,globalFragment) for globalFragment in globalFragments]
    globalFragments = sorted(globalFragments, key=lambda x: x[1]-x[0],reverse=True)
    ### to be changed in the parallel version of this function
    filename = folder +'/'+ filename.split('_')[0] + '_segments.pkl'
    pickle.dump(globalFragments, open(filename, 'wb'))

    return globalFragments

if __name__ == '__main__':
# if False: # used to test functions
    paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.pkl') # '../Conflict8/conflict3and4_20120316T155032_1.pkl'
    # Cafeina5peces/Caffeine5fish_20140206T122428_1.avi
    info = pd.read_pickle('../Cafeina5peces/Caffeine5fish_videoInfo.pkl')
    numAnimals = info['numAnimals']
    maxNumBlobs = info['maxNumBlobs']

    num_cores = multiprocessing.cpu_count()
    # num_cores = 1
    fragmentsIndices = Parallel(n_jobs=num_cores)(delayed(fragmentator)(path, numAnimals) for path in paths)
    fragmentsIndices = sorted(fragmentsIndices, key=lambda x: x[0])
    print fragmentsIndices

    globalFragments = segmentJoiner(paths, fragmentsIndices, numAnimals, maxNumBlobs)
    print globalFragments

    """
    IdInspector
    """
    numSegment = 0
    paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi') #'../Conflict8/conflict3and4_20120316T155032_1.pkl'
    path = paths[numSegment]

    def IdPlayer(path,numAnimals):
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
            pixelsA = df.loc[trackbarValue-1,'pixels']
            pixelsB = df.loc[trackbarValue,'pixels']
            permutation = df.loc[trackbarValue,'permutation']
            print '------------------------------------------------------------'
            print 'previous frame, ', str(trackbarValue-1), ', permutation, ', df.loc[trackbarValue-1,'permutation']
            print 'current frame, ', str(trackbarValue), ', permutation, ', permutation
            trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
            print 'overlapMat, '
            print overlapMat
            print 'permutation, ', s
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
            for i, pixel in enumerate(pixelsB):
                px = np.unravel_index(pixel,(height,width))
                frame[px[0],px[1]] = 255

            # plot numbers if not crossing
            # if not isinstance(permutation,float):
                # print 'pass'
            for i, centroid in enumerate(centroids):
                cv2.putText(frame,'i'+ str(permutation[i]) + '|h' +str(i),centroid, font, .7,0)

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
        numSegment = IdPlayer(paths[int(numSegment)],numAnimals)
        if numSegment == 'q':
            finish = True
