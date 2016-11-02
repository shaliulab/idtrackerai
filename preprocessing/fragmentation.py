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
    thArea = 5000
    # print '*********** NUM ANIMALS **********, ', numAnimals
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
        if (sum(possibleCombination[0]==rows) == 1 and sum(possibleCombination[1]==cols) ==1) and \
            len(pixelsFrameB[possibleCombination[1]]) < thArea and len(pixelsFrameA[possibleCombination[0]]) < thArea:
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
            count += 1
        else:
            newIds[p] = old[s]
    return newIds


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
        missing = list(all_ind - cur_ind)
        # print 'missing individuals, ', missing

        df.loc[i, 'permutation'] = applyPermutation(maxNumBlobs, permutation, old, missing)
        # print 'newID, ', df.loc[i, 'permutation']

        if trueFragment and len(SE)==0: # There is a good overlapping and it is a new fragment
            # print 'There is a good overlapping and it is a new fragment'
            if missing and missing[-1] == numAnimals: # If all individuals are missing
                missing[-1] = -1

            tofill = np.where(df.loc[i-1,'permutation'] == -1)[0][:len(missing)] #which individuals where not good in the previous frame
            df.loc[i-1,'permutation'][tofill] = missing # We give new index to the individuals that where not in the previous frame (because they were crossin or dissapeared)

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
        if trueFragment and i == len(columnPixels)-1:
            # print 'This is the last frame of the segment, we close the fragment and put a NaN at the end'
            SE, SEs = storeFragmentIndices(SE, SEs, np.nan)
    return df, SEs

def fragmentator(path, numAnimals, maxNumBlobs):
    print 'Fragmenting video %s' % path
    df, numSegment = loadFile(path, 'segmentation', time=0)
    numSegment = int(numSegment)
    columnNumBlobs = df.loc[:,'numberOfBlobs']
    columnPixels = df.loc[:,'pixels']


    dfPermutations, fragmentsIndices = computeFragmentOverlap(columnNumBlobs, columnPixels, numAnimals,maxNumBlobs)
    fragmentsIndices = (numSegment, fragmentsIndices)
    df['permutation'] = dfPermutations

    saveFile(path, df, 'segment', time = 0)
    return fragmentsIndices

def segmentJoiner(paths,fragmentsIndices,numAnimals, maxNumBlobs):
    # init first segment
    maxBlobs = maxNumBlobs
    df,_ = loadFile(paths[0], 'segmentation', time=0)
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
        df,_ = loadFile(paths[i], 'segmentation', time=0)
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
            saveFile(paths[i], df, 'segment', time = 0)
            # video = os.path.basename(paths[i])
            # filename, extension = os.path.splitext(video)
            # folder = os.path.dirname(paths[i])
            # df.to_pickle(folder +'/'+ filename + '.pkl')

    if isinstance(fragmentsIndicesB[-1][1],float):
        # print 'pass last condition'
        fragmentsIndicesB[-1][1] = globalFrameCounter

    globalFragments.append(fragmentsIndicesB[-1])
    print 'Global fragments, ', globalFragments
    globalFragments = [map(int,globalFragment) for globalFragment in globalFragments]
    globalFragments = sorted(globalFragments, key=lambda x: x[1]-x[0],reverse=True)
    ### to be changed in the parallel version of this function
    saveFile(paths[0], globalFragments, 'fragments', time = 0)
    # filename = folder +'/'+ filename.split('_')[0] + '_segments.pkl'
    # pickle.dump(globalFragments, open(filename, 'wb'))

    return globalFragments

def fragment(paths):
    info = loadFile(paths[0], 'videoInfo', time=0)
    width = info['width']
    height = info['height']
    numAnimals = int(info['numAnimals'])
    maxNumBlobs = info['maxNumBlobs']

    num_cores = multiprocessing.cpu_count()
    num_cores = 1
    fragmentsIndices = Parallel(n_jobs=num_cores)(delayed(fragmentator)(path, numAnimals, maxNumBlobs) for path in paths)
    fragmentsIndices = sorted(fragmentsIndices, key=lambda x: x[0])
    # print fragmentsIndices

    globalFragments = segmentJoiner(paths, fragmentsIndices, numAnimals, maxNumBlobs)
    # print globalFragments

# def playFragmentation(paths):
#     """
#     IdInspector
#     """
#     info = loadFile(paths[0], 'videoInfo', time=0)
#     width = info['width']
#     height = info['height']
#     numAnimals = info['numAnimals']
#     maxNumBlobs = info['maxNumBlobs']
#     numSegment = 0
#     # paths = scanFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi')
#     # paths = scanFolder('../Conflict8/conflict3and4_20120316T155032_1.avi') #'../Conflict8/conflict3and4_20120316T155032_1.pkl'
#     path = paths[numSegment]
#
#     def IdPlayerFragmentation(path,numAnimals, width, height):
#         df,sNumber = loadFile(path, 'segmentation', time=0)
#         # video = os.path.basename(path)
#         # filename, extension = os.path.splitext(video)
#         # sNumber = int(filename.split('_')[-1])
#         # folder = os.path.dirname(path)
#         # df = pd.read_pickle(folder +'/'+ filename + '.pkl')
#         print 'Visualizing video %s' % path
#         # print df
#         cap = cv2.VideoCapture(path)
#         numFrame = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
#         # width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
#         # height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
#
#         def onChange(trackbarValue):
#             cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,trackbarValue)
#             centroids = df.loc[trackbarValue,'centroids']
#             pixelsA = df.loc[trackbarValue-1,'pixels']
#             pixelsB = df.loc[trackbarValue,'pixels']
#             permutation = df.loc[trackbarValue,'permutation']
#             print '------------------------------------------------------------'
#             print 'previous frame, ', str(trackbarValue-1), ', permutation, ', df.loc[trackbarValue-1,'permutation']
#             print 'current frame, ', str(trackbarValue), ', permutation, ', permutation
#             trueFragment, s, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals)
#             print 'overlapMat, '
#             print overlapMat
#             print 'permutation, ', s
#             # if sNumber == 1 and trackbarValue > 100:
#             #     trueFragment, s = computeFrameIntersection(df.loc[trackbarValue-1,'pixels'],df.loc[trackbarValue,'pixels'],5)
#             #     print trueFragment, s
#             #     result = df.loc[trackbarValue-1,'permutation'][s]
#             #     print 'result, ', result
#             #Get frame from video file
#             ret, frame = cap.read()
#             #Color to gray scale
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#
#             # Plot segmentated blobs
#             for i, pixel in enumerate(pixelsB):
#                 px = np.unravel_index(pixel,(height,width))
#                 frame[px[0],px[1]] = 255
#
#             # plot numbers if not crossing
#             # if not isinstance(permutation,float):
#                 # print 'pass'
#             for i, centroid in enumerate(centroids):
#                 cv2.putText(frame,'i'+ str(permutation[i]) + '|h' +str(i),centroid, font, .7,0)
#
#             cv2.putText(frame,str(trackbarValue),(50,50), font, 3,(255,0,0))
#
#             # Visualization of the process
#             cv2.imshow('IdPlayerFragmentation',frame)
#             pass
#
#         cv2.namedWindow('IdPlayerFragmentation')
#         cv2.createTrackbar( 'start', 'IdPlayerFragmentation', 0, numFrame-1, onChange )
#         # cv2.createTrackbar( 'end'  , 'IdPlayer', numFrame-1, numFrame, onChange )
#
#         onChange(1)
#         cv2.waitKey(0)
#
#         start = cv2.getTrackbarPos('start','IdPlayerFragmentation')
#         return raw_input('Which segment do you want to inspect?')
#
#     finish = False
#     while not finish:
#         # print 'I am here', numSegment
#         numSegment = IdPlayerFragmentation(paths[int(numSegment)],numAnimals, width, height)
#         if numSegment == 'q':
#             finish = True
#     cv2.waitKey(1)
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)




# if __name__ == '__main__':
#
#     # videoPath = '../Conflict8/conflict3and4_20120316T155032_1.avi'
#     videoPath = '../data/library/33dpf/group_1/group_1.avi'
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', default = videoPath, type = str)
#     args = parser.parse_args()
#     paths = scanFolder(args.path)
#     fragment(paths)
#     play(paths)
