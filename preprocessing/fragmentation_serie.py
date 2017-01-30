import cv2
import sys
sys.path.append('../utils')

from py_utils import *
from GUI_utils import *

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
    # if len(s) == numAnimals:
    if numAnimalsA == numAnimals and numAnimalsB == numAnimals and len(s) == numAnimals:
        trueFragment = True

    return trueFragment, s, overlapMat

def applyPermutation(maxNumBlobs, permutation, old, missing):
    newIds = np.multiply(np.ones(maxNumBlobs, dtype='int'),-1)
    count = 0
    for (s,p) in permutation:
        if old[s]== -1:
            newIds[p] = missing[count]
            # old[s] = missing[count] ### FIXME this assigns the correct identity to the blob of the previous frame that overlaped properly with the current one, however it creates a problem in the fragments. The first frame of the fragment sometimes does not have the number of animals that it should.
            ### This fucks the individual fragments when there is a jump or a a crosing with a shadow
            count += 1
        else:
            newIds[p] = old[s]
    return newIds, old

def newFragmentator(videoPaths,numAnimals,maxNumBlobs, numFrames):

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

    fragments = [] # list the indices of fragments in the form SEs = [[s_1^1,e_1^1], ..., [s_n^1, e_n^1]] for every segment
    fragment = []
    globalFrameCounter = 0
    dfGlobal = pd.DataFrame(index = range(numFrames), columns=['permutations','areas', 'centroids'])

    for j, path in enumerate(videoPaths):
        print '-----------------------------------'
        print 'Fragmenting video %s' % path
        df, numSegment = loadFile(path, 'segmentation', time=0)
        numFramesSegment = len(df)
        # print 'Num frames in segment, ', numFramesSegment
        columnNumBlobs = df.loc[:,'numberOfBlobs']
        columnPixels = df.loc[:,'pixels']

        if j == 0:
            dfPermutations = pd.DataFrame(index = range(len(columnPixels)), columns=['permutation'])

        for i in range(len(columnPixels)): # for every frame
            globalFrame = i + globalFrameCounter
            dfGlobal.loc[globalFrame, 'areas'] = df.loc[i,'areas']
            dfGlobal.loc[globalFrame, 'centroids'] = df.loc[i,'centroids']
            # print '*** segment frame, ', i, ', global frame, ', globalFrame
            if globalFrame != 0: # Becuase we look at the past (i-1 and i), the first frame of the first segment does not make sense
                # print 'it is not the first global frame '
                if globalFrame == 1:
                    # print 'it is the second global frame '
                    # If it is the first time we look at the past we need to define pixelsA and the previous permutation,
                    # in the next iterations they will be defined recursively form pixelsB and from the permutation in i
                    pixelsA = columnPixels[i-1]
                    init = np.multiply(np.ones(maxNumBlobs, dtype='int'),-1)
                    init[:int(columnNumBlobs[0])] = np.arange(int(columnNumBlobs[0]))
                    # dfGlobal.set_value(0,'permutation',init)
                    # print 'init, ', init
                    dfGlobal.loc[0,'permutations'] = init
                    old = dfGlobal.loc[0,'permutations']

                    dfPermutations.loc[0,'permutation'] = init

                pixelsB = columnPixels[i]
                trueFragment, permutation, overlapMat = computeFrameIntersection(pixelsA,pixelsB,numAnimals) # compute overlapping between blobs
                # print 'numBlobsA', len(pixelsA)
                # print 'numBlobsB', len(pixelsB)
                # print 'trueFragment, ', trueFragment
                # print 'overlapMat, '
                # print overlapMat
                # print 'permutation, ', permutation

                cur_ind = set(old)
                all_ind = set(range(maxNumBlobs))
                missing = sorted(list(all_ind - cur_ind))

                dfGlobal.loc[globalFrame, 'permutations'], dfGlobal.loc[globalFrame-1,'permutations'] = applyPermutation(maxNumBlobs, permutation, old, missing)

                if i == 0: # if it is the first frame of the segment
                    # I need to update the permutation of the last frame
                    dfPermutations.loc[numFramesA-1,'permutation'] = dfGlobal.loc[globalFrame-1,'permutations']
                    # I save the permutations to the data frame of the segmentation
                    dfA['permutation'] = dfPermutations
                    saveFile(videoPaths[j-1], dfA, 'segment', time = 0)
                    dfPermutations = pd.DataFrame(index = range(len(columnPixels)), columns=['permutation'])
                    dfPermutations.loc[i,'permutation'] = dfGlobal.loc[globalFrame, 'permutations']
                else:
                    dfPermutations.loc[i-1,'permutation'] = dfGlobal.loc[globalFrame-1,'permutations']
                    dfPermutations.loc[i,'permutation'] = dfGlobal.loc[globalFrame, 'permutations']


                if trueFragment: # the current frame has a good overlap with the previous one
                    # print 'There is a good overlaping between A and B'
                    if len(fragment)==0: # it is a new fragment
                        # print 'It is a new fragment'
                        fragment.append(globalFrame-1)

                    if j == len(videoPaths)-1 and i == numFramesSegment-1: # if it is the last frame
                        # print 'It is the last frame of the video'
                        fragment, fragments = storeFragmentIndices(fragment, fragments, globalFrame+1)
                else: # the current frame do not have a good overlap with the previous one
                    # print 'There is not a good overlaping between A and B'
                    fragment, fragments = storeFragmentIndices(fragment, fragments, globalFrame)
                    # print 'This are the fragments so far, ', fragments

                if i == numFramesSegment-1: # it is the last frame of the segment
                    # print 'It is the last frame of the segment'
                    globalFrameCounter += numFramesSegment
                    dfA = df.copy()
                    if j == len(videoPaths)-1:
                        dfA['permutation'] = dfPermutations
                        saveFile(path, dfA, 'segment', time = 0)

                pixelsA = pixelsB
                numFramesA = numFramesSegment
                old = dfGlobal.loc[globalFrame,'permutations']

    print 'fragments , ', fragments
    print 'numFragments, ', len(fragments)
    # fragments = sorted(fragments, key=lambda x: x[1]-x[0],reverse=True)
    # saveFile(videoPaths[0], fragments, 'fragments', time = 0)
    return fragments, dfGlobal

def modelDiffArea(fragments,areas):
    """
    fragment: fragment where to extract the areas to cumpute the mean and std of the diffArea
    areas: areas of all the blobs of the video
    """
    goodFrames = flatten([list(range(fragment[0],fragment[1])) for fragment in fragments])
    individualAreas = np.asarray(flatten(areas[goodFrames].tolist()))
    meanArea = np.mean(individualAreas)
    stdArea = np.std(individualAreas)
    return meanArea, stdArea

def computeVelocity(centroids):
    centroids = np.asarray(centroids).astype('float32')
    vel = np.sqrt(np.sum(np.diff(centroids,axis=0)**2,axis=1))
    return vel.tolist(), np.mean(vel)

def computeDistanceTraveled(centroids):
    '''
    in pixels
    '''
    return np.sum(np.sqrt(np.sum(np.diff(centroids,axis=0)**2,axis=1)),axis=0)

def getIndivFragments(dfGlobal, animalInd,meanIndivArea,stdIndivArea):
    nStd = 4 # num std for area model of single blob
    newdfGlobal = dfGlobal.copy()
    areasFrag = np.asarray(dfGlobal.loc[:,'areas'].tolist())
    centroidsFrag = np.asarray(dfGlobal.loc[:,'centroids'].tolist())
    identities = np.asarray(dfGlobal.loc[:,'permutations'].tolist())
    identitiesInd = np.where(identities==animalInd)
    frames = identitiesInd[0]
    portraitInds = identitiesInd[1]
    potentialCrossings = []
    if frames.size != 0: # there are frames assigned to this individualIndex
        # _, height, width =  portraitsFrag[0].shape
        indivFragments = []
        indivFragment = []
        indivFragmentInterval = ()
        indivFragmentsIntervals = []
        indivCentroidsInterval = []
        lenFragments = []
        avVelFragments = []
        allVels = []
        distTravFragments = []
        sumFragIndices = []
        for i, (frame, portraitInd) in enumerate(zip(frames, portraitInds)):

            if i != len(frames)-1: # if it is not the last frame

                if frames[i+1] - frame == 1 : # if the next frame is a consecutive frame
                    currentArea = areasFrag[frame][portraitInd]
                    currentCentroid = centroidsFrag[frame][portraitInd]

                    if currentArea < meanIndivArea + nStd*stdIndivArea: # if the area is accepted by the model area
                        indivFragment.append((frame, portraitInd))
                        indivCentroidsInterval.append(currentCentroid)

                        if len(indivFragmentInterval) == 0: # is the first frame we append to the interval
                            indivFragmentInterval = indivFragmentInterval + (frame,) # we are using tuples

                    else: # if the area is too big, I close the individual fragment and I add the indices to the list of potentialCrossings
                        print 'changing permutation to -1'
                        potentialCrossings.append((frame,portraitInd)) # save to list of potential crossings
                        newIdentitiesFrame = dfGlobal.loc[frame,'permutations']
                        newIdentitiesFrame[portraitInd] = -1
                        newdfGlobal.set_value(frame, 'permutations', newIdentitiesFrame)

                        if len(indivFragment) != 0:
                            # I close the individual fragment and I open a new one
                            indivFragment = np.asarray(indivFragment)
                            indivFragments.append(indivFragment)

                            indivFragmentInterval = indivFragmentInterval + (frame-1,)
                            indivFragmentsIntervals.append(indivFragmentInterval)

                            # print len(indivFragment)
                            lenFragments.append(len(indivFragment))
                            sumFragIndices.append(sum(lenFragments))
                            vels, avVels = computeVelocity(indivCentroidsInterval)
                            avVelFragments.append(avVels)
                            allVels.append(vels)
                            distTraveled = computeDistanceTraveled(indivCentroidsInterval)
                            distTravFragments.append(distTraveled)
                            indivFragment = []
                            indivCentroidsInterval = []
                            indivFragmentInterval = ()

                else: #the next frame is not a consecutive frame, I close the individual fragment
                    # print 'they are not consecutive frames, I close the fragment'
                    currentArea = areasFrag[frame][portraitInd]
                    currentCentroid = centroidsFrag[frame][portraitInd]

                    if currentArea < meanIndivArea + nStd*stdIndivArea: # if the area is accepted by the model area
                        indivFragment.append((frame, portraitInd))
                        indivCentroidsInterval.append(currentCentroid)

                        if len(indivFragmentInterval) == 0: # is the first frame we append to the interval
                            indivFragmentInterval = indivFragmentInterval + (frame,) # we are using tuples

                    else: # if the area is too big, I close the individual fragment and I add the indices to the list of potentialCrossings
                        potentialCrossings.append((frame,portraitInd)) # save to list of potential crossings
                        newIdentitiesFrame = dfGlobal.loc[frame,'permutations']
                        newIdentitiesFrame[portraitInd] = -1 ### TODO this should be change in the permutation columns of the segments.
                        newdfGlobal.set_value(frame, 'permutations', newIdentitiesFrame)

                    if len(indivFragment) != 0:
                        indivFragment = np.asarray(indivFragment)
                        indivFragments.append(indivFragment)
                        indivFragmentInterval = indivFragmentInterval + (frame,)
                        indivFragmentsIntervals.append(indivFragmentInterval)
                        lenFragments.append(len(indivFragment))
                        sumFragIndices.append(sum(lenFragments))
                        vels, avVels = computeVelocity(indivCentroidsInterval)
                        avVelFragments.append(avVels)
                        allVels.append(vels)
                        distTraveled = computeDistanceTraveled(indivCentroidsInterval)
                        distTravFragments.append(distTraveled)
                        indivFragment = []
                        indivCentroidsInterval = []
                        indivFragmentInterval = ()
            else:
                # print 'it is the last frame, I close the fragments '
                currentArea = areasFrag[frame][portraitInd]
                if currentArea < meanIndivArea + nStd*stdIndivArea: # if the area is accepted by the model area
                    indivFragment.append((frame, portraitInd))
                    indivCentroidsInterval.append(currentCentroid)

                    if len(indivFragmentInterval) == 0: # is the first frame we append to the interval
                        indivFragmentInterval = indivFragmentInterval + (frame,) # we are using tuples


                else: # if the area is too big, I close the individual fragment and I add the indices to the list of potentialCrossings
                    print 'changing permutation to -1'
                    potentialCrossings.append((frame,portraitInd)) # save to list of potential crossings
                    newIdentitiesFrame = dfGlobal.loc[frame,'permutations']
                    newIdentitiesFrame[portraitInd] = -1
                    newdfGlobal.set_value(frame, 'permutations', newIdentitiesFrame)


                # I close the individual fragment and I open a new one
                if len(indivFragment) != 0:
                    indivFragment = np.asarray(indivFragment)
                    indivFragments.append(indivFragment)
                    indivFragmentInterval = indivFragmentInterval + (frame,)
                    indivFragmentsIntervals.append(indivFragmentInterval)
                    lenFragments.append(len(indivFragment))
                    vels, avVels = computeVelocity(indivCentroidsInterval) ### NOTE we are not using it now, we will maybe delete it in the future
                    avVelFragments.append(avVels)
                    allVels.append(vels)
                    distTraveled = computeDistanceTraveled(indivCentroidsInterval)
                    distTravFragments.append(distTraveled)



        return indivFragments,indivFragmentsIntervals, lenFragments, sumFragIndices, avVelFragments, allVels, distTravFragments, newdfGlobal
    else:
        return  [], [], [], [], [], [], [], dfGlobal


def recomputeGlobalFragments(newdfGlobal,numAnimals):
    fragments = []
    fragment = []
    print '*******************************'
    print 'recomputing global fragments...'
    # print newdfGlobal
    for index in newdfGlobal.index:
        # print 'index, ', index
        # print 'permutations, ', newdfGlobal.loc[index,'permutations']
        # print np.sum(newdfGlobal.loc[index,'permutations']>=0)
        # print numAnimals
        if np.sum(newdfGlobal.loc[index,'permutations']>=0) == numAnimals and len(newdfGlobal.loc[index,'areas']) ==numAnimals:
            if len(fragment) == 0:
                fragment.append(index)
        else:
            if len(fragment) == 1:
                fragment.append(index-1)
                fragments.append(fragment)
                fragment = []
        # print 'fragments, ', fragments

    return fragments

def getIndivAllFragments(dfGlobal,meanIndivArea,stdIndivArea,maxNumBlobs,numAnimals):
    print '\n Computing individual fragments ******************'
    newdfGlobal = dfGlobal.copy()
    oneIndivFragIntervals = []
    oneIndivFragFrames = []
    oneIndivFragLens = []
    oneIndivFragSumLens = []
    oneIndivFragVels = []
    oneIndivFragDists = []
    allVelsVideo = []
    print dfGlobal.loc[80:90]
    for i in range(int(maxNumBlobs)):
        print 'Computing individual fragments for blob index, ', i
        indivFragments,indivFragmentsIntervals, lenFragments,sumFragIndices, avVelFragments, allVels, distTravFragments, newdfGlobal = getIndivFragments(newdfGlobal, i,meanIndivArea,stdIndivArea)
        oneIndivFragIntervals.append(indivFragmentsIntervals)
        oneIndivFragFrames.append(indivFragments)
        oneIndivFragLens.append(lenFragments)
        oneIndivFragSumLens.append(sumFragIndices)
        oneIndivFragVels.append(avVelFragments)
        allVelsVideo.append(allVels)
        oneIndivFragDists.append(distTravFragments)

    # allVelsVideo = flatten(allVelsVideo)
    #
    # flatVel = np.asarray(flatten(flatten(allVelsVideo)))
    # plt.figure()
    # plt.hist(np.log(np.add(flatVel[~np.isnan(flatVel)],0.000000000000000001)),  bins=150)
    # plt.show()
    print dfGlobal.loc[80:90]
    fragments = recomputeGlobalFragments(newdfGlobal,numAnimals)
    fragments = np.asarray(fragments)
    return oneIndivFragIntervals, oneIndivFragFrames, oneIndivFragLens, oneIndivFragSumLens, oneIndivFragVels, oneIndivFragDists, newdfGlobal, fragments

def getCoexistence(fragments,oneIndivFragIntervals,oneIndivFragLens,oneIndivFragVels,oneIndivFragFrames,oneIndivFragDists,numAnimals):

    def getOverlap(a, b):
        overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] == b[0] or a[0] == b[1]:
            overlap = 1
        if a[1] == a[0] and (b[0]<=a[0] and b[1]>=a[1]):
            overlap = 1
        if b[1] == b[0] and (a[0]<=b[0] and a[1]>=b[1]):
            overlap = 1
        return overlap

    minLenIndivCompleteFragments = [] # list of the minimum length of the individuals fragments of a complete fragment
    minDistIndivCompleteFragments = []
    intervalsFragments = [] # list of lists of (fragmentsListIndex,fragmentIndex,fragmentInterval,lenfragment)
    intervalsFragmentsDist = []
    framesAndBlobColumns = [] # list of lists of nx2 arrays where n is the len of the individual fragment. the first column is the frame and the second column is the column of the blob
    framesAndBlobColumnsDist = []
    for i, fragment in enumerate(fragments):
        print '******************************************************'
        print '\n Computing lengths of one-individual fragments in complete fragment, ', i, fragment
        lenIndivFrag = []
        distIndivFrag = []
        intervalsFragment = []
        framesAndBlobIndexFragment = []
        for j, (oneIndivFrags, oneIndivFragLen) in enumerate(zip(oneIndivFragIntervals, oneIndivFragLens)):
            oneIndivFragDist = oneIndivFragDists[j]
            print '*** coexistence in one-individual fragments list ', j
            oneIndivFragVel = oneIndivFragVels[j]
            print 'one individual fragments, ', oneIndivFrags
            overlaps = np.asarray([getOverlap(fragment,indivFrag) for indivFrag in oneIndivFrags])
            print 'overlaps, ', overlaps
            coexistingFragments = np.where(overlaps != 0)[0]
            print 'coexisting fragments, ', coexistingFragments
            if len(coexistingFragments) > 1:
                raise ValueError('There cannot be two individual fragments from the same list coexisting with a global fragment')
            if len(coexistingFragments)!=0:
                coexistingFragment = coexistingFragments[0]
                print 'coexisting fragment, ', coexistingFragment, ', interval, ', oneIndivFrags[coexistingFragment], ', length, ', oneIndivFragLen[coexistingFragment], ', dist, ', oneIndivFragDist[coexistingFragment]
                intervalsFragment.append((j,coexistingFragment,oneIndivFrags[coexistingFragment],oneIndivFragLen[coexistingFragment], oneIndivFragVel[coexistingFragment], oneIndivFragDist[coexistingFragment]))
                framesAndBlobIndexFragment.append(oneIndivFragFrames[j][coexistingFragment])
                lenIndivFrag.append(oneIndivFragLen[coexistingFragment])
                distIndivFrag.append(oneIndivFragDist[coexistingFragment])

        if len(intervalsFragment) != numAnimals:
            raise ValueError('The number of one-individual intervals should be the same as the number of animals in the video')
        intervalsFragments.append(intervalsFragment)
        framesAndBlobColumns.append(framesAndBlobIndexFragment)
        minLenIndivCompleteFragments.append(np.min(lenIndivFrag))
        minDistIndivCompleteFragments.append(np.min(distIndivFrag))
    print '******************************************************'
    minLenIndivCompleteFragments = np.asarray(minLenIndivCompleteFragments)
    minDistIndivCompleteFragments = np.asarray(minDistIndivCompleteFragments)

    argsort = minLenIndivCompleteFragments.argsort()[::-1]
    argsortDist = minDistIndivCompleteFragments.argsort()[::-1]

    # print '******************************************************'
    # print '******************************************************'
    # print 'argsort, ', argsort
    # print 'argsortDist, ', argsortDist
    # plt.ion()
    # plt.figure()
    # plt.plot(np.asarray(argsortDist).argsort(),np.asarray(argsort).argsort(),'o')
    # plt.show()
    # print '******************************************************'
    # print '******************************************************'
    #
    # raw_input('Press ENTER to continue.')

    fragments = fragments[argsortDist]
    # fragments = fragments[argsort]
    minLenIndivCompleteFragments = minLenIndivCompleteFragments[argsort]
    minDistIndivCompleteFragments = minDistIndivCompleteFragments[argsortDist]

    framesAndBlobColumnsLen =  [framesAndBlobColumns[i] for i in argsort]
    intervalsFragmentsLen = [intervalsFragments[i] for i in argsort]

    framesAndBlobColumnsDist =  [framesAndBlobColumns[i] for i in argsortDist]
    intervalsFragmentsDist = [intervalsFragments[i] for i in argsortDist]

    return fragments, framesAndBlobColumnsLen, intervalsFragmentsLen , minLenIndivCompleteFragments.tolist(), minDistIndivCompleteFragments.tolist(), framesAndBlobColumnsDist, intervalsFragmentsDist

def fragment(videoPaths,videoInfo = None):
    ''' Load videoInfo if needed '''
    if videoInfo == None:
        videoInfo = loadFile(videoPaths[0], 'videoInfo', time = 0)
        videoInfo = videoInfo.to_dict()[0]
        numFrames = videoInfo['numFrames']
        numAnimals = videoInfo['numAnimals']
        maxNumBlobs = videoInfo['maxNumBlobs']

    ''' Compute permutations and complete fragments '''
    fragments, dfGlobal = newFragmentator(videoPaths,numAnimals,maxNumBlobs, numFrames)
    saveFile(videoPaths[0],dfGlobal,'dfGlobal',time=0)
    playFragmentation(videoPaths,dfGlobal,False)

    ''' Compute model area of individual blob '''
    fragments = np.asarray(fragments)
    meanIndivArea, stdIndivArea = modelDiffArea(fragments, dfGlobal.areas)
    # print 'meanIndivArea, ', meanIndivArea
    # print 'stdindivArea, ', stdIndivArea
    videoInfo = loadFile(videoPaths[0], 'videoInfo', time = 0)
    videoInfo = videoInfo.to_dict()[0]
    videoInfo['meanIndivArea'] = meanIndivArea
    videoInfo['stdIndivArea'] = stdIndivArea
    saveFile(videoPaths[0],videoInfo,'videoInfo',time=0)
    # print 'fragments before individual fragments, ', fragments
    oneIndivFragIntervals, oneIndivFragFrames, oneIndivFragLens, oneIndivFragSumLens, oneIndivFragVels, oneIndivFragDists, dfGlobal, fragments = getIndivAllFragments(dfGlobal,meanIndivArea,stdIndivArea,maxNumBlobs,numAnimals)
    # print 'fragments after individual fragments, ', fragments
    # print dfGlobal.loc[80:90]
    playFragmentation(videoPaths,dfGlobal,False)
    # oneIndivFragIntervals, oneIndivFragFrames, oneIndivFragLens, oneIndivFragSumLens, dfGlobal = getIndivAllFragments(dfGlobal,meanIndivArea,stdIndivArea,maxNumBlobs,numAnimals)

    # print '\n dfGlobal', dfGlobal

    fragments, framesAndBlobColumns, intervalsFragments, minLenIndivCompleteFragments, minDistIndivCompleteFragments, framesAndBlobColumnsDist, intervalsFragmentsDist = getCoexistence(fragments,oneIndivFragIntervals,oneIndivFragLens,oneIndivFragVels,oneIndivFragFrames,oneIndivFragDists,numAnimals)
    fragmentsDict = {
        'fragments': fragments, #global fragments
        'minLenIndivCompleteFragments': minLenIndivCompleteFragments,
        'minDistIndivCompleteFragments': minDistIndivCompleteFragments,
        'framesAndBlobColumns': framesAndBlobColumns,
        'framesAndBlobColumnsDist': framesAndBlobColumnsDist,
        'intervals': intervalsFragments,
        'intervalsDist': intervalsFragmentsDist,
        'oneIndivFragIntervals': oneIndivFragIntervals,
        'oneIndivFragFrames': oneIndivFragFrames,
        'oneIndivFragLens': oneIndivFragLens,
        'oneIndivFragSumLens': oneIndivFragSumLens,
        'oneIndivFragVels': oneIndivFragVels,
        'oneIndivFragDists': oneIndivFragDists
        }
    saveFile(videoPaths[0],fragmentsDict,'fragments',time=0, hdfpkl='pkl')
    saveFile(videoPaths[0],dfGlobal,'portraits',time=0)

    return dfGlobal, fragmentsDict

# videoPath = '/home/lab/Desktop/TF_models/IdTracker/Cafeina5peces/Caffeine5fish_20140206T122428_1.avi'
# videoPaths = scanFolder(videoPath)
# fragment(videoPaths)
