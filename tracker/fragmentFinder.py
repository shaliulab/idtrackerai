# Import standard libraries
# import os
import sys
import numpy as np
np.set_printoptions(precision=2)
import numpy.matlib

# Import application/library specifics
sys.path.append('../utils')
sys.path.append('../CNN')

from py_utils import *
from video_utils import *
from idTrainerTracker import *
from tf_utils import *
from input_data_cnn import *
from cnn_utils import *

# import time


# import argparse
# import glob
# import pandas as pd
# import re
# from joblib import Parallel, delayed
# import multiprocessing
# import cPickle as pickle
# import tensorflow as tf

# from pprint import pprint
# from collections import Counter
# import collections
# import datetime

def getAvVelFragment(portraits,framesAndColumns):
    centroids = []
    for (frame,column) in framesAndColumns:
        centroids.append(portraits.loc[frame,'centroids'][column])
    centroids = np.asarray(centroids).astype('float32')
    vels = np.sqrt(np.sum(np.diff(centroids,axis=0)**2,axis=1))
    return np.mean(vels)

def bestFragmentFinder(accumDict, normFreqFragsAll, fragmentsDict, numAnimals, portraits):
    print '\n--- Entering the bestFragmentFinder ---'

    ''' Retrieve variables from accumDict '''
    accumCounter = accumDict['counter']
    thVels = accumDict['thVels']
    minDistTrav = accumDict['minDist']
    fragsForTrain = accumDict['fragsForTrain']
    badFragments = accumDict['badFragments']

    if accumCounter == 0:
        print '\nFinding first fragment to fine tune'

        ''' Set distance travelled threshold for accumulation '''
        print '\nSetting the threshold for the minimum distance travelled'
        minDistIndivCompleteFragments = flatten(fragmentsDict['minDistIndivCompleteFragments'])
        oneIndivFragVels = np.asarray(flatten(fragmentsDict['oneIndivFragVels']))
        oneIndivFragLens = np.asarray(flatten(fragmentsDict['oneIndivFragLens']))

        avVel = np.nanmean(oneIndivFragVels)
        avLen = np.nanmean(oneIndivFragLens)
        minDistTrav = int(avVel * avLen)
        print 'The threshold for the distance travelled is ', minDistTrav

        ''' Find first global fragment to fine tune '''
        indexFragment = 0
        avVels = [0,0]
        framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumnsDist']
        intervalsDist = fragmentsDict['intervalsDist']

        while any(np.asarray(avVels)<=thVels):
            avVels = []
            print '\nChecking whether the fragmentNumber ', indexFragment, ' is good for training'

            for framesAndColumnsInterval in framesAndColumnsGlobalFrag[indexFragment]:
                avVels.append(getAvVelFragment(portraits,framesAndColumnsInterval))

            print 'The average velocities for each blob are (pixels/frame), '
            print avVels

            if any(np.asarray(avVels)<=thVels):
                accumDict['badFragments'].append(indexFragment)
                indexFragment += 1
                print 'There are some animals that does not move enough. Going to next longest fragment'
                print 'Bad fragments, ', accumDict['badFragments']
            else:
                print 'The fragment ', indexFragment, ' is good for training'

        fragsForTrain = [indexFragment]
        acceptableFragIndices = [indexFragment]
        print '\nThe fine-tuning will start with the ', indexFragment, ' longest global fragment'
        print 'Individual fragments inside the global fragment, ', intervalsDist[indexFragment]
        continueFlag = True
        accumDict['minDist'] = minDistTrav
        accumDict['fragsForTrain'] = fragsForTrain
        accumDict['badFragments'] = badFragments
        accumDict['newFragForTrain'] = acceptableFragIndices

        continueFlag = True
        accumDict['continueFlag'] = continueFlag

    else:

        def computeUniqueness(fragMat, numAnimals):
            unique = True
            fragMat = np.asarray(fragMat)
            # rawsMax = np.max(fragMat, axis=1)
            rawsArgMax = np.argmax(fragMat, axis=1)
            ids = range(numAnimals)
            # misId --> discard
            if ids != list(rawsArgMax):
                unique = False
            # too uncertain --> discard NOTE: commented since the distance should take care of it
            # if np.min(rawsMax) < 0.5:
            #     unique = False
            return unique

        print '\nFinding next best fragment for references'
        # Load data needed and pass it to arrays
        fragments = np.asarray(fragmentsDict['fragments'])
        framesAndBlobColumns = fragmentsDict['framesAndBlobColumns']
        minLenIndivCompleteFragments = fragmentsDict['minLenIndivCompleteFragments']
        minDistIndivCompleteFragments = fragmentsDict['minDistIndivCompleteFragments']
        lens = np.asarray(minLenIndivCompleteFragments)
        distsTrav = np.asarray(minDistIndivCompleteFragments)
        intervalsFragments = fragmentsDict['intervals']

        # Compute distances to the identity matrix for each complete set of individual fragments
        mat = []
        distI = []
        notUnique = []
        identity = np.identity(numAnimals)
        for i, intervals in enumerate(intervalsFragments): # loop in complete set of fragments
            # print 'fragment, ', i
            for j, interval in enumerate(intervals): # loop in individual fragments of the complete set of fragments
                # print 'individual fragment, ', j
                mat.append(normFreqFragsAll[interval[0]][interval[1]])
            matFragment = np.vstack(mat)
            mat = []
            perm = np.argmax(matFragment,axis=1)

            matFragment = matFragment[:,perm]
            uniqueness = computeUniqueness(matFragment, numAnimals)
            if not uniqueness:
                notUnique.append(i)
                print '------------------------------'
                print str(i) + ' is not unique'
                print '------------------------------'

            # print 'permuted mat fragment \n', matFragment
            # print numpy.linalg.norm(matFragment - identity)
            # print lens[i]
            distI.append(numpy.linalg.norm(matFragment - identity)) #TODO when optimizing the code one should compute the matrix distance only for fragments above 100 length

        distI = np.asarray(distI)
        distInorm = distI/np.max(distI)
        # lensnorm = np.true_divide(lens,np.max(lens))
        distsTravNorm = np.true_divide(distsTrav,np.max(distsTrav))

        # Get best values of the parameters length and distance to identity
        distI0norm = np.min(distInorm[fragsForTrain])
        distI0norm = np.ones(len(distI))*distI0norm
        # len0norm = np.max(lensnorm[fragsForTrain])
        # len0norm = np.ones(len(lens))*len0norm
        distTrav0norm = np.max(distsTravNorm[fragsForTrain])
        distTrav0norm = np.ones(len(distsTrav))*distTrav0norm

        # Compute score of every global fragment with respect to the optimal value of the parameters
        # score = np.sqrt((distI0norm-distInorm)**2 + ((len0norm-lensnorm))**2)

        ###NOTE: I wuold avoid the sqrt, it's only time consuming
        score = np.sqrt((distI0norm-distInorm)**2 + ((distTrav0norm-distsTravNorm))**2)

        # Get indices of the best fragments according to its score
        fragIndexesSorted = np.argsort(score).tolist()

        # Remove short fragments
        print '\n(fragIndexesSorted, distTrav, lens) before eliminating the short ones, ', zip(fragIndexesSorted,distsTrav[fragIndexesSorted],lens[fragIndexesSorted])
        print 'Current minimum distance travelled, ', minDistTrav
        fragIndexesSortedLong = [x for x in fragIndexesSorted if distsTrav[x] > minDistTrav]
        print '(fragIndexesSorted, distTrav, lens) after eliminating the short ones, ', zip(fragIndexesSortedLong,distsTrav[fragIndexesSortedLong],lens[fragIndexesSortedLong])

        # We only consider fragments that have not been already picked for fine-tuning
        print '\nCurrent fragsForTrain, ', fragsForTrain
        print 'Bad fragments, ', badFragments
        nextPossibleFragments = [frag for frag in fragIndexesSortedLong if frag not in fragsForTrain and frag not in badFragments and frag not in notUnique]

        # Check whether the animals are moving enough so that the images are going to be different enough
        realNextPossibleFragments = []
        for frag in nextPossibleFragments:
            framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][frag]
            avVels = []
            print '\nChecking whether the fragmentNumber ', frag, ' is good for training'
            for framesAndColumnsInterval in framesAndColumnsGlobalFrag:
                avVels.append(getAvVelFragment(portraits,framesAndColumnsInterval))
            print 'The average velocities for each blob are (pixels/frame), '
            print avVels
            if any(np.asarray(avVels)<=thVels):
                badFragments.append(frag)
                print 'There is some animal that does not move enough'
                print 'Bad fragments, ', badFragments
            else:
                print 'The fragment ', frag, ' is good for training'
                realNextPossibleFragments.append(frag)

        while len(realNextPossibleFragments)==0 and minDistTrav >= 0:
            print '\nThe list of possible fragments is the same as the list of fragments used previously for finetuning'
            print 'We reduce the minLen in 50 units'
            if minDistTrav > 50:
                step = 50
            else:
                step = 10
            minDistTrav -= step
            print 'Current minimum distTrav, ', minDistTrav
            fragIndexesSortedLong = [x for x in fragIndexesSorted if distsTrav[x] > minDistTrav]

            # We only consider fragments that have not been already picked for fine-tuning
            print 'Current fragsForTrain, ', fragsForTrain
            print 'Bad fragments, ', badFragments
            nextPossibleFragments = [frag for frag in fragIndexesSortedLong if frag not in fragsForTrain and frag not in badFragments and frag not in notUnique]
            nextPossibleFragments = np.asarray(nextPossibleFragments)
            print 'Next possible fragments for train', nextPossibleFragments

            realNextPossibleFragments = []
            for frag in nextPossibleFragments:
                framesAndColumnsGlobalFrag = fragmentsDict['framesAndBlobColumns'][frag]
                avVels = []
                print '\nChecking whether the fragmentNumber ', frag, ' is good for training'
                for framesAndColumnsInterval in framesAndColumnsGlobalFrag:
                    avVels.append(getAvVelFragment(portraits,framesAndColumnsInterval))
                print 'The average velocities for each blob are (pixels/frame), '
                print avVels
                if any(np.asarray(avVels)<=thVels):
                    badFragments.append(frag)
                    print 'There is some animal that does not move enough'
                    print 'Bad fragments, ', badFragments
                else:
                    print 'The fragment ', frag, ' is good for training'
                    realNextPossibleFragments.append(frag)

        nextPossibleFragments = np.asarray(realNextPossibleFragments)
        print '\nNext possible fragments for train', nextPossibleFragments


        if len(nextPossibleFragments) != 0:

            lensND = np.asarray([lens[frag] for frag in nextPossibleFragments])
            distsTravND = np.asarray([distsTrav[frag] for frag in nextPossibleFragments])
            distIND = np.asarray([distI[frag] for frag in nextPossibleFragments])
            print '\n(len, distTrav, dist) of the nextPossibleFragments', zip(lensND,distsTravND,distIND)

            bestFragInd = nextPossibleFragments[0]
            bestFragDist = distI[bestFragInd]
            bestFragLen = lens[bestFragInd]
            bestFragDistTrav = distsTrav[bestFragInd]
            print '\nBestFragInd, bestFragDist, bestFragLen, bestFragDistTrav'
            print bestFragInd, bestFragDist, bestFragLen, bestFragDistTrav

            # acceptableFragIndices = np.where((lensND > 100) & (distIND <= bestFragDist))[0]
            acceptable = np.where((distIND <= bestFragDist))[0]
            acceptableFragIndices = nextPossibleFragments[acceptable]

            fragsForTrain = np.asarray(fragsForTrain)
            print '\nOld Frags for train, ', fragsForTrain
            print 'acceptableFragIndices, ', acceptableFragIndices
            fragsForTrain = np.unique(np.hstack([fragsForTrain,acceptableFragIndices])).tolist()
            print 'Fragments for training, ', fragsForTrain
            acceptableFragIndices = acceptableFragIndices.tolist()

            accumDict['minDist'] = minDistTrav
            accumDict['fragsForTrain'] = fragsForTrain
            accumDict['badFragments'] = badFragments
            accumDict['newFragForTrain'] = acceptableFragIndices

            continueFlag = True
            accumDict['continueFlag'] = continueFlag
        else:
            print '\nThere are no more good fragments'
            continueFlag = False
            accumDict['continueFlag'] = continueFlag



    return accumDict
