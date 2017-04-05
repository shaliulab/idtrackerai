from __future__ import division
# Import standard libraries
import sys
import numpy as np
np.set_printoptions(precision=2)
import numpy.matlib

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/CNN')

from py_utils import *
from video_utils import *
from idTrainerTracker import *
from tf_utils import *
from input_data_cnn import *
from cnn_utils import *


def setMinDistTrav(fragmentsDict,accumDict):
    minDistIndivCompleteFragments = flatten(fragmentsDict['minDistIndivCompleteFragments'])
    oneIndivFragVels = np.asarray(flatten(fragmentsDict['oneIndivFragVels']))
    oneIndivFragLens = np.asarray(flatten(fragmentsDict['oneIndivFragLens']))

    avVel = np.nanmean(oneIndivFragVels)
    avLen = np.nanmean(oneIndivFragLens)
    minDistTrav = avVel * avLen
    accumDict['minDist'] = minDistTrav

    return accumDict

def findFirstGlobalFragment(fragmentsDict,accumDict):
    indexFragment = 0
    avVels = [0,0]

    while any(np.asarray(avVels)<=accumDict['thVels']):
        avVels = []
        print '\nChecking whether the fragmentNumber ', indexFragment, ' is good for training'

        for indivFragment in fragmentsDict['intervals'][indexFragment]:
            avVels.append(indivFragment[4])

        print 'The average velocities for each blob are (pixels/frame), '
        print avVels

        if any(np.asarray(avVels)<=accumDict['thVels']):
            accumDict['badFragments'].append(indexFragment)
            indexFragment += 1
            print 'There are some animals that does not move enough. Going to next longest fragment'
            print 'Bad fragments, ', accumDict['badFragments']
        else:
            print 'The fragment ', indexFragment, ' is good for training'
            fragsForTrain = [indexFragment]

    acceptableFragIndices = fragsForTrain
    print '\nThe fine-tuning will start with the ', indexFragment, ' longest global fragment'
    print 'Individual fragments inside the global fragment, ', fragmentsDict['intervalsDist'][indexFragment]
    accumDict['fragsForTrain'] = fragsForTrain
    accumDict['newFragForTrain'] = acceptableFragIndices
    accumDict['continueFlag'] = True
    return accumDict

def computeDistToIdAndUniqueness(fragmentsDict, numAnimals, statistics):

    def computeUniqueness(fragMat, numAnimals):
        unique = True
        fragMat = np.asarray(fragMat)
        # rawsMax = np.max(fragMat, axis=1)
        rawsArgMax = np.argmax(fragMat, axis=1)
        ids = range(numAnimals)
        # misId --> discard
        if ids != list(rawsArgMax):
            unique = False
        # else:
        #     print fragMat
        return unique

    mat = []
    ids = []
    distI = []
    notUnique = []
    unique = []
    identity = np.identity(numAnimals)
    identities = range(1,numAnimals+1)
    for i, globalFrag in enumerate(fragmentsDict['intervalsDist']): # loop in complete set of fragments
        print '\ngloalFrag, ', i
        for j, indivFrag in enumerate(globalFrag): # loop in individual fragments of the complete set of fragments
            blobIndex = indivFrag[0]
            fragNum = indivFrag[1]
            ids.append(statistics['idFragsAll'][blobIndex][fragNum])
            mat.append(statistics['P2FragsAll'][blobIndex][fragNum])


        if set(identities).difference(set(ids)):
            print 'ids, ', ids
            repeated_ids = set([x for x in ids if ids.count(x) > 1])
            print 'The identities ', list(repeated_ids), ' are repeated'
            missing_ids = set(identities).difference(set(ids))
            print 'The identities ', list(missing_ids), 'are missing'
        else:
            print 'All identities are in this fragment'
            print 'ids, ', ids
        matFragment = np.vstack(mat)
        mat = []
        ids = []
        perm = np.argmax(matFragment,axis=1)

        matFragment = matFragment[:,perm]
        uniqueness = computeUniqueness(matFragment, numAnimals)
        if not uniqueness:
            notUnique.append(i)
            print 'Global fragment ' + str(i) + ' is not unique'
        elif uniqueness:
            unique.append(i)
            print 'Global fragment ' + str(i) + ' is unique'


        distI.append(numpy.linalg.norm(matFragment - identity)) #TODO when optimizing the code one should compute the matrix distance only for fragments above 100 length
    distI = np.asarray(distI)

    return distI, notUnique, unique

def computeScore(accumDict,distI,distsTrav):

    distInorm = distI/np.max(distI)
    distsTravNorm = np.true_divide(distsTrav,np.max(distsTrav))

    # Get best values of the parameters length and distance to identity
    distI0norm = np.min(distInorm[accumDict['fragsForTrain']])
    distI0norm = np.ones(len(distI))*distI0norm
    distTrav0norm = np.max(distsTravNorm[accumDict['fragsForTrain']])
    distTrav0norm = np.ones(len(distsTrav))*distTrav0norm

    # Compute score of every global fragment with respect to the optimal value of the parameters
    score = np.sqrt((distI0norm-distInorm)**2 + ((distTrav0norm-distsTravNorm))**2) ###NOTE: I wuold avoid the sqrt, it's only time consuming
    return score

def checkVelFragments(nextPossibleFragments,fragmentsDict,accumDict):

    realNextPossibleFragments = []
    for frag in nextPossibleFragments:
        globalFragmentData = fragmentsDict['intervals'][frag]
        avVels = []
        print '\nChecking whether the fragmentNumber ', frag, ' is good for training'
        for indivFrag in globalFragmentData:
            avVels.append(indivFrag[4])
        print 'The average velocities for each blob are (pixels/frame), '
        print avVels
        if any(np.asarray(avVels)<=accumDict['thVels']):
            accumDict['badFragments'].append(frag)
            print 'There is some animal that does not move enough'
            print 'Bad fragments, ', accumDict['badFragments']
        else:
            print 'The fragment ', frag, ' is good for training'
            realNextPossibleFragments.append(frag)
    return realNextPossibleFragments, accumDict

def checkNoveltyNewFrags(intervals, nextPossibleFragments, usedIndivIntervals, accumDict):
    countNewIntervals = 0
    realNextPossibleFragments = []

    for frag in nextPossibleFragments: # for each complete fragment that has to be used for the training
        intervalsIndivFrags = intervals[frag] # I take the list of individual fragments in terms of intervals

        for intervalIndivFrag in intervalsIndivFrags:
            if not intervalIndivFrag in usedIndivIntervals: # I only use individual fragments that have not been used before
                countNewIntervals += 1
        if countNewIntervals > 0:
            realNextPossibleFragments.append(frag)
        else:
            accumDict['badFragments'].append(frag)

    return realNextPossibleFragments, accumDict

def getNextPossibleFragments(fragmentsDict, accumDict, trainDict, fragIndexesSorted, notUnique, distsTrav):

    print 'Removing short fragments...'
    print 'Current minimum distance travelled, ', accumDict['minDist']
    fragIndexesSortedLong = [x for x in fragIndexesSorted if distsTrav[x] > accumDict['minDist']]
    print 'Number of fragments after eliminating the short ones, ', len(zip(fragIndexesSortedLong,distsTrav[fragIndexesSortedLong]))

    # We only consider fragments that have not been already picked for fine-tuning
    print '\nCurrent fragsForTrain, ', accumDict['fragsForTrain']
    print 'Current bad fragments, ', accumDict['badFragments']
    nextPossibleFragments = [frag for frag in fragIndexesSortedLong if frag not in accumDict['fragsForTrain'] and frag not in accumDict['badFragments'] and frag not in notUnique]
    print 'Next possible fragments after remobing current ones, bad Fragments and the not unique, ', nextPossibleFragments
    # if not nextPossibleFragments:
    #     raise ValueError('There are not more fragments that are good for training')
    # Check whether the animals are moving enough so that the images are going to be different enough
    nextPossibleFragments, accumDict = checkVelFragments(nextPossibleFragments,fragmentsDict,accumDict)

    # Check if at least one individual fragment is new to the network
    intervals = fragmentsDict['intervalsDist']
    usedIndivIntervals = trainDict['usedIndivIntervals']
    nextPossibleFragments, accumDict = checkNoveltyNewFrags(intervals, nextPossibleFragments, usedIndivIntervals, accumDict)

    return nextPossibleFragments, accumDict

def getAcceptableFragments(accumDict, nextPossibleFragments, distsTrav, distI):

    fragsForTrain = accumDict['fragsForTrain']
    if len(nextPossibleFragments) != 0:

        distsTravND = np.asarray([distsTrav[frag] for frag in nextPossibleFragments])
        distIND = np.asarray([distI[frag] for frag in nextPossibleFragments])
        print '\n(distTrav, dist) of the nextPossibleFragments', zip(distsTravND,distIND)

        bestFragInd = nextPossibleFragments[0]
        bestFragDist = distI[bestFragInd]
        bestFragDistTrav = distsTrav[bestFragInd]
        print '\nBestFragInd, bestFragDist, bestFragDistTrav'
        print bestFragInd, bestFragDist, bestFragDistTrav

        # acceptableFragIndices = np.where((lensND > 100) & (distIND <= bestFragDist))[0]
        acceptable = np.where((distIND <= bestFragDist))[0]
        acceptableFragIndices = nextPossibleFragments[acceptable]

        fragsForTrain = np.asarray(fragsForTrain)
        print '\nOld Frags for train, ', fragsForTrain
        print 'acceptableFragIndices, ', acceptableFragIndices
        fragsForTrain = np.unique(np.hstack([fragsForTrain,acceptableFragIndices])).tolist()
        print 'Fragments for training, ', fragsForTrain
        acceptableFragIndices = acceptableFragIndices.tolist()

        accumDict['fragsForTrain'] = fragsForTrain
        accumDict['newFragForTrain'] = acceptableFragIndices

        continueFlag = True
        accumDict['continueFlag'] = continueFlag
    else:
        print '\nThere are no more good fragments'
        continueFlag = False
        accumDict['continueFlag'] = continueFlag

    return accumDict

def bestFragmentFinder(accumDict, trainDict, statistics, fragmentsDict, numAnimals):

    print '\n--- Entering the bestFragmentFinder ---'

    if accumDict['counter'] == 0:
        print '\nFinding first fragment for fine tunning'

        # Set minimum distance travelled
        print '\nSetting the threshold for the minimum distance travelled'
        accumDict = setMinDistTrav(fragmentsDict,accumDict)
        print 'The threshold for the distance travelled is ', accumDict['minDist']

        # Find first global fragment for fine tunning
        accumDict = findFirstGlobalFragment(fragmentsDict,accumDict)

    else:
        print '\nFinding next best fragment for references'

        # Load data needed and pass it to arrays
        distsTrav = np.asarray(fragmentsDict['minDistIndivCompleteFragments'])

        # Compute distances to the identity matrix for each complete set of individual fragments (for each global fragment)
        distI, notUnique, unique = computeDistToIdAndUniqueness(fragmentsDict, numAnimals, statistics)
        print len(notUnique)/len(distI), ' proportion of non unique'
        print len(unique)/len(distI), ' proportion of unique'
        print 'unique fragments, ', unique
        print 'non unique fragments, ', notUnique

        # Compute score from distances to identity and distance travelled
        score = computeScore(accumDict,distI,distsTrav)

        # Get indices of the best fragments according to its score
        fragIndexesSorted = np.argsort(score).tolist()

        # Get next possible fragments
        nextPossibleFragments, accumDict = getNextPossibleFragments(fragmentsDict, accumDict, trainDict, fragIndexesSorted, notUnique, distsTrav)

        while len(nextPossibleFragments)==0 and accumDict['minDist'] >= 0:
            print '\nThe list of possible fragments is the same as the list of fragments used previously for finetuning'

            if accumDict['minDist'] > 50:
                step = 50
                print 'We reduce the minLen in %i units' %step
            else:
                step = 10
                print 'We reduce the minLen in %i units' %step
            accumDict['minDist'] -= step

            nextPossibleFragments, accumDict = getNextPossibleFragments(fragmentsDict, accumDict, trainDict, fragIndexesSorted, notUnique, distsTrav)

        nextPossibleFragments = np.asarray(nextPossibleFragments)
        print '\nNext possible fragments for train', nextPossibleFragments

        accumDict = getAcceptableFragments(accumDict, nextPossibleFragments, distsTrav, distI)

    return accumDict
