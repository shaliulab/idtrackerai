# Import standard libraries
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
        # too uncertain --> discard NOTE: commented since the distance should take care of it
        # if np.min(rawsMax) < 0.5:
        #     unique = False
        return unique

    mat = []
    distI = []
    notUnique = []
    identity = np.identity(numAnimals)
    for i, globalFrag in enumerate(fragmentsDict['intervals']): # loop in complete set of fragments
        for j, indivFrag in enumerate(globalFrag): # loop in individual fragments of the complete set of fragments
            blobIndex = indivFrag[0]
            fragNum = indivFrag[1]
            mat.append(statistics['normFreqFragsAll'][blobIndex][fragNum])
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

        distI.append(numpy.linalg.norm(matFragment - identity)) #TODO when optimizing the code one should compute the matrix distance only for fragments above 100 length
    distI = np.asarray(distI)

    return distI, notUnique

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

def getNextPossibleFragments(fragmentsDict, accumDict, fragIndexesSorted, notUnique, distsTrav):

    print 'Removing short fragments...'
    print 'Current minimum distance travelled, ', accumDict['minDist']
    fragIndexesSortedLong = [x for x in fragIndexesSorted if distsTrav[x] > accumDict['minDist']]
    print '(fragIndexesSorted, distTrav) after eliminating the short ones, ', zip(fragIndexesSortedLong,distsTrav[fragIndexesSortedLong])

    # We only consider fragments that have not been already picked for fine-tuning
    print '\nCurrent fragsForTrain, ', accumDict['fragsForTrain']
    print 'Current bad fragments, ', accumDict['badFragments']
    nextPossibleFragments = [frag for frag in fragIndexesSortedLong if frag not in accumDict['fragsForTrain'] and frag not in accumDict['badFragments'] and frag not in notUnique]

    # Check whether the animals are moving enough so that the images are going to be different enough
    nextPossibleFragments, accumDict = checkVelFragments(nextPossibleFragments,fragmentsDict,accumDict)

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

def bestFragmentFinder(accumDict, statistics, fragmentsDict, numAnimals):

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

        # Compute distances to the identity matrix for each complete set of individual fragments
        distI, notUnique = computeDistToIdAndUniqueness(fragmentsDict, numAnimals, statistics)

        # Compute score from distances to identity and distance travelled
        score = computeScore(accumDict,distI,distsTrav)

        # Get indices of the best fragments according to its score
        fragIndexesSorted = np.argsort(score).tolist()

        # Get next possible fragments
        nextPossibleFragments, accumDict = getNextPossibleFragments(fragmentsDict, accumDict, fragIndexesSorted, notUnique, distsTrav)

        while len(nextPossibleFragments)==0 and accumDict['minDist'] >= 0:
            print '\nThe list of possible fragments is the same as the list of fragments used previously for finetuning'
            print 'We reduce the minLen in 50 units'
            if accumDict['minDist'] > 50:
                step = 50
            else:
                step = 10
            accumDict['minDist'] -= step

            nextPossibleFragments, accumDict = getNextPossibleFragments(fragmentsDict, accumDict, fragIndexesSorted, notUnique, distsTrav)

        nextPossibleFragments = np.asarray(nextPossibleFragments)
        print '\nNext possible fragments for train', nextPossibleFragments

        accumDict = getAcceptableFragments(accumDict, nextPossibleFragments, distsTrav, distI)

    return accumDict
