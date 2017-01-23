import sys
sys.path.append('../utils')
sys.path.append('../CNN')
sys.path.append('../tracker')

from tracker_indFrags  import *
from py_utils import *
from video_utils import *
from idTrainerTracker import *

import time
import numpy as np
np.set_printoptions(precision=2)
import numpy.matlib
import argparse
import os
import glob
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import cPickle as pickle
import tensorflow as tf
from tf_utils import *
from input_data_cnn import *
from cnn_utils import *
from pprint import pprint
from collections import Counter
import collections
from itertools import groupby

videoPath = '../Conflict8/conflict3and4_20120316T155032_1.avi'

"""Load fragmentsDict"""
fragmentsDict = loadFile(videoPath, 'fragments', time=0, hdfpkl = 'pkl')

# fragmentsDict = {
#     'fragments': fragments, #global fragments
#     'minLenIndivCompleteFragments': minLenIndivCompleteFragments,
#     'framesAndBlobColumns': framesAndBlobColumns,
#     'intervals': intervalsFragments,
#     'oneIndivFragIntervals': oneIndivFragIntervals,
#     'oneIndivFragFrames': oneIndivFragFrames,
#     'oneIndivFragLens': oneIndivFragLens,
#     'oneIndivFragSumLens': oneIndivFragSumLens
#     }


"""Load statistics"""
statistics = loadFile(videoPath, 'statistics', time=0, hdfpkl = 'pkl')

# IdsStatistics = {'blobIds':idSoftMaxAllVideo,
#     'probBlobIds':PSoftMaxAllVIdeo,
#     'FreqFrag': freqFragAllVideo,
#     'normFreqFragAllVideo': normFreqFragAllVideo,
#     'idFreqFragAllVideo': idFreqFragAllVideo,
#     'P1Frag': P1FragAllVideo,
#     'fragmentIds':idLogP2FragAllVideo,
#     'probFragmentIds':logP2FragAllVideo,
#     'P2FragAllVideo':P2FragAllVideo, # by single frames
#     'P2FragsAll': P2FragsAll, # organized by individual fragments
#     'overallP2': overallP2}

# Set initial list of individual fragments from the longer global fragment
indivFragsForTrain = fragmentsDict['intervals'][0]
indivFragsForTrainNew = {}
for indivFragForTrain in indivFragsForTrain:
    identity = indivFragForTrain[0]
    P2 = None # P2 is None because it is the first time and we have not computed it yet
    blobIndex = indivFragForTrain[0]
    fragNum = indivFragForTrain[1]
    start = indivFragForTrain[2][0]
    end = indivFragForTrain[2][1]
    length = indivFragForTrain[3]
    indivFragsForTrainNew[identity] = [(identity,P2,blobIndex,fragNum,start,end,length)]
indivFragsForTrain = indivFragsForTrainNew

print '************************************'
print indivFragsForTrain

normFreqFragsAll = statistics['normFreqFragAllVideo']
numAnimals = 8
P2FragsAll = statistics['P2FragsAll']

indivFragsForTrain, continueFlag = bestFragmentFinder(indivFragsForTrain,normFreqFragsAll,fragmentsDict,numAnimals,P2FragsAll)
