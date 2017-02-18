# Import standard libraries
import os
from os.path import isdir, isfile
import sys
import glob
import numpy as np
import cPickle as pickle

# Import third party libraries
import cv2
from pprint import pprint

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/preprocessing')
sys.path.append('IdTrackerDeep/tracker')

from video_utils import *
from py_utils import *
from GUI_utils import *
from idAssigner import *
from fragmentFinder import *
from fineTuner import *

def initializeTracker(videoPath,numAnimals,portraits, preprocParams):

    loadCkpt_folder = selectDir('') #select where to load the model
    loadCkpt_folder = os.path.relpath(loadCkpt_folder)
    # inputs = getMultipleInputs('Training parameters', ['batch size', 'num. epochs', 'learning rate', 'train (1 (from strach) or 2 (from last check point))'])
    # print 'inputs, ', inputs
    print 'Entering into the fineTuner...'
    batchSize = 50 #int(inputs[1])
    numEpochs = 100 #int(inputs[2])
    lr = 0.01 #np.float32(inputs[3])
    train = 1 #int(inputs[4])

    ''' Initialization of variables for the accumulation loop'''
    sessionPath, figurePath = createSessionFolder(videoPath)
    pickle.dump( preprocParams , open( sessionPath + "/preprocparams.pkl", "wb" ))

    accumDict = {
            'counter': 0,
            'thVels': 0.5,
            'minDist': 0,
            'fragsForTrain': [], # to be saved
            'newFragForTrain': [],
            'badFragments': [], # to be saved
            'overallP2': [1./numAnimals],
            'continueFlag': True}

    trainDict = {
            'loadCkpt_folder':loadCkpt_folder,
            'ckpt_dir': '',
            'fig_dir': figurePath,
            'sess_dir': sessionPath,
            'batchSize': batchSize,
            'numEpochs': numEpochs,
            'lr': lr,
            'keep_prob': 1.,
            'train':train,
            'lossAccDict':{},
            'refDict':{},
            'framesColumnsRefDict': {}, #to be saved
            'usedIndivIntervals': [],
            'idUsedIntervals': []}

    handlesDict = {'restoring': False}

    statistics = {'fragmentIds':np.asarray(portraits['identities'])}

    return accumDict, trainDict, handlesDict, statistics

def restoreTracker():
    restoreFromAccPointPath = selectDir('./')

    if 'AccumulationStep_' not in restoreFromAccPointPath:
        raise ValueError('Select an AccumulationStep folder to restore from it.')
    else:
        countpkl = 0
        for file in os.listdir(restoreFromAccPointPath):
            if file.endswith(".pkl"):
                countpkl += 1
        if countpkl != 3:
            raise ValueError('It is not possible to restore from here. Select an accumulation point in which statistics.pkl, accumDict.pkl, and trainDict.pkl have been saved.')
        else:

            statistics = pickle.load( open( restoreFromAccPointPath + "/statistics.pkl", "rb" ) )
            accumDict = pickle.load( open( restoreFromAccPointPath + "/accumDict.pkl", "rb" ) )
            trainDict = pickle.load( open( restoreFromAccPointPath + "/trainDict.pkl", "rb" ) )

    handlesDict = {'restoring': True}
    return accumDict, trainDict, handlesDict, statistic

def tracker(videoPath, fragmentsDict, portraits, accumDict, trainDict, handlesDict, statistics, numAnimals):
      while accumDict['continueFlag']:

          print '\n*** Accumulation ', accumDict['counter'], ' ***'

          ''' Best fragment search '''
          accumDict = bestFragmentFinder(accumDict, statistics, fragmentsDict, numAnimals)

          pprint(accumDict)
          print '---------------\n'

          ''' Fine tuning '''
          trainDict, handlesDict = fineTuner(videoPath, accumDict, trainDict, fragmentsDict, handlesDict, portraits, statistics)

          print 'loadCkpt_folder ', trainDict['loadCkpt_folder']
          print 'ckpt_dir ', trainDict['ckpt_dir']
          print '---------------\n'

          ''' Identity assignation '''
          statistics = idAssigner(videoPath, trainDict, accumDict['counter'], fragmentsDict, portraits)

          ''' Updating training Dictionary'''
          trainDict['train'] = 2
          trainDict['numEpochs'] = 10000
          accumDict['counter'] += 1
          accumDict['overallP2'].append(statistics['overallP2'])

          # Variables to be saved in order to restore the accumulation
          print 'saving dictionaries to enable restore from accumulation'
          pickle.dump( accumDict , open( trainDict['ckpt_dir'] + "/accumDict.pkl", "wb" ) )
          pickle.dump( trainDict , open( trainDict['ckpt_dir'] + "/trainDict.pkl", "wb" ) )
          print 'dictionaries saved in ', trainDict['ckpt_dir']
