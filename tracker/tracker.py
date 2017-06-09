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

from cnn_utils import getCkptvideoPath
from video_utils import *
from py_utils import *
from GUI_utils import *
from idAssigner import *
from fragmentFinder import *
from fineTuner import *
from idTrainerTracker import run_training

def initializeTracker(videoPath,numAnimals,portraits, preprocParams):

    pretrain_flag = getInput('Pre-training','Do you want to pre-train? [y]/n')
    if pretrain_flag == 'n':
        loadCkpt_folder = selectDir('') #select where to load the model
        loadCkpt_folder = os.path.relpath(loadCkpt_folder)
    elif pretrain_flag == 'y' or pretrain_flag == '':
        loadCkpt_folder = ''
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
            'thVels': 0., #FIXME: the vel threshold should be deduced from the distribution of velocity in the video...
            'minDist': 0,
            'fragsForTrain': [], # to be saved
            'newFragForTrain': [],
            'badFragments': [], # to be saved
            'overallP2': [1./numAnimals],
            'continueFlag': True}

    trainDict = {
            'load_ckpt_folder':loadCkpt_folder,
            'pre_train_ckpt_dir': '',
            'ckpt_dir': '',
            'fig_dir': figurePath,
            'sess_dir': sessionPath,
            'batchSize': batchSize,
            'numEpochs': numEpochs,
            'lr': lr,
            'keep_prob': 1.0,
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
    return accumDict, trainDict, handlesDict, statistics

def tracker(videoPath, fragmentsDict, portraits, accumDict, trainDict, handlesDict, statistics, numAnimals):

    print 'Starting tracker...'
    if not trainDict['load_ckpt_folder']: # if not knowledge transfer
        print '\n*** we are going to pre-train from the global fragments ***'
        pre_tracking(videoPath, fragmentsDict, portraits, trainDict, numAnimals)

    while accumDict['continueFlag']:

        print '\n*** Accumulation ', accumDict['counter'], ' ***'

        ''' Best fragment search '''
        accumDict = bestFragmentFinder(accumDict, trainDict, statistics, fragmentsDict, numAnimals)

        pprint(accumDict)
        print '---------------\n'

        ''' Fine tuning '''
        trainDict, handlesDict = fineTuner(videoPath, accumDict, trainDict, fragmentsDict, handlesDict, portraits, statistics)

        print 'loadCkpt_folder ', trainDict['load_ckpt_folder']
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

def pre_tracking(videoPath, fragmentsDict, portraits, trainDict, numAnimals):
    """The idea is to catch the variability by training the network on all global
    fragments. Then, reinitialising the softmax we will start the tracking.

    :param videoPath: path to the video (segment or entire video)
    :param fragmentsDict: dictionary in which the global fragments are stored
           (fragmentsDict['intervalsDist'])
    :param portraits: images
    :param trainDict: dictionary with training parameters (e. g. learning rate, ...)
    :param numAnimals: number of animals to be tracked
    """
    #get ckpt folder for pre-training
    trainDict['pre_train_ckpt_dir'] = getCkptvideoPath(videoPath, -1, train=0)#negative value for pre-training
    batchSize = trainDict['batchSize']

    #take the list of global fragments ordered by distance travelled (you don't want to pre-train starting with a bad global frag)
    framesAndBlobColumns = fragmentsDict['framesAndBlobColumnsDist']
    #loop on the ordered global fragments
    used_individual_frags = []
    non_overlapping_counter = 0

    for k, global_fragment in enumerate(fragmentsDict['intervalsDist']):
        print 'global fragments to go: ', 30 - non_overlapping_counter
        if non_overlapping_counter < 30:
            if len(set(global_fragment).intersection(set(used_individual_frags))) == 0:
                print '\nPre-training on global fragment ', k
                #create images and labels to train the model
                images = []
                labels = []
                intervalsIndivFrags = global_fragment # I take the list of individual fragments in terms of intervals
                framesColumnsIndivFrags = framesAndBlobColumns[k] # I take the list of individual fragments in frames and columns

                print 'collecting images from individual fragments...'
                for i, (framesColumnsIndivFrag,intervalsIndivFrag) in enumerate(zip(framesColumnsIndivFrags,intervalsIndivFrags)):
                    used_individual_frags.append(intervalsIndivFrag)
                    framesColumnsIndivFrag = np.asarray(framesColumnsIndivFrag)
                    frames = framesColumnsIndivFrag[:,0]
                    columns = framesColumnsIndivFrag[:,1]
                    # I loop in all the frames of the individual fragment to add them to the dictionary of references
                    for frame,column in zip(frames,columns):
                        labels.append(i)
                        images.append(portraits.loc[frame,'images'][column])

                print 'preparing data...'
                labels = dense_to_one_hot(labels, numAnimals)
                images = np.expand_dims(images,axis=3)
                images = cropImages(images,32)
                images = standarizeImages(images)
                numImages = len(labels)
                perm = np.random.permutation(numImages)
                images = images[perm]
                labels = labels[perm]
                numTrain = int(0.9*numImages)
                numVal = numImages - numTrain
                X_t = images[:numTrain]
                Y_t = labels[:numTrain]
                X_v = images[numTrain:]
                Y_v = labels[numTrain:]
                Tindices, Titer_per_epoch = get_batch_indices(numTrain,batchSize)
                Vindices, Viter_per_epoch = get_batch_indices(numVal,batchSize)
                height, width, channels = images.shape[1:]

                print 'entering run_pre_training'
                trainDict = run_pre_training(X_t, Y_t, X_v, Y_v,
                                            width, height, channels, numAnimals,
                                            trainDict,
                                            Tindices, Titer_per_epoch, Vindices, Viter_per_epoch,
                                            plotFlag = True,
                                            printFlag = True,
                                            onlySoftmax=False,
                                            weighted_flag = True)
                non_overlapping_counter += 1
                print 'pre-training for global fragment ', k, ' finished\n'
            else:
                print '\nGlobal fragment ', k, ' will be discarded.'
                continue
        else:
            break
