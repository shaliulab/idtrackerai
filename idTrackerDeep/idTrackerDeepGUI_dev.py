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
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../tracker')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *
from idAssigner import *
from fragmentFinder import *
from fineTuner import *
from plotters import *

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    print '\n********************************************************************'
    print 'Selecting the path to the videos...'
    print '********************************************************************\n'

    initialDir = ''
    videoPath = selectFile() ### NOTE The video to be tracked need to be splited in to small segments of video with a suffix '_(numSegment)'. The video selected has to be the one with extension suffix '_1'
    print 'The video selected is, ', videoPath
    videoPaths = scanFolder(videoPath) ### FIXME if the video selected does not finish with '_1' the scanFolder function won't select all of them. This can be improved
    print 'The list of videos is ', videoPaths

    print '\n********************************************************************'
    print 'Asking user whether to reuse preprocessing steps...'
    print '********************************************************************\n'
    reUseAll = getInput('Reuse all preprocessing, ', 'Do you wanna reuse all previos preprocessing? ([y]/n)')

    if reUseAll == 'n':
        print '\n********************************************************************'
        print 'Selecting preprocessing parameters...'
        print '********************************************************************\n'

        prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
        useBkg = int(prepOpts['bkg'])
        useROI =  int(prepOpts['ROI'])
        print 'useBkg set to ', useBkg
        print 'useROI set to ', useROI

        print '\nLooking for finished steps in previous session...'
        processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragments','portraits']

        existentFiles, srcSubFolder = getExistentFiles(videoPath, processesList)
        print 'List of processes finished, ', existentFiles
        print '\nSelecting files to load from previous session...'
        loadPreviousDict = selectOptions(processesList, existentFiles, text='Already processed steps in this video \n (check to load from ' + srcSubFolder + ')')

        usePreviousROI = loadPreviousDict['ROI']
        usePreviousBkg = loadPreviousDict['bkg']
        usePreviousPrecParams = loadPreviousDict['preprocparams']
        print 'usePreviousROI set to ', usePreviousROI
        print 'usePreviousBkg set to ', usePreviousBkg
        print 'usePreviousPrecParams set to ', usePreviousPrecParams

        ''' ROI selection/loading '''
        width, height, mask, centers = ROISelectorPreview(videoPaths, useROI, usePreviousROI, numSegment=0)
        ''' BKG computation/loading '''
        bkg = checkBkg(videoPaths, useBkg, usePreviousBkg, 0, width, height)

        ''' Selection/loading preprocessing parameters '''
        preprocParams = selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg)
        print 'The video will be preprocessed according to the following parameters: ', preprocParams

        ''' Loading preprocessing image '''
        img = cv2.imread('../utils/loadingIdDeep.png')
        cv2.imshow('Bars',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        print '\n********************************************************************'
        print 'The preprocessing paramemters will be loaded from last time they were computed.'
        print '********************************************************************\n'
        loadPreviousDict = {'ROI': 1, 'bkg': 1, 'preprocparams': 1, 'segmentation': 1, 'fragments': 1, 'portraits': 1}

    else:
        raise ValueError('The input introduced do not match the possible options')

    print '\n********************************************************************'
    print 'Segmentation'
    print '********************************************************************\n'
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not loadPreviousDict['segmentation']:
        preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
        EQ = 0
        print 'The preprocessing parameters dictionary loaded is ', preprocParams
        segment(videoPaths, preprocParams, mask, centers, useBkg, bkg, EQ)

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print '\n********************************************************************'
    print 'Fragmentation'
    print '********************************************************************\n'
    if not loadPreviousDict['fragments']:
        dfGlobal, fragmentsDict = fragment(videoPaths,videoInfo=None)

        playFragmentation(videoPaths,dfGlobal,visualize=False)

        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        dfGlobal = loadFile(videoPaths[0],'portraits')
        fragmentsDict = loadFile(videoPaths[0],'fragments',hdfpkl='pkl')

    print '\n********************************************************************'
    print 'Portraying'
    print '********************************************************************\n'
    if not loadPreviousDict['portraits']:
        portraits = portrait(videoPaths,dfGlobal)
    else:
        portraits = loadFile(videoPaths[0], 'portraits')

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print '\n********************************************************************'
    print 'Tracker'
    print '********************************************************************\n'
    preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
    numAnimals = preprocParams['numAnimals']

    restoreFromAccPoint = getInput('Restore from a previous accumulation step','Do you want to restore from an accumulation point? y/[n]')

    if restoreFromAccPoint == 'n' or restoreFromAccPoint == '':
        loadCkpt_folder = selectDir(initialDir) #select where to load the model
        loadCkpt_folder = os.path.relpath(loadCkpt_folder)
        # inputs = getMultipleInputs('Training parameters', ['batch size', 'num. epochs', 'learning rate', 'train (1 (from strach) or 2 (from last check point))'])
        # print 'inputs, ', inputs
        print 'Entering into the fineTuner...'
        batchSize = 50 #int(inputs[1])
        numEpochs = 100 #int(inputs[2])
        lr = 0.01 #np.float32(inputs[3])
        train = 1 #int(inputs[4])

        ''' Initialization of variables for the accumulation loop'''
        def createSessionFolder(videoPath):
            def getLastSession(subFolders):
                if len(subFolders) == 0:
                    lastIndex = 0
                else:
                    subFolders = natural_sort(subFolders)[::-1]
                    lastIndex = int(subFolders[0].split('_')[-1])
                return lastIndex

            video = os.path.basename(videoPath)
            folder = os.path.dirname(videoPath)
            filename, extension = os.path.splitext(video)
            subFolder = folder + '/CNN_models'
            subSubFolders = glob.glob(subFolder +"/*")
            lastIndex = getLastSession(subSubFolders)
            sessionPath = subFolder + '/Session_' + str(lastIndex + 1)
            os.makedirs(sessionPath)
            print 'You just created ', sessionPath
            figurePath = sessionPath + '/figures'
            os.makedirs(figurePath)
            print 'You just created ', figurePath

            return sessionPath, figurePath

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

        normFreqFragments = None
    elif restoreFromAccPoint == 'y':
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
                normFreqFragments = statistics['normFreqFragsAll']
                portraits = accumDict['portraits']

        handlesDict = {'restoring': True}
    else:
        raise ValueError('You typed ' + restoreFromAccPoint + ' the accepted values are y or n.')

    while accumDict['continueFlag']:
        print '\n*** Accumulation ', accumDict['counter'], ' ***'

        ''' Best fragment search '''
        accumDict = bestFragmentFinder(accumDict, normFreqFragments, fragmentsDict, numAnimals, portraits)
        # fragmentAccumPlotter(fragmentsDict,portraits,accumDict,figurePath)

        pprint(accumDict)
        print '---------------\n'

        ''' Fine tuning '''
        trainDict, handlesDict = fineTuner(videoPath, accumDict, trainDict, fragmentsDict, handlesDict, portraits)

        print 'loadCkpt_folder ', trainDict['loadCkpt_folder']
        print 'ckpt_dir ', trainDict['ckpt_dir']
        print '---------------\n'

        ''' Identity assignation '''
        normFreqFragments, portraits, overallP2 = idAssigner(videoPath, trainDict, accumDict['counter'], fragmentsDict, portraits)

        # P2AccumPlotter(fragmentsDict,portraits,accumDict,figurePath,trainDict['ckpt_dir'])

        ''' Updating training Dictionary'''
        trainDict['train'] = 2
        trainDict['numEpochs'] = 10000
        accumDict['counter'] += 1
        accumDict['portraits'] = portraits
        accumDict['overallP2'].append(overallP2)
        # Variables to be saved in order to restore the accumulation
        print 'saving dictionaries to enable restore from accumulation'
        pickle.dump( accumDict , open( trainDict['ckpt_dir'] + "/accumDict.pkl", "wb" ) )
        pickle.dump( trainDict , open( trainDict['ckpt_dir'] + "/trainDict.pkl", "wb" ) )
        print 'dictionaries saved in ', trainDict['ckpt_dir']
