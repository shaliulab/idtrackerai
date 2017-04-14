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
# sys.path.append('IdTrackerDeep/tracker')

from video import Video
from segmentation import *
# from fragmentation import *
# from get_portraits import *
from GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview, selectPreprocParams
from py_utils import getExistentFiles
from video_utils import checkBkg

# from idAssigner import *
# from fragmentFinder import *
# from fineTuner import *
# from tracker import *
# from plotters import *


if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...
    #select path to video
    video_path = selectFile()
    #instantiate object video
    video = Video()
    #set path
    video.video_path = video_path
    #############################################################
    ####################   Preprocessing   ######################
    #############################################################
    #Asking user whether to reuse preprocessing steps...'
    reUseAll = getInput('Reuse all preprocessing, ', 'Do you wanna reuse all previous preprocessing? ([y]/n)')

    if reUseAll == 'n':
        #Selecting preprocessing parameters
        prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
        video.subtract_bkg = bool(prepOpts['bkg'])
        video.apply_ROI =  bool(prepOpts['ROI'])
        print 'subtract background? ', video.subtract_bkg
        print 'apply ROI? ', video.apply_ROI

        print '\nLooking for finished steps in previous session...'
        processes_list = ['bkg', 'ROI', 'preprocparams', 'segmentation','fragments','portraits']
        #get existent files and paths to load them
        existentFiles = getExistentFiles(video, processes_list)
        print('existent files ', existentFiles)
        #selecting files to load from previous session...'
        loadPreviousDict = selectOptions(processes_list, existentFiles, text='Steps already processed in this video \n (loaded from ' + video._video_folder + ')')
        #use previous values and parameters (bkg, roi, preprocessing parameters)?
        usePreviousROI = loadPreviousDict['ROI']
        usePreviousBkg = loadPreviousDict['bkg']
        usePreviousPrecParams = loadPreviousDict['preprocparams']
        #ROI selection/loading
        video.ROI = ROISelectorPreview(video, usePreviousROI)
        #BKG computation/loading
        video.bkg = checkBkg(video, usePreviousBkg)
        #Selection/loading preprocessing parameters
        selectPreprocParams(video, usePreviousPrecParams)
        video.save()
        print('The video will be preprocessed according to the following parameters: ')
        pprint(video.__dict__)
        #Loading logo during preprocessing
        img = cv2.imread('../utils/loadingIdDeep.png')
        cv2.imshow('Bars',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        # the preprocessing parameters will be loaded from last time they were computed
        loadPreviousDict = {'bkg': 1, 'ROI': 1, 'preprocparams': 1,'segmentation': 1, 'fragments': 1, 'portraits': 1}
        video.Video()
        video.load()
    else:
        raise ValueError('The input introduced does not match the possible options')

    #############################################################
    ####################   Segmentation   #######################
    #############################################################
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not loadPreviousDict['segmentation']:
        print 'The parameters used to preprocess the video are '
        pprint(video.__dict__)
        segment(video)
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
#----------------------------------------------------------------------------->8
    # print '\n********************************************************************'
    # print 'Fragmentation'
    # print '********************************************************************\n'
    # if not loadPreviousDict['fragments']:
    #     dfGlobal, fragmentsDict = fragment(videoPaths, segmPaths, videoInfo=None)
    #
    #     playFragmentation(videoPaths,segmPaths,dfGlobal,visualize=False)
    #
    #     cv2.waitKey(1)
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)
    # else:
    #     dfGlobal = loadFile(videoPaths[0],'portraits')
    #     fragmentsDict = loadFile(videoPaths[0],'fragments',hdfpkl='pkl')
    #
    # print '\n********************************************************************'
    # print 'Portraying'
    # print '********************************************************************\n'
    # if not loadPreviousDict['portraits']:
    #     animal_type = preprocParams['animal_type']
    #     print 'you are going to track ', animal_type
    #     portraits = portrait(segmPaths,dfGlobal, animal_type)
    # else:
    #     portraits = loadFile(videoPaths[0], 'portraits')
    #
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    #
    # print '\n********************************************************************'
    # print 'Tracker'
    # print '********************************************************************\n'
    # preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
    # numAnimals = preprocParams['numAnimals']
    #
    # restoreFromAccPoint = getInput('Restore from a previous accumulation step','Do you want to restore from an accumulation point? y/[n]')
    #
    # if restoreFromAccPoint == 'n' or restoreFromAccPoint == '':
    #     accumDict, trainDict, handlesDict, statistics = initializeTracker(videoPath,numAnimals,portraits, preprocParams)
    #
    # elif restoreFromAccPoint == 'y':
    #     accumDict, trainDict, handlesDict, statistics = restoreTracker()
    #
    # else:
    #     raise ValueError('You typed ' + restoreFromAccPoint + ' the accepted values are y or n.')
    #
    # tracker(videoPath, fragmentsDict, portraits, accumDict, trainDict, handlesDict, statistics, numAnimals)
