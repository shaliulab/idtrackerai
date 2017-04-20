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
sys.path.append('./utils')
sys.path.append('./preprocessing')
# sys.path.append('IdTrackerDeep/tracker')

from video import Video
from blob import connect_blob_list, apply_model_area_to_video
from globalfragment import compute_model_area, give_me_list_of_global_fragments
from segmentation import *
# from fragmentation import *
# from get_portraits import *
from GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview, selectPreprocParams, fragmentation_inspector
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
    #### video preprocessing through a simple GUI that       ####
    #### allows to set parameters as max/min area of the     ####
    #### blobs, max/min threshold and ROIs. All these        ####
    #### parameters are then saved, the GUI gives the        ####
    #### possibility to load them.                           ####
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
        existentFiles, old_video = getExistentFiles(video, processes_list)
        print('existent files ', existentFiles)
        #selecting files to load from previous session...'
        loadPreviousDict = selectOptions(processes_list, existentFiles, text='Steps already processed in this video \n (loaded from ' + video._video_folder + ')')
        #use previous values and parameters (bkg, roi, preprocessing parameters)?
        usePreviousROI = loadPreviousDict['ROI']
        usePreviousBkg = loadPreviousDict['bkg']
        usePreviousPrecParams = loadPreviousDict['preprocparams']
        #ROI selection/loading
        video.ROI = ROISelectorPreview(video, old_video, usePreviousROI)
        #BKG computation/loading
        video.bkg = checkBkg(video, old_video, usePreviousBkg)
        #Selection/loading preprocessing parameters
        selectPreprocParams(video, old_video, usePreviousPrecParams)
        video.save()
        print('The video will be preprocessed according to the following parameters: ')
        pprint({name : getattr(video, name) for name in video.__dict__.keys() if 'threshold'  in name or 'area'})
        #Loading logo during preprocessing
        img = cv2.imread('../utils/loadingIdDeep.png')
        cv2.imshow('Bars',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        # the preprocessing parameters will be loaded from last time they were computed
        processes_list = ['bkg', 'ROI', 'preprocparams', 'segmentation','fragments','portraits']
        existentFiles, old_video = getExistentFiles(video, processes_list)
        old_video = Video()
        video = np.load(old_video._name).item()
    else:
        raise ValueError('The input introduced does not match the possible options')

    #############################################################
    ####################   Segmentation   #######################
    #### detect blobs in the video according to parameters   ####
    #### specified by the user, and save them for future use ####
    #############################################################
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not loadPreviousDict['segmentation']:
        print 'The parameters used to preprocess the video are '
        # pprint(video.__dict__)
        blobs = segment(video)
        print('idTrackerDeepGUI line 102, has been segmented ', video._has_been_segmented)
        video.save()
    else:
        blobs = np.load(video.get_blobs_path())
        old_video = Video()
        old_video.video_path = video_path
        video = np.load(old_video._name).item()
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    #############################################################
    ####################   Fragmentation   ######################
    #### 1. create a list of potential global fragments      ####
    #### in which all animals are visible.                   ####
    #### 2. compute a model of the area of the animals       ####
    #### (mean and variance)                                 ####
    #### 3. identify global and individual fragments         ####
    #### 4. create a list of objects GlobalFragment() that   ####
    #### will be used to train the network                   ####
    #############################################################
    if not loadPreviousDict['fragments']:
        print('number of animals ', video.num_animals)
        # potential_global_fragments = give_me_list_of_potential_global_fragments(blobs, video._num_animals)
        model_area = compute_model_area(blobs, video.num_animals)
        apply_model_area_to_video(blobs, model_area)
        connect_blob_list(blobs)

        fragmentation_inspector(video, blobs)


        global_fragments = give_me_list_of_global_fragments(blobs, video.num_animals)
        #dfGlobal, fragmentsDict = fragment(videoPaths, segmPaths, videoInfo=None)
        # playFragmentation(videoPaths,segmPaths,dfGlobal,visualize=False)
        #
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
    else:
        dfGlobal = loadFile(videoPaths[0],'portraits')
        fragmentsDict = loadFile(videoPaths[0],'fragments',hdfpkl='pkl')
    #

#----------------------------------------------------------------------------->8

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
