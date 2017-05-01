# Import standard libraries
import os
from os.path import isdir, isfile
import sys
sys.setrecursionlimit(10000)
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
from globalfragment import compute_model_area, give_me_list_of_global_fragments, ModelArea, order_global_fragments_by_distance_travelled, give_me_pre_training_global_fragments
from segmentation import segment
from GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview, selectPreprocParams, fragmentation_inspector
from py_utils import getExistentFiles
from video_utils import checkBkg
from pre_trainer import pre_train
from cnn_config import Network_Params

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
    #### parameters are then saved. The same GUI gives the   ####
    #### possibility to load them.                           ####
    #############################################################
    #Asking user whether to reuse preprocessing steps...'
    reUseAll = getInput('Reuse all preprocessing, ', 'Do you wanna reuse all previous preprocessing? ([y]/n)')
    processes_list = ['bkg', 'ROI', 'preprocparams', 'preprocessing','pretraining']
    #get existent files and paths to load them
    existentFiles, old_video = getExistentFiles(video, processes_list)
    if reUseAll == 'n':
        #Selecting preprocessing parameters
        prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
        video.subtract_bkg = bool(prepOpts['bkg'])
        video.apply_ROI =  bool(prepOpts['ROI'])
        print '\nLooking for finished steps in previous session...'

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
        print 'The parameters used to preprocess the video are '
        pprint({key: getattr(video, key) for key in video.__dict__ if 'ROI' in key or 'bkg' in key})
        #Loading logo during preprocessing
        img = cv2.imread('../utils/loadingIdDeep.png')
        cv2.imshow('Bars',img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        old_video = Video()
        video = np.load(old_video._path_to_video_object).item()
    else:
        raise ValueError('The input introduced does not match the possible options')

    #############################################################
    ####################  Preprocessing   #######################
    #### 1. detect blobs in the video according to parameters####
    #### specified by the user                               ####
    #### 2. create a list of potential global fragments      ####
    #### in which all animals are visible.                   ####
    #### 3. compute a model of the area of the animals       ####
    #### (mean and variance)                                 ####
    #### 4. identify global fragments                        ####
    #### 5. create a list of objects GlobalFragment() that   ####
    #### will be used to (pre)train the network              ####
    #### save them for future use                            ####
    #############################################################
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    if not loadPreviousDict['preprocessing']:
        blobs = segment(video)
        np.save(video.blobs_path,blobs)
        #compute a model of the area of the animals (considering frames in which
        #all the animals are visible)
        model_area = compute_model_area(blobs, video.num_animals)
        #discard blobs that do not respect such model
        apply_model_area_to_video(blobs, model_area)
        #connect blobs that overlap in consecutive frames
        connect_blob_list(blobs)
        #compute the global fragments (all animals are visible + each animals overlaps
        #with a single blob in the consecutive frame + the blobs respect the area model)
        global_fragments = give_me_list_of_global_fragments(blobs, video.num_animals)
        #save connected blobs in video (organized frame-wise) and list of global fragments
        video._has_been_preprocessed = True
        np.save(video.global_fragments_path, global_fragments)
        np.save(video.blobs_path,blobs)
        video.save()
        #take a look to the resulting fragmentation
        fragmentation_inspector(video, blobs)
    else:
        old_video = Video()
        old_video.video_path = video_path
        video = np.load(old_video._path_to_video_object).item()
        blobs = np.load(video.blobs_path)
        global_fragments = np.load(video.global_fragments_path)
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    #############################################################
    ####################      Tracker      ######################
    ####
    #############################################################
    #create the folder training in which all the CNN-related process will be
    #stored. The structure is /training/session_num, where num is an natural number.
    # num increases each time a training is launched on the video.
    if not loadPreviousDict['pretraining']:
        video.create_training_and_session_folder()
        pretrain_flag = getInput('Pretraining','Do you want to perform pretraining? [Y/n]')
        if pretrain_flag == 'y' or pretrain_flag == '':
            #set pretraining parameters
            number_of_global_fragments = getInput('Pretraining','Choose the number of global fragments that will be used to pretrain the network. Default 10')
            try:
                number_of_global_fragments = int(number_of_global_fragments)
            except:
                number_of_global_fragments = 10
            print("pretraining with %i" %number_of_global_fragments, ' global fragments')
            video.create_pretraining_folder(number_of_global_fragments)
            #pretraining network parameters
            pretrain_network_params = Network_Params(video,
                                                    learning_rate = 0.005,
                                                    keep_prob = 1.0,
                                                    use_adam_optimiser = False,
                                                    scopes_layers_to_optimize = [],
                                                    restore_folder = None,
                                                    save_folder = video._pretraining_path,
                                                    knowledge_transfer_folder = video._pretraining_path)
            #start pretraining
            pre_train(global_fragments,
                    number_of_global_fragments,
                    pretrain_network_params,
                    store_accuracy_and_error = False,
                    check_for_loss_plateau = True,
                    save_summaries = False,
                    print_flag = False)
            #save changes
            video._has_been_pretrained = True
            video.save()
    else:
        old_video = Video()
        old_video.video_path = video_path
        video = np.load(old_video._path_to_video_object).item()
        blobs = np.load(video.blobs_path)
        global_fragments = np.load(video.global_fragments_path)
        



#----------------------------------------------------------------------------->8
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
