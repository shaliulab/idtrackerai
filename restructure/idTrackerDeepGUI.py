# Import standard libraries
import os
from os.path import isdir, isfile
import sys
sys.setrecursionlimit(100000)
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
from blob import connect_blob_list, apply_model_area_to_video, ListOfBlobs, get_images_from_blobs_in_video
from globalfragment import compute_model_area, give_me_list_of_global_fragments, ModelArea, order_global_fragments_by_distance_travelled, give_me_pre_training_global_fragments, assign_identity_to_global_fragment_used_for_training
from globalfragment import get_images_and_labels_from_global_fragments, get_images_and_labels_from_global_fragment, get_images_from_test_global_fragments, assign_identities_and_check_eligibility_for_training_global_fragments, split_predictions_after_network_assignment
from segmentation import segment
from GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview, selectPreprocParams, fragmentation_inspector, frame_by_frame_identity_inspector
from py_utils import getExistentFiles
from video_utils import checkBkg
from pre_trainer import pre_train
from network_params import NetworkParams
from trainer import train
from assigner import assign, assign_identity_to_blobs_in_video, assign_identity_to_blobs_in_fragment
from visualize_embeddings import visualize_embeddings_global_fragments

NUM_CHUNKS_BLOB_SAVING = 10 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
NUMBER_OF_SAMPLES = 3000

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...
    video_path = selectFile() #select path to video
    video = Video() #instantiate object video
    video.video_path = video_path #set path
    video.create_session_folder()
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
    processes_list = ['bkg', 'ROI', 'preprocparams', 'preprocessing', 'pretraining', 'accumulation', 'training', 'assignment']
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
            video.create_preprocessing_folder()
            blobs = segment(video)
            #compute a model of the area of the animals (considering frames in which
            #all the animals are visible)
            model_area = compute_model_area(blobs, video.number_of_animals)
            #discard blobs that do not respect such model
            apply_model_area_to_video(blobs, model_area)
            #connect blobs that overlap in consecutive frames
            connect_blob_list(blobs)
            #compute the global fragments (all animals are visible + each animals overlaps
            #with a single blob in the consecutive frame + the blobs respect the area model)
            global_fragments = give_me_list_of_global_fragments(blobs, video.number_of_animals)
            #save connected blobs in video (organized frame-wise) and list of global fragments
            video._has_been_preprocessed = True
            saved = False
            np.save(video.global_fragments_path, global_fragments)
            video.save()
            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
            blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            #take a look to the resulting fragmentation
            fragmentation_inspector(video, blobs)
        else:
            # old_video = Video()
            old_video.video_path = video_path
            video = np.load(old_video._path_to_video_object).item()
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video
            global_fragments = np.load(video.global_fragments_path)
        #destroy windows to prevent openCV errors
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #############################################################
        ##################      Pre-trainer      ####################
        #### create the folder training in which all the         ####
        #### CNN-related process will be stored. The structure   ####
        #### is /training/session_num, where num is an natural   ####
        #### number. num increases each time a training is       ####
        #### launched                                            ####
        #############################################################
        if not loadPreviousDict['pretraining']:
            pretrain_flag = getInput('Pretraining','Do you want to perform pretraining? [Y/n]')
            if pretrain_flag == 'y' or pretrain_flag == '':
                #set pretraining parameters
                number_of_global_fragments = getInput('Pretraining','Choose the number of global fragments that will be used to pretrain the network. Default 10')
                try:
                    number_of_global_fragments = int(number_of_global_fragments)
                    print("Pretraining with ", number_of_global_fragments , " fragments")
                    pretraining_global_fragments = order_global_fragments_by_distance_travelled(give_me_pre_training_global_fragments(global_fragments, number_of_global_fragments = number_of_global_fragments))

                except:
                    number_of_global_fragments = len(global_fragments)
                    pretraining_global_fragments = order_global_fragments_by_distance_travelled(global_fragments)

                print("pretraining with %i" %number_of_global_fragments, ' global fragments')
                video.create_pretraining_folder(number_of_global_fragments)
                #pretraining network parameters
                pretrain_network_params = NetworkParams(video,
                                                        learning_rate = 0.01,
                                                        keep_prob = 1.0,
                                                        use_adam_optimiser = False,
                                                        scopes_layers_to_optimize = None,
                                                        restore_folder = None,
                                                        save_folder = video._pretraining_folder,
                                                        knowledge_transfer_folder = video._pretraining_folder)
                #start pretraining
                pre_train(pretraining_global_fragments,#TODO: change it to get images as input
                        number_of_global_fragments,
                        pretrain_network_params,
                        store_accuracy_and_error = False,
                        check_for_loss_plateau = True,
                        save_summaries = True,
                        print_flag = True,
                        plot_flag = True)
                #save changes
                video._has_been_pretrained = True
                video.save()
        else:
            old_video = Video()
            old_video.video_path = video_path
            video = np.load(old_video._path_to_video_object).item()
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video
            global_fragments = np.load(video.global_fragments_path)
        #############################################################
        ###################    Accumulation   #######################
        #### take references in 'good' global fragments          ####
        #############################################################
        if not loadPreviousDict['accumulation']:
            video.create_accumulation_folder()
            train_network_params = NetworkParams(video,
                                                learning_rate = 0.005,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = ['fully-connected1','softmax1'],
                                                restore_folder = None,
                                                save_folder = video._accumulation_folder,
                                                knowledge_transfer_folder = video._pretraining_folder)
            first_training_global_fragment = order_global_fragments_by_distance_travelled(global_fragments)[0]
            max_distance_travelled = first_training_global_fragment.min_distance_travelled
            images, labels = get_images_and_labels_from_global_fragment(first_training_global_fragment)
            train(images, labels,
                train_network_params,
                store_accuracy_and_error = False,
                check_for_loss_plateau = True,
                save_summaries = True,
                print_flag = True,
                plot_flag = True)
            first_training_global_fragment._used_for_training = True
            first_training_global_fragment.acceptable_for_training = False
            assign_identity_to_global_fragment_used_for_training(first_training_global_fragment, blobs)
            accumulation_network_params = NetworkParams(video,
                                                learning_rate = 0.005,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = ['fully-connected1','softmax1'],
                                                restore_folder = video._accumulation_folder,
                                                save_folder = video._accumulation_folder,
                                                knowledge_transfer_folder = None)
            assign_network_params = NetworkParams(video,restore_folder = video._accumulation_folder)
            global_fragments_for_training = ['first accumulation']

            while True:
                print("\n\n*** New accumulation ***")
                #take images from global fragments not used in training (in the remainder test global fragments)
                images = get_images_from_test_global_fragments(global_fragments)
                # get predictions for images in test global fragments
                assigner = assign(video, images, assign_network_params, print_flag = True)
                split_predictions_after_network_assignment(global_fragments, assigner._predictions, assigner._softmax_probs)
                # assign identities to the global fragments based on the predictions
                assign_identities_and_check_eligibility_for_training_global_fragments(global_fragments, video.number_of_animals)
                # get global fragments for training
                global_fragments_for_training = [global_fragment for global_fragment in global_fragments
                                                    if global_fragment.acceptable_for_training == True]
                if len(global_fragments_for_training) == 0:
                    break
                print("Number of global fragments for training, ", len(global_fragments_for_training))
                # get images of globalfragments for training
                images, labels = get_images_and_labels_from_global_fragments(global_fragments_for_training)
                print("Images, ", images.shape)
                print("Labels, ", labels.shape)
                train(images, labels,
                    accumulation_network_params,
                    store_accuracy_and_error = False,
                    check_for_loss_plateau = True,
                    save_summaries = True,
                    print_flag = True,
                    plot_flag = True)

                for global_fragment_for_training in global_fragments_for_training:
                    global_fragment_for_training._used_for_training = True
                    global_fragment_for_training.acceptable_for_training = False
                    assign_identity_to_global_fragment_used_for_training(global_fragment_for_training, blobs)

            video.accumulation_finished = True
            video.save()
            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
            blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            np.save(video.global_fragments_path, global_fragments)
        else:
            old_video = Video()
            old_video.video_path = video_path
            video = np.load(old_video._path_to_video_object).item()
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video
            global_fragments = np.load(video.global_fragments_path)

        #############################################################
        ###################     Training       ######################
        ####
        #############################################################
        if not loadPreviousDict['training']:
            video.create_training_folder()
            global_fragments_used_for_training = [global_fragment for global_fragment in global_fragments
                                                    if global_fragment._used_for_training == True]
            minimum_number_of_portraits_per_individual_in_training = min(np.sum([global_fragment._number_of_portraits_per_individual_fragment[global_fragment._temporary_ids]
                                                                for global_fragment in global_fragments_used_for_training], axis = 0))
            images, labels = get_images_and_labels_from_global_fragments(global_fragments_used_for_training)
            number_of_samples = NUMBER_OF_SAMPLES
            if minimum_number_of_portraits_per_individual_in_training < number_of_samples:
                number_of_samples = minimum_number_of_portraits_per_individual_in_training
            subsampled_images, subsampled_labels = subsample_images_for_last_training(images, labels, number_of_animals, number_of_samples = number_of_samples)
            last_train_network_params = NetworkParams(video,
                                                learning_rate = 0.005,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = ['fully-connected1','softmax1'],
                                                restore_folder = video._accumulation_folder,
                                                save_folder = video._final_training_folder,
                                                knowledge_transfer_folder = None)
            train(subsampled_images, subsampled_labels,
                last_train_network_params,
                store_accuracy_and_error = False,
                check_for_loss_plateau = True,
                save_summaries = True,
                print_flag = True,
                plot_flag = True)
            video.training_finished = True
            video.save()
        else:
            old_video = Video()
            old_video.video_path = video_path
            video = np.load(old_video._path_to_video_object).item()
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video
            global_fragments = np.load(video.global_fragments_path)
        #############################################################
        ###################     Assigner      ######################
        ####
        #############################################################
        if not loadPreviousDict['assignment']:
            assign_network_params = NetworkParams(video, restore_folder = video._final_training_folder)
            # Get images from the blob collection
            images = get_images_from_blobs_in_video(blobs, video._episodes_start_end)
            # get predictions
            assigner = assign(video, images, assign_network_params, print_flag = True)
            # assign identities to each blob in each frame
            assign_identity_to_blobs_in_video(blobs, assigner)
            # assign identities based on individual fragments
            assign_identity_to_blobs_in_fragment(video, blobs)
            # visualise proposed tracking
            frame_by_frame_identity_inspector(video, blobs)
            # finish and save
            video.has_been_assigned = True
            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
            blobs_list.generate_cut_points(10)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            video.save()
        else:
            old_video = Video()
            old_video.video_path = video_path
            video = np.load(old_video._path_to_video_object).item()
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        old_video = Video()
        old_video.video_path = video_path
        video = np.load(old_video._path_to_video_object).item()
        list_of_blobs = ListOfBlobs.load(video.blobs_path)
        blobs = list_of_blobs.blobs_in_video
        global_fragments = np.load(video.global_fragments_path)
        frame_by_frame_identity_inspector(video, blobs)
    else:
        raise ValueError('The input introduced does not match the possible options')
