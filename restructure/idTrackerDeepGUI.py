from __future__ import absolute_import, division, print_function
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
from blob import compute_fragment_identifier_and_blob_index,\
                connect_blob_list,\
                apply_model_area_to_video,\
                ListOfBlobs,\
                get_images_from_blobs_in_video,\
                reset_blobs_fragmentation_parameters
from globalfragment import compute_model_area_and_body_length,\
                            give_me_list_of_global_fragments,\
                            ModelArea,\
                            give_me_pre_training_global_fragments,\
                            get_images_and_labels_from_global_fragments,\
                            subsample_images_for_last_training,\
                            order_global_fragments_by_distance_travelled,\
                            filter_global_fragments_by_minimum_number_of_frames
from segmentation import segment
from GUI_utils import selectFile,\
                    getInput,\
                    selectOptions,\
                    ROISelectorPreview,\
                    selectPreprocParams,\
                    fragmentation_inspector,\
                    frame_by_frame_identity_inspector,\
                    selectDir
from py_utils import getExistentFiles
from video_utils import checkBkg
from pre_trainer import pre_train
from accumulation_manager import AccumulationManager
from network_params import NetworkParams
from trainer import train
from assigner import assign,\
                    assign_identity_to_blobs_in_video,\
                    compute_P1_for_blobs_in_video,\
                    assign_identity_to_blobs_in_video_by_fragment
from visualize_embeddings import visualize_embeddings_global_fragments
from id_CNN import ConvNetwork


NUM_CHUNKS_BLOB_SAVING = 500 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
NUMBER_OF_SAMPLES = 30000
###
np.random.seed(0)
###
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
    processes_list = ['bkg', 'ROI', 'preprocparams', 'preprocessing', 'pretraining', 'accumulation', 'assignment']
    #get existent files and paths to load them
    existentFiles, old_video = getExistentFiles(video, processes_list)
    if reUseAll == 'n':
        #############################################################
        ############ Select preprocessing parameters   ##############
        ####                                                     ####
        #############################################################
        prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
        video.subtract_bkg = bool(prepOpts['bkg'])
        video.apply_ROI =  bool(prepOpts['ROI'])
        print('\nLooking for finished steps in previous session...')
        #selecting files to load from previous session...'
        loadPreviousDict = selectOptions(processes_list, existentFiles, text='Steps already processed in this video \n (loaded from ' + video._video_folder + ')')
        #use previous values and parameters (bkg, roi, preprocessing parameters)?
        usePreviousROI = loadPreviousDict['ROI']
        usePreviousBkg = loadPreviousDict['bkg']
        usePreviousPrecParams = loadPreviousDict['preprocparams']
        print("video session folder, ", video._session_folder)
        #ROI selection/loading
        video.ROI = ROISelectorPreview(video, old_video, usePreviousROI)
        print("video session folder, ", video._session_folder)
        #BKG computation/loading
        video.bkg = checkBkg(video, old_video, usePreviousBkg)
        print("video session folder, ", video._session_folder)
        #Selection/loading preprocessing parameters
        selectPreprocParams(video, old_video, usePreviousPrecParams)
        print("video session folder, ", video._session_folder)
        video.save()
        print('The parameters used to preprocess the video are ')
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
            print("\n**** Preprocessing ****\n")
            cv2.namedWindow('Bars')
            video.create_preprocessing_folder()
            if not old_video or not old_video._has_been_segmented or usePreviousPrecParams == False:
                blobs = segment(video)
                video._has_been_segmented = True
                blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path_segmented)
                blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
                blobs_list.cut_in_chunks()
                blobs_list.save()
                reset_blobs_fragmentation_parameters(blobs)
                blobs = blobs_list.blobs_in_video
                print("Blobs segmented saved")
            else:
                # Load blobs and global fragments
                print("It has been segmented")
                blobs_list = ListOfBlobs.load(old_video.blobs_path_segmented)
                video._preprocessing_folder = old_video._preprocessing_folder
                video._blobs_path_segmented = old_video._blobs_path_segmented
                video._has_been_segmented = True
                video._maximum_number_of_blobs = old_video.maximum_number_of_blobs
                blobs = blobs_list.blobs_in_video
                reset_blobs_fragmentation_parameters(blobs)

            # compute a model of the area of the animals (considering frames in which
            # all the animals are visible)
            model_area, maximum_body_length = compute_model_area_and_body_length(blobs, video.number_of_animals)
            if video.animal_type == 'fish':
                portrait_size = int(maximum_body_length/2)
                portrait_size =  portrait_size + portrait_size%2 #this is to make the portrait_size even
                video.portrait_size = (portrait_size, portrait_size, 1)
            elif video.animal_type == 'fly':
                portrait_size = int(np.sqrt(maximum_body_length ** 2 / 2))
                portrait_size = portrait_size + portrait_size%2  #this is to make the portrait_size
                video.portrait_size = (portrait_size, portrait_size, 1)
            #discard blobs that do not respect such model
            apply_model_area_to_video(video, blobs, model_area, video.portrait_size[0])
            #connect blobs that overlap in consecutive frames
            connect_blob_list(blobs)
            #assign an identifier to each blobl belonging to an individual fragment
            compute_fragment_identifier_and_blob_index(blobs, video.maximum_number_of_blobs)
            #save connected blobs in video (organized frame-wise) and list of global fragments
            video._has_been_preprocessed = True
            saved = False

            video.save()
            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
            blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            blobs = blobs_list.blobs_in_video
            #compute the global fragments (all animals are visible + each animals overlaps
            #with a single blob in the consecutive frame + the blobs respect the area model)
            global_fragments = give_me_list_of_global_fragments(blobs, video.number_of_animals)
            global_fragments = filter_global_fragments_by_minimum_number_of_frames(global_fragments, minimum_number_of_frames = 3)
            np.save(video.global_fragments_path, global_fragments)
            print("Blobs saved")
            #take a look to the resulting fragmentation
            #fragmentation_inspector(video, blobs)
        else:
            # Update folders and paths from previous video_object
            cv2.namedWindow('Bars')
            video._preprocessing_folder = old_video._preprocessing_folder
            video._blobs_path = old_video.blobs_path
            video._global_fragments_path = old_video.global_fragments_path
            video._maximum_number_of_blobs = old_video.maximum_number_of_blobs
            video.portrait_size = old_video.portrait_size
            # Set preprocessed flag to True
            video._has_been_preprocessed = True
            video.save()
            # Load blobs and global fragments
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video
            global_fragments = np.load(video.global_fragments_path)
            # fragmentation_inspector(video, blobs)
        #destroy windows to prevent openCV errors
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        #############################################################
        ##################   Knowledge transfer  ####################
        ####   Take the weights from a different model already   ####
        ####   trained. Works better when transfering to similar ####
        ####   conditions (light, animal type, age, ...)         ####
        #############################################################
        # knowledge_transfer_flag = getInput('Knowledge transfer','Do you want to perform knowledge transfer from another model? [y]/n')
        knowledge_transfer_flag = 'n'
        if knowledge_transfer_flag.lower() == 'y' or knowledge_transfer_flag == '':
            video.knowledge_transfer_model_folder = selectDir('', text = "Select a session folder to perform knowledge transfer from the last accumulation point") #select path to video
            video.tracking_with_knowledge_transfer = True
        elif knowledge_transfer_flag.lower() == 'n':
            pass
        else:
            raise ValueError("Invalid value, type either 'y' or 'n'")
        #############################################################
        ##################      Pre-trainer      ####################
        #### create the folder training in which all the         ####
        #### CNN-related process will be stored. The structure   ####
        #### is /training/session_num, where "num" is an natural ####
        #### number used to count how many time a video has been ####
        #### processed                                           ####
        #############################################################
        print("\n**** Pretraining ****\n")
        if not loadPreviousDict['pretraining']:
            # pretrain_flag = getInput('Pretraining','Do you want to perform pretraining? [y]/n')
            pretrain_flag = 'y'
            if pretrain_flag == 'y' or pretrain_flag == '':
                #set pretraining parameters
                #number_of_global_fragments = getInput('Pretraining','Choose the number of global fragments that will be used to pretrain the network. Default 10')
                number_of_global_fragments = 100
                #Reset used_for_training and acceptable_for_training flags
                if old_video and old_video._accumulation_finished == True:
                    for global_fragment in global_fragments:
                        global_fragment.reset_accumulation_params()
                try:
                    number_of_global_fragments = int(number_of_global_fragments)
                    pretraining_global_fragments = order_global_fragments_by_distance_travelled(give_me_pre_training_global_fragments(global_fragments, number_of_pretraining_global_fragments = number_of_global_fragments))
                except:
                    number_of_global_fragments = len(global_fragments)
                    pretraining_global_fragments = order_global_fragments_by_distance_travelled(global_fragments)

                print("pretraining with %i" %number_of_global_fragments, ' global fragments\n')
                #create folder to store pretraining
                video.create_pretraining_folder(number_of_global_fragments)
                #pretraining network parameters
                pretrain_network_params = NetworkParams(video.number_of_animals,
                                                        learning_rate = 0.01,
                                                        keep_prob = 1.0,
                                                        save_folder = video._pretraining_folder,
                                                        image_size = video.portrait_size)

                if video.tracking_with_knowledge_transfer:
                    print("Performing knowledge transfer from %s" %video.knowledge_transfer_model_folder)
                    pretrain_network_params.restore_folder = video.knowledge_transfer_model_folder

                #start pretraining
                net = pre_train(video, blobs,
                                pretraining_global_fragments,
                                pretrain_network_params,
                                store_accuracy_and_error = False,
                                check_for_loss_plateau = True,
                                save_summaries = True,
                                print_flag = False,
                                plot_flag = False)
                #save changes
                video._has_been_pretrained = True
                video.save()

        else:
            # Update folders and paths from previous video_object
            video._pretraining_folder = old_video._pretraining_folder
            pretrain_network_params = NetworkParams(video.number_of_animals,
                                                    learning_rate = 0.01,
                                                    keep_prob = 1.0,
                                                    use_adam_optimiser = False,
                                                    scopes_layers_to_optimize = None,
                                                    restore_folder = video._pretraining_folder,
                                                    save_folder = video._pretraining_folder,
                                                    image_size = video.portrait_size)
            net = ConvNetwork(pretrain_network_params)
            net.restore()
            # Set preprocessed flag to True
            video._has_been_pretrained = True
            video.save()
        #############################################################
        ###################    Accumulation   #######################
        #### take references in 'good' global fragments          ####
        #############################################################
        print("\n**** Accumulation ****")
        if not loadPreviousDict['accumulation']:
            #create folder to store accumulation models
            video.create_accumulation_folder()
            #Reset used_for_training and acceptable_for_training flags if the old video already had the accumulation done
            if old_video and old_video._accumulation_finished == True:
                print("Cleaning previous accumulation")
                for global_fragment in global_fragments:
                    global_fragment.reset_accumulation_params()
            #set network params for the accumulation model
            accumulation_network_params = NetworkParams(video.number_of_animals,
                                        learning_rate = 0.005,
                                        keep_prob = 1.0,
                                        scopes_layers_to_optimize = ['fully-connected1','softmax1'],
                                        save_folder = video._accumulation_folder,
                                        image_size = video.portrait_size)
            if video._has_been_pretrained:
                print("We will restore the network from a previous pretraining: %s\n" %video._pretraining_folder)
                accumulation_network_params.restore_folder = video._pretraining_folder
            elif not video._has_been_pretrained:
                if video.tracking_with_knowledge_transfer:
                    print("We will restore the network from a previous model (knowledge transfer): %s\n" %video.knowledge_transfer_model_folder)
                    accumulation_network_params.restore_folder = video.knowledge_transfer_model_folder
                else:
                    print("The network will be trained from scracth during accumulation\n")
                    accumulation_network_params.scopes_layers_to_optimize = None

            #instantiate network object
            net = ConvNetwork(accumulation_network_params)
            #restore variables from the pretraining
            net.restore()
            net.reinitialize_softmax_and_fully_connected()
            #instantiate accumulation manager
            accumulation_manager = AccumulationManager(global_fragments, video.number_of_animals)
            #set global epoch counter to 0
            global_step = 0
            while accumulation_manager.continue_accumulation:
                print("\n***new accumulation step %i" %accumulation_manager.counter)
                #get next fragments for accumulation
                accumulation_manager.get_next_global_fragments()
                #get images from the new global fragments
                #(we do not take images from individual fragments already used)
                accumulation_manager.get_new_images_and_labels()
                #get images for training
                #(we mix images already used with new images)
                images, labels = accumulation_manager.get_images_and_labels_for_training()
                print("images: ", images.shape)
                print("labels: ", labels.shape)
                #start training
                global_step, net, _ = train(video, blobs,
                                        global_fragments,
                                        net, images, labels,
                                        store_accuracy_and_error = False,
                                        check_for_loss_plateau = True,
                                        save_summaries = True,
                                        print_flag = False,
                                        plot_flag = False,
                                        global_step = global_step,
                                        first_accumulation_flag = accumulation_manager == 0)
                # update used_for_training flag to True for fragments used
                accumulation_manager.update_global_fragments_used_for_training()
                # update the set of images used for training
                accumulation_manager.update_used_images_and_labels()
                # assign identities fo the global fragments that have been used for training
                accumulation_manager.assign_identities_to_accumulated_global_fragments(blobs)
                # update the list of individual fragments that have been used for training
                accumulation_manager.update_individual_fragments_used()
                # Set accumulation params for rest of the accumulation
                #take images from global fragments not used in training (in the remainder test global fragments)
                candidates_next_global_fragments = [global_fragment for global_fragment in global_fragments if not global_fragment.used_for_training]
                print("number of candidate global fragments, ", len(candidates_next_global_fragments))
                if any([not global_fragment.used_for_training for global_fragment in global_fragments]):
                    images, _, candidate_individual_fragments_indices, indices_to_split = get_images_and_labels_from_global_fragments(candidates_next_global_fragments,accumulation_manager.individual_fragments_used)
                    print("*** individual fragments to be used: ", len(np.unique(candidate_individual_fragments_indices)))
                else:
                    print("All the global fragments have been used for accumulation")
                    break
                # get predictions for images in test global fragments
                assigner = assign(net, video, images, print_flag = True)
                accumulation_manager.split_predictions_after_network_assignment(assigner._predictions, assigner._softmax_probs, indices_to_split)
                # assign identities to the global fragments based on the predictions
                print('0****', len(accumulation_manager.individual_fragments_used), len(accumulation_manager.identities_of_individual_fragments_used))
                accumulation_manager.assign_identities_and_check_eligibility_for_training_global_fragments(candidate_individual_fragments_indices)
                accumulation_manager.update_counter()

            print("there are no more acceptable global_fragments for training\n")

            video._accumulation_finished = True
            video.save()
            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
            blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            np.save(video.global_fragments_path, global_fragments)
        else:
            # Update folders and paths from previous video_object
            video._accumulation_folder = old_video._accumulation_folder
            accumulation_network_params = NetworkParams(video.number_of_animals,
                                                        learning_rate = 0.005,
                                                        keep_prob = 1.0,
                                                        use_adam_optimiser = False,
                                                        scopes_layers_to_optimize = ['fully-connected1','softmax1'],
                                                        restore_folder = video._accumulation_folder,
                                                        save_folder = video._accumulation_folder,
                                                        image_size = video.portrait_size)
            net = ConvNetwork(pretrain_network_params)
            net.restore()
            # Set preprocessed flag to True
            video._accumulation_finished = True
            video.save()
        #############################################################
        ###################     Assigner      ######################
        ####
        #############################################################
        print("\n**** Assignment ****")
        if not loadPreviousDict['assignment']:
            # Get images from the blob collection
            images = get_images_from_blobs_in_video(blobs)#, video._episodes_start_end)
            print("images shape before entering to assign, ", images.shape)
            # get predictions
            assigner = assign(net, video, images, print_flag = True)
            print("number of predictions, ", len(assigner._predictions))
            print("predictions range", np.unique(assigner._predictions))
            # assign identities to each blob in each frame
            assign_identity_to_blobs_in_video(blobs, assigner)
            # compute P1 vector for individual fragmets
            compute_P1_for_blobs_in_video(video, blobs)
            # assign identities based on individual fragments
            assign_identity_to_blobs_in_video_by_fragment(video, blobs)
            # finish and save
            video._has_been_assigned = True
            # # get identity for individual fragments' extremes
            # for blobs_in_frame in blobs:
            #     for blob in blobs_in_frame:
            #         #if a blob has not been assigned but it is a fish and overlaps with one fragment
            #         #assign it!
            #         if blob.identity == 0 and blob.is_a_fish:
            #             if len(blob.next) == 1: blob.identity = blob.next[0].identity
            #             elif len(blob.previous) == 1: blob.identity = blob.previous[0].identity

            # visualise proposed tracking

            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
            blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            video.save()
            frame_by_frame_identity_inspector(video, blobs)
        else:
            # Set preprocessed flag to True
            video._has_been_assigned = True
            video.save()
            # Load blobs and global fragments
            list_of_blobs = ListOfBlobs.load(video.blobs_path)
            blobs = list_of_blobs.blobs_in_video
            global_fragments = np.load(video.global_fragments_path)
            # visualise proposed tracking
            frame_by_frame_identity_inspector(video, blobs)

    elif reUseAll == '' or reUseAll.lower() == 'y' :
        video = old_video
        list_of_blobs = ListOfBlobs.load(video.blobs_path)
        blobs = list_of_blobs.blobs_in_video
        global_fragments = np.load(video.global_fragments_path)
        frame_by_frame_identity_inspector(video, blobs)
    else:
        raise ValueError('The input introduced does not match the possible options')
