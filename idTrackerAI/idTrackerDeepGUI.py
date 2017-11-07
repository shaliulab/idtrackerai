from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys

import glob
import numpy as np
import cPickle as pickle
from tqdm import tqdm
import time
# Import third party libraries
import cv2
import psutil
import logging.config
import yaml
# Import application/library specifics
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./postprocessing')
sys.path.append('./network')
sys.path.append('./network/crossings_detector_model')
sys.path.append('./network/identification_model')
sys.path.append('./groundtruth_utils')
sys.path.append('./tf_cnnvis')
sys.path.append('./plots')
# sys.path.append('IdTrackerDeep/tracker')

from video import Video
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments, create_list_of_fragments
from list_of_global_fragments import ListOfGlobalFragments, create_list_of_global_fragments
from global_fragments_statistics import compute_and_plot_fragments_statistics
from segmentation import segment
from GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview,\
                    selectPreprocParams, fragmentation_inspector,\
                    frame_by_frame_identity_inspector, selectDir,\
                    check_resolution_reduction
from py_utils import getExistentFiles
from video_utils import checkBkg
from crossing_detector import detect_crossings
from pre_trainer import pre_trainer
from accumulation_manager import AccumulationManager
from accumulator import accumulate
from network_params import NetworkParams
from trainer import train
from assigner import assigner
from visualize_embeddings import visualize_embeddings_global_fragments
from id_CNN import ConvNetwork
from correct_duplications import solve_duplications, mark_fragments_as_duplications
from correct_impossible_velocity_jumps import correct_impossible_velocity_jumps
from solve_crossing import give_me_identities_in_crossings
from get_trajectories import produce_trajectories, smooth_trajectories
from generate_light_groundtruth_blob_list import GroundTruth, GroundTruthBlob
from compute_statistics_against_groundtruth import get_statistics_against_groundtruth
from compute_velocity_model import compute_model_velocity
# from visualise_cnn import visualise

NUM_CHUNKS_BLOB_SAVING = 500 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
VEL_PERCENTILE = 99
THRESHOLD_ACCEPTABLE_ACCUMULATION = .9
RESTORE_CRITERION = 'last'
###
# seed numpy
np.random.seed(0)
###
def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG',
    path_to_save_logs = './',
    video_object = None):
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        if os.path.exists(path_to_save_logs) and video_object is not None:
            video_object.logs_folder = os.path.join(path_to_save_logs, 'log_files')
            if not os.path.isdir(video_object.logs_folder):
                os.makedirs(video_object.logs_folder)
            config['handlers']['info_file_handler']['filename'] = os.path.join(video_object.logs_folder, 'info.log')
            config['handlers']['error_file_handler']['filename'] = os.path.join(video_object.logs_folder, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger(__name__)
    logger.propagate = True
    logger.setLevel("DEBUG")
    return logger

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...
    video_path = selectFile() #select path to video
    video = Video() #instantiate object video
    video.video_path = video_path #set path
    new_name_session_folder = getInput('Session name, ', 'Input session name. Use an old session name to load and overwrite files')
    video.create_session_folder(name = new_name_session_folder)
    video.init_processes_time_attributes()
    # set log config
    logger = setup_logging(path_to_save_logs = video.session_folder, video_object = video)
    logger.info("Starting working on session %s" %new_name_session_folder)
    logger.info("Log files saved in %s" %video.logs_folder)
    #Asking user whether to reuse preprocessing steps...'
    # processes_list = ['preprocessing', 'pretraining', 'accumulation', 'assignment', 'solving_duplications', 'crossings', 'trajectories']
    processes_list = ['preprocessing',
                    'use_previous_knowledge_transfer_decision',
                    'first_accumulation',
                    'pretraining',
                    'second_accumulation',
                    'assignment',
                    'solving_duplications',
                    'crossings',
                    'trajectories']
    #get existent files and paths to load them
    existentFiles, old_video = getExistentFiles(video, processes_list)
    #selecting files to load from previous session...'
    loadPreviousDict = selectOptions(processes_list, existentFiles,
                    text='Steps already processed in this video \n (loaded from ' + video.video_folder + ')')
    #use previous values and parameters (bkg, roi, preprocessing parameters)?
    logger.debug("Video session folder: %s " %video.session_folder)
    video.save()
    #############################################################
    ##################   Knowledge transfer  ####################
    ####   Take the weights from a different model already   ####
    ####   trained. Works better when transfering to similar ####
    ####   conditions (light, animal type, age, ...)         ####
    #############################################################
    if not bool(loadPreviousDict['use_previous_knowledge_transfer_decision']):
        knowledge_transfer_flag = getInput('Knowledge transfer','Do you want to perform knowledge transfer from another model? [y]/n')
        # knowledge_transfer_flag = 'n'
        if knowledge_transfer_flag.lower() == 'y' or knowledge_transfer_flag == '':
            video.knowledge_transfer_model_folder = selectDir('', text = "Select a session folder to perform knowledge transfer from the last accumulation point") #select path to video
            video.tracking_with_knowledge_transfer = True
        elif knowledge_transfer_flag.lower() == 'n':
            video.tracking_with_knowledge_transfer = False
        else:
            raise ValueError("Invalid value, type either 'y' or 'n'")
    else:
        video.copy_attributes_between_two_video_objects(old_video, ['knowledge_transfer_model_folder','tracking_with_knowledge_transfer'])
        video.use_previous_knowledge_transfer_decision = True
    #############################################################
    ####################  Preprocessing   #######################
    #### 1. detect blobs in the video                        ####
    #### 2. create a list of potential global fragments      ####
    #### in which all animals are visible.                   ####
    #### 3. compute a model of the area of the animals       ####
    #### (mean and variance)                                 ####
    #### 4. identify global fragments                        ####
    #### 5. create a list of objects GlobalFragment()        ####
    #############################################################
    #Selection/loading preprocessing parameters
    print('\nPreprocessing ---------------------------------------------------------')
    usePreviousPrecParams = bool(loadPreviousDict['preprocessing'])
    restore_segmentation = selectPreprocParams(video, old_video, usePreviousPrecParams)
    video.save()
    preprocessing_parameters_dict = {key: getattr(video, key) for key in video.__dict__ if 'apply_ROI' in key or 'subtract_bkg' in key or 'min' in key or 'max' in key}
    logger.info('The parameters used to preprocess the video are %s', preprocessing_parameters_dict)
    #destroy windows to prevent openCV errors
    #Loading logo during preprocessing
    img = cv2.imread('./utils/loadingIdDeep.png')
    cv2.imshow('Bars',img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    video.preprocessing_time = time.time()
    if not loadPreviousDict['preprocessing']:
        logger.info("Starting preprocessing")
        cv2.namedWindow('Bars')
        video.create_preprocessing_folder()
        if not old_video or not old_video.has_been_segmented or restore_segmentation == 'n':
            logger.debug("Starting segmentation")
            blobs = segment(video)
            logger.debug("Segmentation finished")
            list_of_blobs = ListOfBlobs(video, blobs_in_video = blobs)
            list_of_blobs.save(video.blobs_path_segmented, number_of_chunks = video.number_of_frames)
            logger.debug("Segmented blobs saved")
            video._has_been_segmented = True
        else:
            # Load blobs and global fragments
            logger.debug("Loading previously segmented blobs")
            preprocessing_parameters_dict = {key: getattr(video, key)
                                            for key in video.__dict__
                                            if 'apply_ROI' in key
                                            or 'subtract_bkg' in key
                                            or 'min_' in key
                                            or 'max_' in key}
            logger.debug('The parameters used to preprocess the video are %s', preprocessing_parameters_dict)
            list_of_blobs = ListOfBlobs.load(old_video.blobs_path_segmented)
            video._has_been_segmented = True
            logger.debug("Segmented blobs loaded")
        video.save()
        logger.info("Computing maximum number of blobs detected in the video")
        list_of_blobs.check_maximal_number_of_blob()
        logger.info("Computing a model of the area of the individuals")
        video._model_area, video._median_body_length = list_of_blobs.compute_model_area_and_body_length()
        video.compute_identification_image_size(video.median_body_length)
        list_of_blobs.video = video
        if not list_of_blobs.blobs_are_connected:
            list_of_blobs.compute_overlapping_between_subsequent_frames()
        detect_crossings(list_of_blobs, video, video.model_area, use_network = False)
        #connect blobs that overlap in consecutive frames
        list_of_blobs.compute_overlapping_between_subsequent_frames()
        #assign an identifier to each blob belonging to an individual fragment
        list_of_blobs.compute_fragment_identifier_and_blob_index()
        #assign an identifier to each blob belonging to a crossing fragment
        list_of_blobs.compute_crossing_fragment_identifier()
        #create list of fragments
        fragments = create_list_of_fragments(list_of_blobs.blobs_in_video,
                                            video.number_of_animals)
        list_of_fragments = ListOfFragments(video, fragments)
        video._fragment_identifier_to_index = list_of_fragments.get_fragment_identifier_to_index_list()
        #compute the global fragments (all animals are visible + each animals overlaps
        #with a single blob in the consecutive frame + the blobs respect the area model)
        global_fragments = create_list_of_global_fragments(list_of_blobs.blobs_in_video,
                                                            list_of_fragments.fragments,
                                                            video.number_of_animals)
        list_of_global_fragments = ListOfGlobalFragments(video, global_fragments)
        video.individual_fragments_lenghts, \
        video.individual_fragments_distance_travelled = compute_and_plot_fragments_statistics(video,
                                                                                            video.model_area,
                                                                                            list_of_blobs,
                                                                                            list_of_fragments,
                                                                                            list_of_global_fragments)
        video.number_of_global_fragments = list_of_global_fragments.number_of_global_fragments
        list_of_global_fragments.filter_candidates_global_fragments_for_accumulation()
        video.number_of_global_fragments_candidates_for_accumulation = list_of_global_fragments.number_of_global_fragments
        list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
        video._number_of_unique_images_in_global_fragments = list_of_fragments.compute_total_number_of_images_in_global_fragments()
        list_of_global_fragments.compute_maximum_number_of_images()
        video._maximum_number_of_images_in_global_fragments = list_of_global_fragments.maximum_number_of_images
        list_of_fragments.get_accumulable_individual_fragments_identifiers(list_of_global_fragments)
        list_of_fragments.get_not_accumulable_individual_fragments_identifiers(list_of_global_fragments)
        list_of_fragments.set_fragments_as_accumulable_or_not_accumulable()
        #save connected blobs in video (organized frame-wise)
        list_of_blobs.video = video
        list_of_blobs.save(number_of_chunks = video.number_of_frames)
        list_of_fragments.save()
        list_of_global_fragments.save(list_of_fragments.fragments)
        video._has_been_preprocessed = True
        video.save()
        logger.info("Blobs detection and fragmentation finished succesfully.")
    else:
        cv2.namedWindow('Bars')
        logger.info("Loading preprocessed video")
        path_attributes = ['preprocessing_folder', 'blobs_path', 'global_fragments_path', 'fragments_path']
        video.copy_attributes_between_two_video_objects(old_video, path_attributes)
        video._has_been_segmented = True
        video._has_been_preprocessed = True
        video.save()
        # Load blobs and global fragments
        logger.info("Loading blob objects")
        list_of_blobs = ListOfBlobs.load(video.blobs_path)
        list_of_blobs.video = video
        logger.info("Loading list of fragments")
        list_of_fragments = ListOfFragments.load(video.fragments_path)
        logger.info("Loading list of global fragments")
        list_of_global_fragments = ListOfGlobalFragments.load(video.global_fragments_path, list_of_fragments.fragments)
    video.preprocessing_time = time.time() - video.preprocessing_time
    #take a look to the resulting fragmentation
    # fragmentation_inspector(video, list_of_blobs.blobs_in_video)
    #destroy windows to prevent openCV errors
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    #############################################################
    ##################   Protocols cascade   ####################
    #############################################################
    #### Accumulation ####
    print('\nAccumulation 0 ---------------------------------------------------------')
    video.first_accumulation_time = time.time()
    video.accumulation_trial = 0
    video.create_accumulation_folder(iteration_number = 0)
    logger.info("Set accumulation network parameters")
    accumulation_network_params = NetworkParams(video.number_of_animals,
                                learning_rate = 0.005,
                                keep_prob = 1.0,
                                scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax'],
                                restore_folder = video.accumulation_folder,
                                save_folder = video.accumulation_folder,
                                image_size = video.identification_image_size)
    if not bool(loadPreviousDict['first_accumulation']):
        logger.info("Starting accumulation")
        list_of_fragments.reset(roll_back_to = 'fragmentation')
        list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        if video.tracking_with_knowledge_transfer:
            logger.info("We will restore the network from a previous model (knowledge transfer): %s" %video.knowledge_transfer_model_folder)
            accumulation_network_params.restore_folder = video.knowledge_transfer_model_folder
        else:
            logger.info("The network will be trained from scratch during accumulation")
            accumulation_network_params.scopes_layers_to_optimize = None
        #instantiate network object
        logger.info("Initialising accumulation network")
        net = ConvNetwork(accumulation_network_params)
        #if knowledge transfer is performed on the same animals we don't reinitialise the classification part of the net
        video._knowledge_transfer_from_same_animals = False
        if video.tracking_with_knowledge_transfer:
            net.restore()
            same_animals = getInput("Same animals", "Are you tracking the same animals? y/N")
            if same_animals.lower() == 'n' or same_animals == '':
                net.reinitialize_softmax_and_fully_connected()
            else:
                video._knowledge_transfer_from_same_animals = True
        #instantiate accumulation manager
        logger.info("Initialising accumulation manager")
        # the list of global fragments is ordered in place from the distance (in frames) wrt
        # the core of the first global fragment that will be accumulated
        video._first_frame_first_global_fragment.append(list_of_global_fragments.set_first_global_fragment_for_accumulation(accumulation_trial = 0))
        list_of_global_fragments.video = video
        list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(accumulation_trial = 0)
        accumulation_manager = AccumulationManager(video, list_of_fragments,
                                                    list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION,
                                                    restore_criterion = RESTORE_CRITERION)
        #set global epoch counter to 0
        logger.info("Start accumulation")
        global_step = 0
        video._ratio_accumulated_images = accumulate(accumulation_manager,
                                            video,
                                            global_step,
                                            net,
                                            video.knowledge_transfer_from_same_animals)
        logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
        video._first_accumulation_finished = True
        ### NOTE: save all the accumulation statistics
        video.save()
        logger.info("Saving fragments")
        list_of_fragments.save()
        list_of_global_fragments.save(list_of_fragments.fragments)
    else:
        ### NOTE: load all the accumulation statistics
        logger.info("Restoring accumulation network")
        list_of_attributes = ['accumulation_folder',
                    'second_accumulation_finished',
                    'number_of_accumulated_global_fragments',
                    'number_of_non_certain_global_fragments',
                    'number_of_randomly_assigned_global_fragments',
                    'number_of_nonconsistent_global_fragments',
                    'number_of_nonunique_global_fragments',
                    'number_of_acceptable_global_fragments',
                    'validation_accuracy', 'validation_individual_accuracies',
                    'training_accuracy', 'training_individual_accuracies',
                    'ratio_of_accumulated_images', 'accumulation_trial',
                    'ratio_accumulated_images', 'first_accumulation_finished',
                    'knowledge_transfer_from_same_animals', 'accumulation_statistics',
                    'first_frame_first_global_fragment']
        is_property = [True, True, False, False,
                        False, False, False, False,
                        False, False, False, False,
                        False, False, True, True,
                        True, False, True]
        video.copy_attributes_between_two_video_objects(old_video, list_of_attributes, is_property = is_property)
        accumulation_network_params.restore_folder = video._accumulation_folder
        net = ConvNetwork(accumulation_network_params)
        net.restore()
        logger.info("Saving video")
        video.save()
    video.first_accumulation_time = time.time() - video.first_accumulation_time
    list_of_fragments.save_light_list(video._accumulation_folder)
    if video.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
        if isinstance(video.first_frame_first_global_fragment, list):
            video._first_frame_first_global_fragment = video.first_frame_first_global_fragment[video.accumulation_trial]
            list_of_fragments.video = video
            list_of_global_fragments.video = video
        video.assignment_time = time.time()
        if not loadPreviousDict['assignment']:
            #### Assigner ####
            print('\nAssignment ---------------------------------------------------------')
            assigner(list_of_fragments, video, net)
            video._has_been_assigned = True
            ### NOTE: save all the assigner statistics
        else:
            ### NOTE: load all the assigner statistics
            video._has_been_assigned = True
        video.assignment_time = time.time() - video.assignment_time
        video.save()
    else:
        print('\nPretraining ---------------------------------------------------------')
        video.pretraining_time = time.time()
        #create folder to store pretraining
        video.create_pretraining_folder()
        #pretraining if first accumulation trial does not cover 90% of the images in global fragments
        pretrain_network_params = NetworkParams(video.number_of_animals,
                                                learning_rate = 0.01,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = None,
                                                save_folder = video.pretraining_folder,
                                                image_size = video.identification_image_size)
        if not loadPreviousDict['pretraining']:
            #### Pre-trainer ####
            list_of_fragments.reset(roll_back_to = 'fragmentation')
            list_of_global_fragments.order_by_distance_travelled()
            pre_trainer(old_video, video, list_of_fragments, list_of_global_fragments, pretrain_network_params)
            logger.info("Pretraining ended")
            #save changes
            logger.info("Saving changes in video object")
            video._has_been_pretrained = True
            video.save()
            ### NOTE: save pre-training statistics
        else:
            logger.info("Restoring pretrained network")
            video.copy_attributes_between_two_video_objects(old_video, ['pretraining_folder', 'has_been_pretrained'])
            pretrain_network_params.restore_folder = video.pretraining_folder
            net = ConvNetwork(pretrain_network_params)
            net.restore()
            # Set preprocessed flag to True
            video.save()
            ### NOTE: load pre-training statistics
        video.pretraining_time = time.time() - video.pretraining_time
        #### Accumulation ####
        #Last accumulation after pretraining
        video.second_accumulation_time = time.time()
        if not loadPreviousDict['second_accumulation']:
            percentage_of_accumulated_images = []
            for i in range(1,4):
                print('\nAccumulation %i ---------------------------------------------------------' %i)
                logger.info("Starting accumulation")
                #create folder to store accumulation models
                video.create_accumulation_folder(iteration_number = i)
                video.accumulation_trial = i
                #Reset used_for_training and acceptable_for_training flags if the old video already had the accumulation done
                list_of_fragments.reset(roll_back_to = 'fragmentation')
                list_of_global_fragments.reset(roll_back_to = 'fragmentation')
                logger.info("We will restore the network from a previous pretraining: %s" %video.pretraining_folder)
                accumulation_network_params.save_folder = video.accumulation_folder
                accumulation_network_params.restore_folder = video.pretraining_folder
                accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
                logger.info("Initialising accumulation network")
                net = ConvNetwork(accumulation_network_params)
                #restore variables from the pretraining
                net.restore()
                net.reinitialize_softmax_and_fully_connected()
                #instantiate accumulation manager
                logger.info("Initialising accumulation manager")
                video._first_frame_first_global_fragment.append(list_of_global_fragments.set_first_global_fragment_for_accumulation(accumulation_trial = i - 1))
                list_of_global_fragments.video = video
                list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(accumulation_trial = i - 1)
                accumulation_manager = AccumulationManager(video,
                                                            list_of_fragments, list_of_global_fragments,
                                                            threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION,
                                                            restore_criterion = RESTORE_CRITERION)
                #set global epoch counter to 0
                logger.info("Start accumulation")
                global_step = 0
                video._ratio_accumulated_images = accumulate(accumulation_manager,
                                                            video,
                                                            global_step,
                                                            net,
                                                            video.knowledge_transfer_from_same_animals)
                logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
                if video.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                    break
                else:
                    percentage_of_accumulated_images.append(video.ratio_accumulated_images)
                    logger.info("This accumulation was not satisfactory. Try to start from a different global fragment")
                    list_of_fragments.save_light_list(video._accumulation_folder)


            if len(percentage_of_accumulated_images) > 1 and np.argmax(percentage_of_accumulated_images) != 2:
                video.accumulation_trial = np.argmax(percentage_of_accumulated_images) + 1
                video._first_frame_first_global_fragment = video.first_frame_first_global_fragment[video.accumulation_trial]
                list_of_fragments.video = video
                list_of_global_fragments.video = video
                accumulation_folder_name = 'accumulation_' + str(video.accumulation_trial)
                video._accumulation_folder = os.path.join(video.session_folder, accumulation_folder_name)
                list_of_fragments.load_light_list(video._accumulation_folder)
            video._second_accumulation_finished = True
            logger.info("Saving global fragments")
            list_of_fragments.save()
            list_of_global_fragments.save(list_of_fragments.fragments)
            ### NOTE: save second_accumulation statistics
            video.save()
        else:
            list_of_attributes = ['accumulation_folder',
                        'second_accumulation_finished',
                        'number_of_accumulated_global_fragments',
                        'number_of_non_certain_global_fragments',
                        'number_of_randomly_assigned_global_fragments',
                        'number_of_nonconsistent_global_fragments',
                        'number_of_nonunique_global_fragments',
                        'number_of_acceptable_global_fragments',
                        'validation_accuracy', 'validation_individual_accuracies',
                        'training_accuracy', 'training_individual_accuracies',
                        'ratio_of_accumulated_images', 'accumulation_trial',
                        'ratio_accumulated_images', 'first_accumulation_finished',
                        'knowledge_transfer_from_same_animals', 'accumulation_statistics',
                        'first_frame_first_global_fragment']
            is_property = [True, True, False, False,
                            False, False, False, False,
                            False, False, False, False,
                            False, False, True, True,
                            True, False, True]
            video.copy_attributes_between_two_video_objects(old_video, list_of_attributes)
            logger.info("Restoring trained network")
            accumulation_network_params.restore_folder = video._accumulation_folder
            list_of_fragments.load_light_list(video._accumulation_folder)
            net = ConvNetwork(accumulation_network_params)
            net.restore()
            # Set preprocessed flag to True
            video.save()
            ### NOTE: load pre-training statistics
        video.second_accumulation_time = time.time() - video.second_accumulation_time
        video.assignment_time = time.time()
        if not loadPreviousDict['assignment']:
            #### Assigner ####
            print('\n---------------------------------------------------------')
            assigner(list_of_fragments, video, net)
            video._has_been_assigned = True
            ### NOTE: save all the assigner statistics
        else:
            ### NOTE: load all the assigner statistics
            video._has_been_assigned = True
        video.assignment_time = time.time() - video.assignment_time
        video.save()

    # finish and save
    logger.debug("Saving list of fragments, list of global fragments and video object")
    list_of_fragments.save()
    list_of_global_fragments.save(list_of_fragments.fragments)
    video.save()
    #############################################################
    ################### CNN-visualisation  ######################
    ####
    #############################################################
    # accumulated_global_fragments = [global_fragment for global_fragment in global_fragments
    #                                 if global_fragment.used_for_training]
    # for i in range(10):
    #     image = accumulated_global_fragments[0].portraits[0][i]
    #     image = np.expand_dims(image, 2)
    #     image = np.expand_dims(image, 0)
    #     label = accumulated_global_fragments[0]._temporary_ids[0]
    #
    #     visualise(video, net, image, label)
    #############################################################
    ###################   Solve duplications      ###############
    ####
    #############################################################
    video.solve_duplications_time = time.time()
    if not loadPreviousDict['solving_duplications']:
        logger.info("Start checking for and solving duplications")
        list_of_fragments.reset(roll_back_to = 'assignment')
        # mark fragments as duplications
        mark_fragments_as_duplications(list_of_fragments.fragments)
        # solve duplications
        solve_duplications(list_of_fragments)
        video._has_duplications_solved = True
        logger.info("Done")
        # finish and save
        logger.info("Saving")
        # list_of_blobs = ListOfBlobs(video, blobs_in_video = blobs)
        # list_of_blobs.save(video.blobs_path, NUM_CHUNKS_BLOB_SAVING)
        list_of_fragments.save()
        video.save()
        logger.info("Done")
        # visualise proposed tracking
        # frame_by_frame_identity_inspector(video, blobs)
    else:
        # Set duplications flag to True
        logger.info("Duplications have already been checked. Using previous information")
        video._has_duplications_solved = True
        video.save()
    video.solve_duplications_time = time.time() - video.solve_duplications_time
    #############################################################
    ###################  Solving impossible jumps    ############
    list_of_fragments.video = video
    video.solve_impossible_jumps_time = time.time()
    print("\n**** Correct impossible velocity jump ****")
    logging.info("Solving impossible velocity jumps")
    if hasattr(old_video,'velocity_threshold') and not hasattr(video,'velocity_threshold'):
        video.velocity_threshold = old_video.velocity_threshold
    elif not hasattr(old_video, 'velocity_threshold') and not hasattr(video,'velocity_threshold'):
        video.velocity_threshold = compute_model_velocity(list_of_fragments.fragments, video.number_of_animals, percentile = VEL_PERCENTILE)
    # fix_identity_of_blobs_in_video(blobs)
    correct_impossible_velocity_jumps(list_of_fragments)
    logger.info("Done")
    # finish and save
    logger.info("Saving")
    list_of_fragments.save()
    # video.compute_overall_P2(blobs)
    video.save()
    logger.info("Done")
    video.solve_impossible_jumps_time = time.time() - video.solve_impossible_jumps_time
    #############################################################
    ##############   Solve crossigns   ##########################
    ####
    #############################################################
    # print("\n**** Assign crossings ****")
    # if not loadPreviousDict['crossings']:
    #     video._has_crossings_solved = False
    #     pass

    #############################################################
    ##############   Invididual fragments stats #################
    ####
    #############################################################
    video.individual_fragments_stats = list_of_fragments.get_stats(list_of_global_fragments)
    video.compute_overall_P2(list_of_fragments.fragments)
    print("individual overall_P2 ", video.individual_P2)
    print("overall_P2 ", video.overall_P2)
    list_of_fragments.plot_stats()
    list_of_fragments.save_light_list(video._accumulation_folder)
    video.save()

    #############################################################
    ##############   Update list of blobs   #####################
    ####
    #############################################################
    list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments)
    if False:
        list_of_blobs.compute_nose_and_head_coordinates()
    list_of_blobs.save(number_of_chunks = video.number_of_frames)

    #############################################################
    ##############   Create trajectories    #####################
    ####
    #############################################################
    video.generate_trajectories_time = time.time()
    if not loadPreviousDict['trajectories']:
        video.create_trajectories_folder()
        logger.info("Generating trajectories. The trajectories files are stored in %s" %video.trajectories_folder)
        trajectories = produce_trajectories(list_of_blobs.blobs_in_video, video.number_of_frames, video.number_of_animals)
        logger.info("Saving trajectories")
        for name in trajectories:
            np.save(os.path.join(video.trajectories_folder, name + '_trajectories.npy'), trajectories[name])
            np.save(os.path.join(video.trajectories_folder, name + '_smooth_trajectories.npy'), smooth_trajectories(trajectories[name]))
            np.save(os.path.join(video.trajectories_folder, name + '_smooth_velocities.npy'), smooth_trajectories(trajectories[name], derivative = 1))
            np.save(os.path.join(video.trajectories_folder,name + '_smooth_accelerations.npy'), smooth_trajectories(trajectories[name], derivative = 2))
        video._has_trajectories = True
        video.save()
        logger.info("Done")
    else:
        video._has_trajectories = True
        video.save()
    video.generate_trajectories_time = time.time() - video.generate_trajectories_time
    #############################################################
    ##############   Compute groundtruth    #####################
    ####
    #############################################################
    groundtruth_path = os.path.join(video.video_folder,'_groundtruth.npy')
    if os.path.isfile(groundtruth_path):
        print("\n**** Computing accuracy wrt. groundtruth ****")
        groundtruth = np.load(groundtruth_path).item()
        groundtruth.list_of_blobs = groundtruth.list_of_blobs[groundtruth.start:groundtruth.end]
        blobs_to_compare_with_groundtruth = list_of_blobs.blobs_in_video[groundtruth.start:groundtruth.end]

        video.gt_accuracy, video.gt_individual_accuracy, video.gt_accuracy_assigned, video.gt_individual_accuracy_assigned = get_statistics_against_groundtruth(groundtruth, blobs_to_compare_with_groundtruth)
        video.gt_start_end = (groundtruth.start, groundtruth.end)
        video.save()
    video.total_time = sum([video.generate_trajectories_time,
                            video.solve_impossible_jumps_time,
                            video.solve_duplications_time,
                            video.assignment_time,
                            video.second_accumulation_time,
                            video.pretraining_time,
                            video.assignment_time,
                            video.first_accumulation_time,
                            video.preprocessing_time])
    video.save()
