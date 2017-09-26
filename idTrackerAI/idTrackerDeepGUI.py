from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys
sys.setrecursionlimit(100000)
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
# sys.path.append('IdTrackerDeep/tracker')

from video import Video
from blob import compute_fragment_identifier_and_blob_index,\
                compute_crossing_fragment_identifier,\
                connect_blob_list,\
                apply_model_area_to_video,\
                ListOfBlobs,\
                get_images_from_blobs_in_video,\
                reset_blobs_fragmentation_parameters,\
                compute_portrait_size,\
                check_number_of_blobs
from fragment import create_list_of_fragments
from globalfragment import compute_model_area_and_body_length,\
                            give_me_list_of_global_fragments,\
                            ModelArea,\
                            give_me_pre_training_global_fragments,\
                            get_images_and_labels_from_global_fragments,\
                            subsample_images_for_last_training,\
                            order_global_fragments_by_distance_travelled,\
                            filter_global_fragments_by_minimum_number_of_frames,\
                            compute_and_plot_global_fragments_statistics,\
                            check_uniquenss_of_global_fragments,\
                            get_number_of_images_in_global_fragments_list,\
                            give_me_number_of_unique_images_in_global_fragments
from get_portraits import get_body
from segmentation import segment
from get_crossings_data_set import CrossingDataset
from network_params_crossings import NetworkParams_crossings
from cnn_architectures import cnn_model_crossing_detector
from crossings_detector_model import ConvNetwork_crossings
from train_crossings_detector import TrainDeepCrossing
from get_predictions_crossings import GetPredictionCrossigns
from GUI_utils import selectFile,\
                    getInput,\
                    selectOptions,\
                    ROISelectorPreview,\
                    selectPreprocParams,\
                    fragmentation_inspector,\
                    frame_by_frame_identity_inspector,\
                    selectDir,\
                    check_resolution_reduction
from py_utils import getExistentFiles
from video_utils import checkBkg
from pre_trainer import pre_trainer
from accumulation_manager import AccumulationManager, get_predictions_of_candidates_global_fragments
from accumulator import accumulate
from network_params import NetworkParams
from trainer import train
from assigner import assigner
from visualize_embeddings import visualize_embeddings_global_fragments
from id_CNN import ConvNetwork
from correct_duplications import solve_duplications, mark_blobs_as_duplications
from correct_impossible_velocity_jumps import fix_identity_of_blobs_in_video, correct_impossible_velocity_jumps
from solve_crossing import give_me_identities_in_crossings
from get_trajectories import produce_trajectories, smooth_trajectories
from generate_light_groundtruth_blob_list import GroundTruth, GroundTruthBlob
from compute_statistics_against_groundtruth import get_statistics_against_groundtruth
from compute_velocity_model import compute_model_velocity
# from visualise_cnn import visualise

NUM_CHUNKS_BLOB_SAVING = 500 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
PERCENTAGE_OF_GLOBAL_FRAGMENTS_PRETRAINING = .25
VEL_PERCENTILE = 99
THRESHOLD_ACCEPTABLE_ACCUMULATION = .9
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
    logger = setup_logging(path_to_save_logs = video._session_folder, video_object = video)
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
    print("existentFiles ", existentFiles)
    #selecting files to load from previous session...'
    loadPreviousDict = selectOptions(processes_list, existentFiles, text='Steps already processed in this video \n (loaded from ' + video._video_folder + ')')
    print("loadPreviousDict ", loadPreviousDict)
    #use previous values and parameters (bkg, roi, preprocessing parameters)?
    logger.debug("Video session folder: %s " %video._session_folder)
    video.save()
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
        if not old_video or not old_video._has_been_segmented or restore_segmentation == 'n':
            logger.debug("Starting segmentation")
            blobs = segment(video)
            logger.debug("Segmentation finished")
            video._has_been_segmented = True
            blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path_segmented)
            blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
            blobs_list.cut_in_chunks()
            blobs_list.save()
            reset_blobs_fragmentation_parameters(blobs)
            logger.debug("Saving segmented blobs")
            blobs = blobs_list.blobs_in_video
            logger.debug("Segmented blobs saved")
        else:
            # Load blobs and global fragments
            logger.debug("Loading previously segmented blobs")
            preprocessing_parameters_dict = {key: getattr(video, key) for key in video.__dict__ if 'apply_ROI' in key or 'subtract_bkg' in key or 'min_' in key or 'max_' in key}
            logger.debug('The parameters used to preprocess the video are %s', preprocessing_parameters_dict)
            blobs_list = ListOfBlobs.load(old_video.blobs_path_segmented)
            video.save()
            blobs = blobs_list.blobs_in_video
            logger.debug("Segmented blobs loaded. Reset blobs for fragmentation")
            logger.debug("Blobs reset")
            reset_blobs_fragmentation_parameters(blobs)
        video.save()
        logger.info("Computing maximum number of blobs detected in the video")
        frames_with_more_blobs_than_animals = check_number_of_blobs(video, blobs)
        logger.info("Computing a model of the area of the individuals")
        model_area, median_body_length = compute_model_area_and_body_length(blobs, video.number_of_animals)
        video.median_body_length = median_body_length
        # compute portrait size
        compute_portrait_size(video, median_body_length)
        logger.info("Discriminating blobs representing individuals from blobs associated to crossings")
        apply_model_area_to_video(video, blobs, model_area, video.portrait_size[0])
        # use_crossings_detector = getInput('Crossings detector', 'Do you want to you the crossings detector? y/N')
        # if use_crossings_detector == 'y':
        #     video.create_crossings_detector_folder()
        #     logger.info("Get individual and crossing images labelled data")
        #     training_set = CrossingDataset(blobs, video, scope = 'training')
        #     training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .9)
        #     validation_set = CrossingDataset(blobs, video, scope = 'validation',
        #                                                     crossings = training_set.crossings,
        #                                                     fish = training_set.fish,
        #                                                     image_size = training_set.image_size)
        #     validation_set.get_data(sampling_ratio_start = .9, sampling_ratio_end = 1.)
        #     logger.info("Start crossing detector training")
        #     logger.info("Crossing detector training finished")
        #     crossing_image_size = training_set.image_size
        #     crossing_image_shape = training_set.images.shape[1:]
        #     logger.info("crossing image shape %s" %str(crossing_image_shape))
        #     crossings_detector_network_params = NetworkParams_crossings(number_of_classes = 2,
        #                                                                 learning_rate = 0.001,
        #                                                                 architecture = cnn_model_crossing_detector,
        #                                                                 keep_prob = 1.0,
        #                                                                 save_folder = video._crossings_detector_folder,
        #                                                                 image_size = crossing_image_shape)
        #     net = ConvNetwork_crossings(crossings_detector_network_params)
        #     TrainDeepCrossing(net, training_set, validation_set, num_epochs = 95, plot_flag = True)
        #     logger.debug("crossing image size %s" %str(crossing_image_size))
        #     video.crossing_image_shape = crossing_image_shape
        #     video.crossing_image_size = crossing_image_size
        #     video.save()
        #     logger.debug("Freeing memory. Validation and training crossings sets deleted")
        #     validation_set = None
        #     training_set = None
        #     test_set = CrossingDataset(blobs, video, scope = 'test',
        #                                             image_size = video.crossing_image_size)
        #     # get predictions of individual blobs outside of global fragments
        #     logger.debug("Classify individuals and crossings")
        #     crossings_predictor = GetPredictionCrossigns(net)
        #     predictions = crossings_predictor.get_all_predictions(test_set)
        #     # set blobs as crossings by deleting the portrait
        #     [setattr(blob,'_portrait',None) if prediction == 1 else setattr(blob,'bounding_box_image', None)
        #                                     for blob, prediction in zip(test_set.test, predictions)]
        #     # delete bounding_box_image from blobs that have portraits
        #     [setattr(blob,'bounding_box_image', None) for blobs_in_frame in blobs
        #                                                 for blob in blobs_in_frame
        #                                                 if blob.is_a_fish
        #                                                 and blob.bounding_box_image is not None]
        #     logger.debug("Freeing memory. Test crossings set deleted")
        #     test_set = None
        #connect blobs that overlap in consecutive frames
        logger.debug("Generate individual and crossing fragments")
        connect_blob_list(blobs)
        #assign an identifier to each blob belonging to an individual fragment
        next_fragment_identifier = compute_fragment_identifier_and_blob_index(blobs, video.maximum_number_of_blobs)
        #assign an identifier to each blob belonging to a crossing fragment
        compute_crossing_fragment_identifier(blobs, next_fragment_identifier)
        #create list of fragments
        list_of_fragments = create_list_of_fragments(blobs, video.number_of_animals)
        #save connected blobs in video (organized frame-wise) and list of global fragments
        video._has_been_preprocessed = True
        logger.debug("Saving individual and crossing fragments")
        blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
        blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
        blobs_list.cut_in_chunks()
        blobs_list.save()
        blobs = blobs_list.blobs_in_video
        #compute the global fragments (all animals are visible + each animals overlaps
        #with a single blob in the consecutive frame + the blobs respect the area model)
        logger.info("Generate global fragments")
        global_fragments = give_me_list_of_global_fragments(blobs, list_of_fragments, video.number_of_animals)
        video.individual_fragments_lenghts, video.individual_fragments_distance_travelled = compute_and_plot_global_fragments_statistics(video, list_of_fragments, global_fragments)
        video.number_of_non_filtered_global_fragments = len(global_fragments)
        global_fragments = filter_global_fragments_by_minimum_number_of_frames(global_fragments, minimum_number_of_frames = 3)
        video.number_of_global_fragments = len(global_fragments)
        logger.info("Global fragments have been generated")
        # video.individual_fragments_lenghts, video.individual_fragments_distance_travelled = compute_and_plot_global_fragments_statistics(video, blobs, global_fragments)
        video.number_of_unique_images_in_global_fragments = give_me_number_of_unique_images_in_global_fragments(global_fragments)
        video.maximum_number_of_portraits_in_global_fragments = np.max([global_fragment.total_number_of_images for global_fragment in global_fragments])
        logger.info("Saving global fragments.")
        np.save(video.global_fragments_path, global_fragments)
        saved = False
        video.save()
        logger.info("Done.")
    else:
        cv2.namedWindow('Bars')
        logger.info("Loading preprocessed video")
        path_attributes = ['_preprocessing_folder', '_blobs_path', '_global_fragments_path']
        video.copy_attributes_between_two_video_objects(old_video, path_attributes)
        video._has_been_preprocessed = True
        video.save()
        # Load blobs and global fragments
        logger.info("Loading blob objects")
        list_of_blobs = ListOfBlobs.load(video.blobs_path)
        blobs = list_of_blobs.blobs_in_video
        logger.info("Loading global fragments")
        global_fragments = np.load(video.global_fragments_path)
        logger.info("Done")
    video.preprocessing_time = time.time() - video.preprocessing_time
    #take a look to the resulting fragmentation
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
        video.copy_attributes_between_two_video_objects(old_video, ['knowledge_transfer_model_folder','tracking_with_knowledge_transfer',])
        video.use_previous_knowledge_transfer_decision = True
    #############################################################
    ##################   Protocols cascade   ####################
    #############################################################
    #### Accumulation ####
    video.first_accumulation_time = time.time()
    video.create_accumulation_folder(iteration_number = 0)
    logger.info("Set accumulation network parameters")
    accumulation_network_params = NetworkParams(video.number_of_animals,
                                learning_rate = 0.005,
                                keep_prob = 1.0,
                                scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax'],
                                save_folder = video._accumulation_folder,
                                image_size = video.portrait_size)
    if not bool(loadPreviousDict['first_accumulation']):
        logger.info("Starting accumulation")
        reset_blobs_fragmentation_parameters(blobs, recovering_from = 'accumulation')
        #Reset used_for_training and acceptable_for_training flags if the old video already had the accumulation done
        [global_fragment.reset_accumulation_params() for global_fragment in global_fragments]
        if video.tracking_with_knowledge_transfer:
            logger.info("We will restore the network from a previous model (knowledge transfer): %s" %video.knowledge_transfer_model_folder)
            accumulation_network_params.restore_folder = video.knowledge_transfer_model_folder
        else:
            logger.info("The network will be trained from scratch during accumulation")
            accumulation_network_params.scopes_layers_to_optimize = None
        #instantiate network object
        logger.info("Initialising accumulation network")
        net = ConvNetwork(accumulation_network_params)
        #restore variables from the pretraining
        net.restore()
        #if knowledge transfer is performed on the same animals we don't reinitialise the classification part of the net
        video.knowledge_transfer_from_same_animals = False
        if video.tracking_with_knowledge_transfer:
            same_animals = getInput("Same animals", "Are you tracking the same animals? y/N")
            if same_animals.lower() == 'n' or same_animals == '':
                net.reinitialize_softmax_and_fully_connected()
            else:
                video.knowledge_transfer_from_same_animals = True
        #instantiate accumulation manager
        logger.info("Initialising accumulation manager")
        accumulation_manager = AccumulationManager(blobs, global_fragments, video.number_of_animals)
        #set global epoch counter to 0
        logger.info("Start accumulation")
        global_step = 0
        ratio_accumulated_images_over_all_unique_images_in_global_fragments = accumulate(accumulation_manager,
                                                                                            video,
                                                                                            blobs,
                                                                                            global_fragments,
                                                                                            global_step,
                                                                                            net,
                                                                                            video.knowledge_transfer_from_same_animals,
                                                                                            get_ith_global_fragment = 0)
        logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
        video.ratio_accumulated_images_over_all_unique_images_in_global_fragments = ratio_accumulated_images_over_all_unique_images_in_global_fragments
        video._first_accumulation_finished = True
        ### NOTE: save all the accumulation statistics
        video.save()
    else:
        ### NOTE: load all the accumulation statistics
        logger.info("Restoring accumulation network")
        video.copy_attributes_between_two_video_objects(old_video, ['_accumulation_folder',\
                                                                    'ratio_accumulated_images_over_all_unique_images_in_global_fragments',\
                                                                    '_first_accumulation_finished',\
                                                                    'knowledge_transfer_from_same_animals'])
        accumulation_network_params.restore_folder = video._accumulation_folder
        net = ConvNetwork(accumulation_network_params)
        net.restore()
        logger.info("Saving video")
        video.save()
    video.first_accumulation_time = time.time() - video.first_accumulation_time
    if video.ratio_accumulated_images_over_all_unique_images_in_global_fragments > THRESHOLD_ACCEPTABLE_ACCUMULATION:
        video.assignment_time = time.time()
        if not loadPreviousDict['assignment']:
            #### Assigner ####
            assigner(blobs, video, net)
            video._has_been_assigned = True
            ### NOTE: save all the assigner statistics
        else:
            ### NOTE: load all the assigner statistics
            video._has_been_assigned = True
        video.assignment_time = time.time() - video.assignment_time
    else:
        video.pretraining_time = time.time()
        #create folder to store pretraining
        video.create_pretraining_folder()
        #pretraining if first accumulation trial does not cover 90% of the images in global fragments
        pretrain_network_params = NetworkParams(video.number_of_animals,
                                                learning_rate = 0.01,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = None,
                                                save_folder = video._pretraining_folder,
                                                image_size = video.portrait_size)
        if not loadPreviousDict['pretraining']:
            #### Pre-trainer ####
            pre_trainer(old_video, video, blobs, global_fragments, pretrain_network_params)
            logger.info("Pretraining ended")
            #save changes
            logger.info("Saving changes in video object")
            video._has_been_pretrained = True
            video.save()
            ### NOTE: save pre-training statistics
        else:
            logger.info("Initialising network for accumulation")
            video.copy_attributes_between_two_video_objects(old_video, ['_pretraining_folder', '_has_been_pretrained'])
            pretrain_network_params.restore_folder = video._pretraining_folder
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
                logger.info("Starting accumulation")
                #create folder to store accumulation models
                video.create_accumulation_folder(iteration_number = i)
                reset_blobs_fragmentation_parameters(blobs, recovering_from = 'accumulation')
                #Reset used_for_training and acceptable_for_training flags if the old video already had the accumulation done
                [global_fragment.reset_accumulation_params() for global_fragment in global_fragments]
                logger.info("We will restore the network from a previous pretraining: %s" %video._pretraining_folder)
                accumulation_network_params.save_folder = video._accumulation_folder
                accumulation_network_params.restore_folder = video._pretraining_folder
                accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
                logger.info("Initialising accumulation network")
                net = ConvNetwork(accumulation_network_params)
                #restore variables from the pretraining
                net.restore()
                net.reinitialize_softmax_and_fully_connected()
                #instantiate accumulation manager
                logger.info("Initialising accumulation manager")
                accumulation_manager = AccumulationManager(blobs, global_fragments, video.number_of_animals)
                #set global epoch counter to 0
                logger.info("Start accumulation")
                global_step = 0
                ratio_accumulated_images_over_all_unique_images_in_global_fragments = accumulate(accumulation_manager,
                                                                                                    video,
                                                                                                    blobs,
                                                                                                    global_fragments,
                                                                                                    global_step,
                                                                                                    net,
                                                                                                    video.knowledge_transfer_from_same_animals,
                                                                                                    get_ith_global_fragment = i)
                logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
                if ratio_accumulated_images_over_all_unique_images_in_global_fragments > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                    break
                else:
                    percentage_of_accumulated_images.append(ratio_accumulated_images_over_all_unique_images_in_global_fragments)
                    logger.info("This accumulation was not satisfactory. Try to start from a different global fragment")

            if len(percentage_of_accumulated_images) > 1 and np.argmax(percentage_of_accumulated_images) != 2:
                accumulation_folder_name = 'accumulation_' + str(np.argmax(percentage_of_accumulated_images))
                video._accumulation_folder = os.path.join(video._session_folder, accumulation_folder_name)
            video._second_accumulation_finished = True
            logger.info("Saving global fragments")
            np.save(video.global_fragments_path, global_fragments)
            ### NOTE: save second_accumulation statistics
            video.save()
        else:
            video.copy_attributes_between_two_video_objects(old_video, ['_accumulation_folder', '_second_accumulation_finished'])
            ### NOTE: load pre-training statistics
        video.second_accumulation_time = time.time() - video.second_accumulation_time
        video.assignment_time = time.time()
        if not loadPreviousDict['assignment']:
            #### Assigner ####
            assigner(blobs, video, net)
            video._has_been_assigned = True
            ### NOTE: save all the assigner statistics
        else:
            ### NOTE: load all the assigner statistics
            video._has_been_assigned = True
        video.assignment_time = time.time() - video.assignment_time

    # finish and save
    logger.info("Saving blobs objects and video object")
    blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
    blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
    blobs_list.cut_in_chunks()
    logger.info("Saving list of blobs")
    blobs_list.save()
    logger.info("Saving global fragments")
    np.save(video.global_fragments_path, global_fragments)
    logger.info("Saving video")
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
        reset_blobs_fragmentation_parameters(blobs, recovering_from = 'solving_duplications')
        # mark blobs as duplications
        mark_blobs_as_duplications(blobs, video.number_of_animals)
        # solve duplications
        solve_duplications(video, blobs, global_fragments, video.number_of_animals)
        video._has_duplications_solved = True
        logger.info("Done")
        # finish and save
        logger.info("Saving")
        blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
        blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
        blobs_list.cut_in_chunks()
        blobs_list.save()
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
    video.solve_impossible_jumps_time = time.time()
    print("\n**** Correct impossible velocity jump ****")
    logging.info("Solving impossible velocity jumps")
    if hasattr(old_video,'velocity_threshold') and not hasattr(video,'velocity_threshold'):
        video.velocity_threshold = old_video.velocity_threshold
    elif not hasattr(old_video, 'velocity_threshold') and not hasattr(video,'velocity_threshold'):
        video.velocity_threshold = compute_model_velocity(blobs, video.number_of_animals, percentile = VEL_PERCENTILE)
    if hasattr(old_video, 'first_frame_for_validation'):
        video.first_frame_for_validation = old_video.first_frame_for_validation
    elif not hasattr(old_video, 'first_frame_for_validation'):
        max_distance_travelled_global_fragment = order_global_fragments_by_distance_travelled(global_fragments)[0]
        video.first_frame_for_validation = max_distance_travelled_global_fragment.index_beginning_of_fragment
    video.save()
    fix_identity_of_blobs_in_video(blobs)
    correct_impossible_velocity_jumps(video, blobs)
    logger.info("Done")
    # finish and save
    logger.info("Saving")
    blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
    blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
    blobs_list.cut_in_chunks()
    blobs_list.save()
    video.compute_overall_P2(blobs)
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
    ##############   Create trajectories    #####################
    ####
    #############################################################
    video.generate_trajectories_time = time.time()
    if not loadPreviousDict['trajectories']:
        video.create_trajectories_folder()
        logger.info("Generating trajectories. The trajectories files are stored in %s" %video.trajectories_folder)
        number_of_animals = video.number_of_animals
        number_of_frames = len(blobs)
        trajectories = produce_trajectories(blobs, number_of_frames, number_of_animals)
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
    groundtruth_path = os.path.join(video._video_folder,'_groundtruth.npy')
    if os.path.isfile(groundtruth_path):
        print("\n**** Computing accuracy wrt. groundtruth ****")
        groundtruth = np.load(groundtruth_path).item()
        print(len(groundtruth.list_of_blobs))
        print(len(blobs))
        groundtruth.list_of_blobs = groundtruth.list_of_blobs[groundtruth.start:groundtruth.end]
        blobs_to_compare_with_groundtruth = blobs[groundtruth.start:groundtruth.end]

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
