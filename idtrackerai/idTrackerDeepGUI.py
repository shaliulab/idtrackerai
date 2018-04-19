# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)
 

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
import copy
import tensorflow as tf
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
#import from idTrackerai
from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments, create_list_of_fragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments,\
                                        create_list_of_global_fragments
from global_fragments_statistics import compute_and_plot_fragments_statistics
from idtrackerai.preprocessing.segmentation import segment, resegment
from idtrackerai.preprocessing.erosion import compute_erosion_disk
from idtrackerai.utils.GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview,\
                    resegmentation_preview, selectPreprocParams, selectDir, \
                    check_resolution_reduction
from idtrackerai.utils.py_utils import  getExistentFiles
from idtrackerai.utils.video_utils import check_background_substraction
from idtrackerai.crossing_detector import detect_crossings
from idtrackerai.pre_trainer import pre_trainer
from idtrackerai.accumulation_manager import AccumulationManager
from idtrackerai.accumulator import accumulate
from idtrackerai.network.identification_model.network_params import NetworkParams
from idtrackerai.trainer import train
from idtrackerai.assigner import assigner
from visualize_embeddings import visualize_embeddings_global_fragments
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from correct_duplications import solve_duplications,\
                                    mark_fragments_as_duplications
from idtrackerai.postprocessing.correct_impossible_velocity_jumps import correct_impossible_velocity_jumps
from idtrackerai.postprocessing.get_trajectories import produce_output_dict
from idtrackerai.groundtruth_utils.generate_groundtruth import GroundTruth, GroundTruthBlob
from idtrackerai.groundtruth_utils.compute_groundtruth_statistics import get_accuracy_wrt_groundtruth
from idtrackerai.postprocessing.compute_velocity_model import compute_model_velocity
from idtrackerai.postprocessing.assign_them_all import close_trajectories_gaps
from idtrackerai.postprocessing.identify_non_assigned_with_interpolation import assign_zeros_with_interpolation_identities
# from idtrackerai.visualise_cnn import visualise

from idtrackerai.constants import  THRESHOLD_ACCEPTABLE_ACCUMULATION, VEL_PERCENTILE
np.random.seed(0)

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
    processes_list = ['preprocessing',
                    'first_accumulation',
                    'pretraining',
                    'second_accumulation',
                    'assignment',
                    'solving_duplications',
                    'crossings',
                    'trajectories',
                    'trajectories_wo_gaps']
    #get existent files and paths to load them
    existentFiles, old_video = getExistentFiles(video, processes_list)
    #selecting files to load from previous session...'
    loadPreviousDict = selectOptions(processes_list, existentFiles,
                    text='Steps already processed in this video \n (loaded from ' + video.video_folder + ')')
    #use previous values and parameters (bkg, roi, preprocessing parameters)?
    logger.debug("Video session folder: %s " %video.session_folder)
    video.save()
    # try:
    #############################################################
    ##################   Knowledge transfer  ####################
    ####   Take the weights from a different model already   ####
    ####   trained. Works better when transfering to similar ####
    ####   conditions (light, animal type, age, ...)         ####
    #############################################################
    loadPreviousDict['use_previous_knowledge_transfer_decision'] = True
    if not bool(loadPreviousDict['use_previous_knowledge_transfer_decision']):
        knowledge_transfer_flag = getInput('Knowledge transfer','Do you want to perform knowledge transfer from another model? [y]/n')
        if knowledge_transfer_flag.lower() == 'y' or knowledge_transfer_flag == '':
            video.knowledge_transfer_model_folder = selectDir('', text = "Select a session folder to perform knowledge transfer from the last accumulation point") #select path to video
            video._tracking_with_knowledge_transfer = True
            video.knowledge_transfer_info_dict = np.load(os.path.join(video.knowledge_transfer_model_folder, 'info.npy')).item()
            same_animals = getInput("Same animals", "Are you tracking the same animals? y/N").lower()
            if same_animals == 'y':
                video._identity_transfer = True
                #we transfer identities if the number of animals to be identified is less or equal to the one providing the knowledge
                if not video.is_identity_transfer_possible():
                    video._identity_transfer = False
                    logger.warn("Identity transfer is not possible. We will proceed by using standard knowledge transfer.")
            elif same_animals == 'n' or same_animals == '':
                video._identity_transfer = False
            else:
                raise ValueError("Invalid input.")

        elif knowledge_transfer_flag.lower() == 'n':
            video._tracking_with_knowledge_transfer = False
        else:
            raise ValueError("Invalid input, type either 'y' or 'n'")
    else:
        video.copy_attributes_between_two_video_objects(old_video, ['knowledge_transfer_model_folder',
                                                                    'identity_transfer',
                                                                    'tracking_with_knowledge_transfer'],
                                                                    [False, True, True,False])
        if old_video.tracking_with_knowledge_transfer:
            video.knowledge_transfer_info_dict = old_video.knowledge_transfer_info_dict
        video.use_previous_knowledge_transfer_decision = True
    #############################################################
    ####################  Preprocessing   #######################
    #### 1. detect blobs in the video                        ####
    #### 2. create a list of potential global fragments      ####
    #### in which all animals are visible.                   ####
    #### 3. compute a model of the area of the animals,      ####
    #### train the Deep Crossing Detector and identify       ####
    #### the crossings                                       ####
    #### 4. compute global fragments                         ####
    #### 5. create a list of objects GlobalFragment()        ####
    #############################################################
    #Selection/loading preprocessing parameters
    print('\nPreprocessing ---------------------------------------------------------')
    usePreviousPrecParams = bool(loadPreviousDict['preprocessing'])
    restore_segmentation = selectPreprocParams(video, old_video, usePreviousPrecParams)
    video.save()
    preprocessing_parameters_dict = {key: getattr(video, key)
                                    for key in video.__dict__ if 'apply_ROI' in key
                                    or 'subtract_bkg' in key
                                    or 'min' in key or 'max' in key}
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
        print("restore segmentation ", restore_segmentation)
        if not old_video or not old_video.has_been_segmented or not restore_segmentation:
            logger.debug("Starting segmentation")
            blobs = segment(video)
            logger.debug("Segmentation finished")
            list_of_blobs = ListOfBlobs(blobs_in_video = blobs)
            frames_with_more_blobs_than_animals = list_of_blobs.check_maximal_number_of_blob(video.number_of_animals)
            while len(frames_with_more_blobs_than_animals) > 0:
                new_preprocessing_parameters = {'min_threshold': video.min_threshold,
                                            'max_threshold': video.max_threshold,
                                            'min_area': video.min_area,
                                            'max_area': video.max_area}
                new_preprocessing_parameters = resegmentation_preview(video, frames_with_more_blobs_than_animals[0], new_preprocessing_parameters)

                for frame_number in tqdm(frames_with_more_blobs_than_animals, desc = 'Correcting segmentation'):
                    maximum_number_of_blobs = resegment(video, frame_number, list_of_blobs, new_preprocessing_parameters)
                    if maximum_number_of_blobs <= video.number_of_animals:
                        video._resegmentation_parameters.append((frame_number,new_preprocessing_parameters))
                frames_with_more_blobs_than_animals = list_of_blobs.check_maximal_number_of_blob(video.number_of_animals)
                cv2.namedWindow('Bars')

            video._has_been_segmented = True
            if len(list_of_blobs.blobs_in_video[-1]) == 0:
                list_of_blobs.blobs_in_video = list_of_blobs.blobs_in_video[:-1]
                list_of_blobs.number_of_frames = len(list_of_blobs.blobs_in_video)
                video._number_of_frames = list_of_blobs.number_of_frames
                video.save()
            list_of_blobs.save(video, video.blobs_path_segmented, number_of_chunks = video.number_of_frames)
            logger.debug("Segmented blobs saved")
            logger.info("Computing maximum number of blobs detected in the video")

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
            list_of_blobs = ListOfBlobs.load(video, old_video.blobs_path_segmented)
            video._has_been_segmented = True
            logger.debug("Segmented blobs loaded")
        video.save()

        if False:
            video._erosion_kernel_size = compute_erosion_disk(video, list_of_blobs.blobs_in_video)
            list_of_blobs.erode(video)

        logger.info("Computing a model of the area of the individuals")
        video._model_area, video._median_body_length = list_of_blobs.compute_model_area_and_body_length(video.number_of_animals)
        video.compute_identification_image_size(video.median_body_length)
        # video._identification_image_size = (48, 48, 1)
        if not list_of_blobs.blobs_are_connected:
            list_of_blobs.compute_overlapping_between_subsequent_frames()
        detect_crossings(list_of_blobs, video, video.model_area, use_network = True)
        list_of_blobs.compute_overlapping_between_subsequent_frames()
        list_of_blobs.compute_fragment_identifier_and_blob_index(video.number_of_animals)
        list_of_blobs.compute_crossing_fragment_identifier()
        fragments = create_list_of_fragments(list_of_blobs.blobs_in_video,
                                            video.number_of_animals)
        list_of_fragments = ListOfFragments(fragments)
        video._fragment_identifier_to_index = list_of_fragments.get_fragment_identifier_to_index_list()
        #compute the global fragments (all animals are visible + each animals overlaps
        #with a single blob in the consecutive frame + the blobs respect the area model)
        global_fragments = create_list_of_global_fragments(list_of_blobs.blobs_in_video,
                                                            list_of_fragments.fragments,
                                                            video.number_of_animals)
        list_of_global_fragments = ListOfGlobalFragments(global_fragments)
        video.number_of_global_fragments = list_of_global_fragments.number_of_global_fragments
        list_of_global_fragments.filter_candidates_global_fragments_for_accumulation()
        video.number_of_global_fragments_candidates_for_accumulation = list_of_global_fragments.number_of_global_fragments
        video.individual_fragments_lenghts, \
        video.individual_fragments_distance_travelled, \
            video._gamma_fit_parameters = compute_and_plot_fragments_statistics(video,
                                                                            video.model_area,
                                                                            list_of_blobs,
                                                                            list_of_fragments,
                                                                            list_of_global_fragments)
        list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
        video._number_of_unique_images_in_global_fragments = list_of_fragments.compute_total_number_of_images_in_global_fragments()
        list_of_global_fragments.compute_maximum_number_of_images()
        video._maximum_number_of_images_in_global_fragments = list_of_global_fragments.maximum_number_of_images
        list_of_fragments.get_accumulable_individual_fragments_identifiers(list_of_global_fragments)
        list_of_fragments.get_not_accumulable_individual_fragments_identifiers(list_of_global_fragments)
        list_of_fragments.set_fragments_as_accumulable_or_not_accumulable()
        #save connected blobs in video (organized frame-wise)
        video._has_been_preprocessed = True
        list_of_blobs.save(video, video.blobs_path, number_of_chunks = video.number_of_frames)
        list_of_fragments.save(video.fragments_path)
        list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
        video.save()
        logger.info("Blobs detection and fragmentation finished succesfully.")
    else:
        cv2.namedWindow('Bars')
        logger.info("Loading preprocessed video")
        path_attributes = ['preprocessing_folder', 'blobs_path', 'global_fragments_path', 'fragments_path', 'gamma_fit_parameters']
        video.copy_attributes_between_two_video_objects(old_video, path_attributes)
        video._has_been_segmented = True
        video._has_been_preprocessed = True
        video.save()
        # Load blobs and global fragments
        logger.info("Loading blob objects")
        list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
        logger.info("Loading list of fragments")
        list_of_fragments = ListOfFragments.load(video.fragments_path)
        logger.info("Loading list of global fragments")
        list_of_global_fragments = ListOfGlobalFragments.load(video.global_fragments_path, list_of_fragments.fragments)
    video.preprocessing_time = time.time() - video.preprocessing_time
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
    video.create_accumulation_folder(iteration_number = 0, delete = not bool(loadPreviousDict['first_accumulation']))
    logger.info("Set accumulation network parameters")
    number_of_animals = video.number_of_animals if not video.identity_transfer else video.knowledge_transfer_info_dict['number_of_animals']
    accumulation_network_params = NetworkParams(number_of_animals,
                                learning_rate = 0.005,
                                keep_prob = 1.0,
                                scopes_layers_to_optimize = None,
                                save_folder = video.accumulation_folder,
                                image_size = video.identification_image_size,
                                video_path = video.video_path)
    if not bool(loadPreviousDict['first_accumulation']):
        logger.info("Starting accumulation")
        list_of_fragments.reset(roll_back_to = 'fragmentation')
        list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        if video.tracking_with_knowledge_transfer:
            if video.identity_transfer:
                logger.info("We will restore the network from a previous model (convolutional layers and classifier): %s" %video.knowledge_transfer_model_folder)
                accumulation_network_params.restore_folder = video.knowledge_transfer_model_folder
                accumulation_network_params.check_identity_transfer_consistency(video.knowledge_transfer_info_dict)
            else:
                logger.info("We will restore the network from a previous model (only convolutional layers): %s" %video.knowledge_transfer_model_folder)
                accumulation_network_params.knowledge_transfer_folder = video.knowledge_transfer_model_folder
            accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        else:
            logger.info("The network will be trained from scratch during accumulation")
        logger.info("Initialising accumulation network")
        net = ConvNetwork(accumulation_network_params)
        #if knowledge transfer is performed on the same animals we don't reinitialise the classification part of the net
        if video.tracking_with_knowledge_transfer:
            net.restore()
        logger.info("Initialising accumulation manager")
        # the list of global fragments is ordered in place from the distance (in frames) wrt
        # the core of the first global fragment that will be accumulated
        video._first_frame_first_global_fragment.append(list_of_global_fragments.set_first_global_fragment_for_accumulation(video, accumulation_trial = 0))
        if video.identity_transfer and video.number_of_animals < video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            accumulation_network_params.number_of_animals = video.number_of_animals
            accumulation_network_params._restore_folder = None
            accumulation_network_params.knowledge_transfer_folder = video.knowledge_transfer_model_folder
            net = ConvNetwork(accumulation_network_params)
            net.restore()

        list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(video, accumulation_trial = 0)
        accumulation_manager = AccumulationManager(video, list_of_fragments,
                                                    list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        #set global epoch counter to 0
        logger.info("Start accumulation")
        global_step = 0
        video._ratio_accumulated_images = accumulate(accumulation_manager,
                                            video,
                                            global_step,
                                            net,
                                            video.identity_transfer)
        logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
        video._first_accumulation_finished = True
        video._percentage_of_accumulated_images = [video.ratio_accumulated_images]
        video.save()
        logger.info("Saving fragments")
        list_of_fragments.save(video.fragments_path)
        list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
    else:
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
                    'percentage_of_accumulated_images', 'accumulation_trial',
                    'ratio_accumulated_images', 'first_accumulation_finished',
                    'identity_transfer', 'accumulation_statistics',
                    'first_frame_first_global_fragment']
        is_property = [True, True, False, False,
                        False, False, False, False,
                        False, False, False, False,
                        True, False, True, True,
                        True, False, True]
        video.copy_attributes_between_two_video_objects(old_video, list_of_attributes, is_property = is_property)
        video._ratio_accumulated_images = video.percentage_of_accumulated_images[0]
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
        video.assignment_time = time.time()
        if not loadPreviousDict['assignment']:
            print('\nAssignment 1 ---------------------------------------------------------')
            list_of_fragments.reset(roll_back_to = 'accumulation')
            assigner(list_of_fragments, video, net)
            video._has_been_assigned = True
            ### NOTE: save all the assigner statistics
        else:
            ### NOTE: load all the assigner statistics
            video._has_been_assigned = True
        video.assignment_time = time.time() - video.assignment_time
        video.pretraining_time = 0
        video.second_accumulation_time = 0
        video.save()
    else:
        print('\nPretraining ---------------------------------------------------------')
        video.pretraining_time = time.time()
        video.create_pretraining_folder()
        pretrain_network_params = NetworkParams(video.number_of_animals,
                                                learning_rate = 0.01,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = None,
                                                save_folder = video.pretraining_folder,
                                                image_size = video.identification_image_size,
                                                video_path = video.video_path)
        if not loadPreviousDict['pretraining']:
            #### Pre-trainer ####
            list_of_fragments.reset(roll_back_to = 'fragmentation')
            list_of_global_fragments.order_by_distance_travelled()
            pre_trainer(old_video, video, list_of_fragments, list_of_global_fragments, pretrain_network_params)
            logger.info("Pretraining ended")
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
        video.second_accumulation_time = time.time()
        video._percentage_of_accumulated_images = [video.ratio_accumulated_images]
        print("****************************************************************")
        print("**************************** loadPreviousDict ", loadPreviousDict)
        print("****************************************************************")
        if not loadPreviousDict['second_accumulation']:
            if isinstance(video.first_frame_first_global_fragment, int):
                video._first_frame_first_global_fragment = [video.first_frame_first_global_fragment]
            for i in range(1,4):
                print('\nAccumulation %i ---------------------------------------------------------' %i)
                logger.info("Starting accumulation")
                video.create_accumulation_folder(iteration_number = i, delete = not bool(loadPreviousDict['second_accumulation']))
                video.accumulation_trial = i
                list_of_fragments.reset(roll_back_to = 'fragmentation')
                list_of_global_fragments.reset(roll_back_to = 'fragmentation')
                logger.info("We will restore the network from a previous pretraining: %s" %video.pretraining_folder)
                accumulation_network_params.save_folder = video.accumulation_folder
                accumulation_network_params.restore_folder = video.pretraining_folder
                accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
                logger.info("Initialising accumulation network")
                net = ConvNetwork(accumulation_network_params)
                net.restore()
                net.reinitialize_softmax_and_fully_connected()
                logger.info("Initialising accumulation manager")
                video._first_frame_first_global_fragment.append(list_of_global_fragments.set_first_global_fragment_for_accumulation(video, accumulation_trial = i - 1))
                if video.identity_transfer and video.number_of_animals < video.knowledge_transfer_info_dict['number_of_animals']:
                    tf.reset_default_graph()
                    accumulation_network_params.number_of_animals = video.number_of_animals
                    accumulation_network_params.restore_folder = video.pretraining_folder
                    net = ConvNetwork(accumulation_network_params)
                    net.restore()
                    net.reinitialize_softmax_and_fully_connected()
                list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(video, accumulation_trial = i - 1)
                accumulation_manager = AccumulationManager(video,
                                                            list_of_fragments, list_of_global_fragments,
                                                            threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
                logger.info("Start accumulation")
                global_step = 0
                video._ratio_accumulated_images = accumulate(accumulation_manager,
                                                            video,
                                                            global_step,
                                                            net,
                                                            video.identity_transfer)
                logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
                video._percentage_of_accumulated_images.append(video.ratio_accumulated_images)
                list_of_fragments.save_light_list(video._accumulation_folder)
                if video.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                    break
                else:
                    logger.info("This accumulation was not satisfactory. Try to start from a different global fragment")



            video.accumulation_trial = np.argmax(video.percentage_of_accumulated_images)
            video._first_frame_first_global_fragment = video.first_frame_first_global_fragment[video.accumulation_trial]
            video._ratio_accumulated_images = video.percentage_of_accumulated_images[video.accumulation_trial]
            accumulation_folder_name = 'accumulation_' + str(video.accumulation_trial)
            video._accumulation_folder = os.path.join(video.session_folder, accumulation_folder_name)
            list_of_fragments.load_light_list(video._accumulation_folder)
            video._second_accumulation_finished = True
            logger.info("Saving global fragments")
            list_of_fragments.save(video.fragments_path)
            list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
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
                        'percentage_of_accumulated_images', 'accumulation_trial',
                        'ratio_accumulated_images', 'first_accumulation_finished',
                        'identity_transfer', 'accumulation_statistics',
                        'first_frame_first_global_fragment']
            is_property = [True, True, False, False,
                            False, False, False, False,
                            False, False, False, False,
                            True, False, True, True,
                            True, False, True]
            video.copy_attributes_between_two_video_objects(old_video, list_of_attributes)
            logger.info("Restoring trained network")
            accumulation_network_params.restore_folder = video._accumulation_folder
            list_of_fragments.load_light_list(video._accumulation_folder)
            net = ConvNetwork(accumulation_network_params)
            net.restore()
            video.save()
            ### NOTE: load pre-training statistics
        video.second_accumulation_time = time.time() - video.second_accumulation_time
        video.assignment_time = time.time()
        if not loadPreviousDict['assignment']:
            #### Assigner ####
            print('\n---------------------------------------------------------')
            list_of_fragments.reset(roll_back_to = 'accumulation')
            assigner(list_of_fragments, video, net)
            video._has_been_assigned = True
            ### NOTE: save all the assigner statistics
        else:
            ### NOTE: load all the assigner statistics
            video._has_been_assigned = True
        video.assignment_time = time.time() - video.assignment_time
        video.save()

    logger.debug("Saving list of fragments, list of global fragments and video object")
    list_of_fragments.save(video.fragments_path)
    list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
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

    print("************** Before solving duplications ************************")
    print("Number of fragments with zero identity: ", len([f for f in list_of_fragments.fragments
                                                            if f.assigned_identity == 0]))
    print("Number of fragments with zero identity by P2: ",
                    len([f for f in list_of_fragments.fragments
                    if f.assigned_identity == 0
                    and hasattr(f, 'zero_identity_assigned_by_P2')]))

    #############################################################
    ###################   Solve duplications      ###############
    ####
    #############################################################
    # video.solve_duplications_time = time.time()
    # if not loadPreviousDict['solving_duplications']:
    #     logger.info("Start checking for and solving duplications")
    #     list_of_fragments.reset(roll_back_to = 'assignment')
    #     mark_fragments_as_duplications(list_of_fragments.fragments)
    #     solve_duplications(list_of_fragments, video.first_frame_first_global_fragment)
    #     video._has_duplications_solved = True
    #     logger.info("Saving")
    #     list_of_fragments.save(video.fragments_path)
    #     video.save()
    # else:
    #     logger.info("Duplications have already been checked. Using previous information")
    #     video._has_duplications_solved = True
    #     video.save()
    # video.solve_duplications_time = time.time() - video.solve_duplications_time
    #
    # print("************** After solving duplications ************************")
    # print("Number of fragments with zero identity: ", len([f for f in list_of_fragments.fragments
    #                                                         if f.assigned_identity == 0]))
    # print("Number of fragments with zero identity by P2: ",
    #                 len([f for f in list_of_fragments.fragments
    #                 if f.assigned_identity == 0
    #                 and hasattr(f, 'zero_identity_assigned_by_P2')]))

    #############################################################
    ###################  Solving impossible jumps    ############
    #############################################################
    video.solve_impossible_jumps_time = time.time()
    print("\n**** Correct impossible velocity jump ****")
    logging.info("Solving impossible velocity jumps")
    if hasattr(old_video,'velocity_threshold') and not hasattr(video,'velocity_threshold'):
        video.velocity_threshold = old_video.velocity_threshold
    elif not hasattr(old_video, 'velocity_threshold') and not hasattr(video,'velocity_threshold'):
        video.velocity_threshold = compute_model_velocity(list_of_fragments.fragments, video.number_of_animals, percentile = VEL_PERCENTILE)
    correct_impossible_velocity_jumps(video, list_of_fragments)
    logger.info("Saving")
    list_of_fragments.save(video.fragments_path)
    video.save()
    logger.info("Done")
    video.solve_impossible_jumps_time = time.time() - video.solve_impossible_jumps_time

    print("************** After solving impossible jumps ************************")
    print("Number of fragments with zero identity: ", len([f for f in list_of_fragments.fragments
                                                            if f.assigned_identity == 0]))
    print("Number of fragments with zero identity by P2: ",
                    len([f for f in list_of_fragments.fragments
                    if f.assigned_identity == 0
                    and hasattr(f, 'zero_identity_assigned_by_P2')]))

    #############################################################
    ############# Check identification consistency ##############
    ####
    #############################################################



    #############################################################
    ##############   Invididual fragments stats #################
    ####
    #############################################################
    video.individual_fragments_stats = list_of_fragments.get_stats(list_of_global_fragments)
    video.compute_overall_P2(list_of_fragments.fragments)
    print("overall_P2 ", video.overall_P2)
    list_of_fragments.plot_stats(video)
    list_of_fragments.save_light_list(video._accumulation_folder)
    video.save()

    #############################################################
    ##############   Update list of blobs   #####################
    ####
    #############################################################
    list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments, video.fragment_identifier_to_index)
    if False:
        list_of_blobs.compute_nose_and_head_coordinates()
    list_of_blobs.save(video, video.blobs_path, number_of_chunks = video.number_of_frames)

    #############################################################
    ############ Create trajectories (w gaps) ###################
    #############################################################
    video.generate_trajectories_time = time.time()
    if not loadPreviousDict['trajectories']:
        video.create_trajectories_folder()
        trajectories_file = os.path.join(video.trajectories_folder, 'trajectories.npy')
        trajectories = produce_output_dict(list_of_blobs.blobs_in_video, video)
        np.save(trajectories_file, trajectories)
        logger.info("Saving trajectories")
        video._has_trajectories = True
        video.save()
    else:
        video._has_trajectories = True
        video.save()
    video.generate_trajectories_time = time.time() - video.generate_trajectories_time

    #############################################################
    ##############   Compute groundtruth    #####################
    #############################################################
    groundtruth_path = os.path.join(video.video_folder,'_groundtruth.npy')
    if os.path.isfile(groundtruth_path):
        print("\n**** Computing accuracy wrt. groundtruth ****")
        try:
            groundtruth = np.load(groundtruth_path).item()
            blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
            blobs_in_video = list_of_blobs.blobs_in_video[groundtruth.start:groundtruth.end]
            video.gt_accuracy, video.gt_results = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth, blobs_in_video)
            video.gt_start_end = (groundtruth.start, groundtruth.end)
            video.save()
        except:
            print("error computing the ground truth")

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

    #############################################################
    ##############   Solve crossigns   ##########################
    ####
    #############################################################
    print("\n**** Assign crossings ****")
    if not loadPreviousDict['crossings']:
        list_of_blobs_no_gaps = copy.deepcopy(list_of_blobs)
        # if not hasattr(list_of_blobs_no_gaps.blobs_in_video[0][0], '_was_a_crossing'):
        #     logger.debug("adding attribute was_a_crossing to every blob")
        #     [setattr(blob, '_was_a_crossing', False) for blobs_in_frame in
        #         list_of_blobs_no_gaps.blobs_in_video for blob in blobs_in_frame]
        video._has_crossings_solved = False
        list_of_blobs_no_gaps = close_trajectories_gaps(video, list_of_blobs_no_gaps, list_of_fragments)
        video.blobs_no_gaps_path = os.path.join(os.path.split(video.blobs_path)[0], 'blobs_collection_no_gaps.npy')
        list_of_blobs_no_gaps.save(video, path_to_save = video.blobs_no_gaps_path, number_of_chunks = video.number_of_frames)
        video._has_crossings_solved = True
        video.save()
    else:
        video.copy_attributes_between_two_video_objects(old_video, ['blobs_no_gaps_path'], [False])
        list_of_blobs_no_gaps = ListOfBlobs.load(video, video.blobs_no_gaps_path)
        video._has_crossings_solved = True
        video.save()

    #############################################################
    ########### Create trajectories (w/o gaps) ##################
    #############################################################
    video.generate_trajectories_wogaps_time = time.time()
    if not loadPreviousDict['trajectories_wo_gaps']:
        video.create_trajectories_wo_gaps_folder()
        logger.info("Generating trajectories. The trajectories files are stored in %s" %video.trajectories_wo_gaps_folder)
        trajectories_wo_gaps_file = os.path.join(video.trajectories_wo_gaps_folder, 'trajectories_wo_gaps.npy')
        trajectories_wo_gaps = produce_output_dict(list_of_blobs_no_gaps.blobs_in_video, video)
        np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)
        logger.info("Saving trajectories")
        video._has_trajectories_wo_gaps = True
        video.save()
    else:
        video._has_trajectories_wo_gaps = True
        video.save()
    video.generate_trajectories_wogaps_time = time.time() - video.generate_trajectories_wogaps_time