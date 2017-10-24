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
import pandas as pd
from pprint import pprint
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
sys.path.append('./plots')
sys.path.append('./library')
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

from library_utils import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs, LibraryJobConfig, check_if_repetition_has_been_computed

NUM_CHUNKS_BLOB_SAVING = 500 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
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
    '''
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: test_number

    e.g.
    run_library_tests.py 1 1 P None 0 .5 .1 DEF afs 1_2 (running in the cluster, job1, pretraining, libraries DEF, all individuals in library D and first half obf E second half of F, repetitions[1 2])
    '''
    print('\n\n ********************************************* \n\n')
    print("cluster:", sys.argv[1])
    print("test_number:", sys.argv[2])

    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
    test_dictionary = tests_data_frame.loc[int(sys.argv[2])].to_dict()
    pprint(test_dictionary)

    job_config = LibraryJobConfig(cluster = sys.argv[1], test_dictionary = test_dictionary)
    job_config.create_folders_structure()

    if os.path.isfile('./library/results_data_frame.pkl'):
        print("results_data_frame.pkl already exists \n")
        results_data_frame = pd.read_pickle('./library/results_data_frame.pkl')
    else:
        print("results_data_frame.pkl does not exist \n")
        results_data_frame = pd.DataFrame()

    dataset = Dataset(IMDB_codes = job_config.IMDB_codes, ids_codes = job_config.ids_codes, preprocessing_type = job_config.preprocessing_type)
    dataset.loadIMDBs()
    print("images shape, ", dataset.images.shape)
    imsize = dataset.images.shape[1]

    for repetition in job_config.repetitions:

        for group_size in job_config.group_sizes:

            for frames_in_video in job_config.frames_in_video:

                for i, (mean_frames_per_fragment, var_frames_per_fragment) in enumerate(zip(job_config.mean_frames_per_individual_fragment,job_config.std_frames_per_individual_fragment)):

                    repetition_path = os.path.join(job_config.condition_path,'group_size_' + str(group_size),
                                                            'num_frames_' + str(frames_in_video),
                                                            'mean_frames_in_fragment_' + str(mean_frames_per_fragment),
                                                            'std_frames_in_fragment_' + str(var_frames_per_fragment),
                                                            'repetition_' + str(repetition))


                    print("\n********** group size %i - frames_in_video %i - mean_frames_per_fragment %s -  std_frames_in_fragment %s - repetition %i ********" %(group_size,frames_in_video,str(mean_frames_per_fragment), str(var_frames_per_fragment), repetition))
                    already_computed = False
                    if os.path.isfile('./library/results_data_frame.pkl'):
                        already_computed = check_if_repetition_has_been_computed(results_data_frame, job_config, group_size, frames_in_video, mean_frames_per_fragment, var_frames_per_fragment, repetition)
                        print("already_computed flag: ", already_computed)
                    if already_computed:
                        print("The algorithm with this comditions has been already tested")
                    else:

                        video = Video() #instantiate object video
                        video.video_path = os.path.join(repetition_path,'_fake_0.avi') #set path
                        video.create_session_folder()
                        video._number_of_animals = group_size #int: number of animals in the video
                        video._maximum_number_of_blobs = group_size #int: the maximum number of blobs detected in the video
                        video.number_of_frames = frames_in_video
                        video.tracking_with_knowledge_transfer = job_config.knowledge_transfer_flag
                        video.knowledge_transfer_model_folder = job_config.knowledge_transfer_folder
                        video._identification_image_size = (imsize, imsize, 1) #NOTE: this can change if the library changes. BUILD next library with new preprocessing.

                        #############################################################
                        ####################   Preprocessing   ######################
                        #### prepare blobs list and global fragments from the    ####
                        #### library                                             ####
                        #############################################################

                        list_of_blobs_config = BlobsListConfig(number_of_animals = group_size,
                                                number_of_frames_per_fragment = mean_frames_per_fragment,
                                                var_number_of_frames_per_fragment = var_frames_per_fragment,
                                                number_of_frames = frames_in_video,
                                                repetition = repetition)
                        identification_images, centroids = subsample_dataset_by_individuals(dataset, list_of_blobs_config)
                        blobs = generate_list_of_blobs(identification_images, centroids, list_of_blobs_config)
                        video._has_been_segmented = True
                        list_of_blobs = ListOfBlobs(video, blobs_in_video = blobs)
                        logger.info("Computing maximum number of blobs detected in the video")
                        list_of_blobs.check_maximal_number_of_blob()
                        logger.info("Computing a model of the area of the individuals")
                        list_of_blobs.video = video
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

                        aaa

                        #############################################################
                        ##################      Pre-trainer      ####################
                        #### create the folder training in which all the         ####
                        #### CNN-related process will be stored. The structure   ####
                        #### is /training/session_num, where num is an natural   ####
                        #### number. num increases each time a training is       ####
                        #### launched                                            ####
                        #############################################################
                        start = time.time()
                        print("\n**** Pretraining ****\n")
                        if job_config.pretraining_flag:
                            if job_config.percentage_of_frames_in_pretaining != 1.:
                                number_of_pretraining_global_fragments = int(len(global_fragments) * job_config.percentage_of_frames_in_pretaining)
                                pretraining_global_fragments = order_global_fragments_by_distance_travelled(give_me_pre_training_global_fragments(global_fragments, number_of_pretraining_global_fragments = number_of_pretraining_global_fragments))
                            else:
                                number_of_pretraining_global_fragments = len(global_fragments)
                                pretraining_global_fragments = order_global_fragments_by_distance_travelled(global_fragments)
                            print("pretraining with %i" %number_of_pretraining_global_fragments, ' global fragments\n')
                            #create folder to store pretraining
                            video.create_pretraining_folder(number_of_pretraining_global_fragments)
                            #pretraining network parameters
                            pretrain_network_params = NetworkParams(video.number_of_animals,
                                                                    learning_rate = 0.01,
                                                                    keep_prob = 1.0,
                                                                    save_folder = video.pretraining_folder,
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
                                            save_summaries = False,
                                            print_flag = False,
                                            plot_flag = False)

                            video._has_been_pretrained = True
                        else:
                            print("no pretraining")
                            number_of_pretraining_global_fragments = 0

                        pretraining_time = time.time() - start
                        #############################################################
                        ###################    Accumulation   #######################
                        #### take references in 'good' global fragments          ####
                        #############################################################
                        start = time.time()
                        print("\n**** Acumulation ****")
                        #create folder to store accumulation models
                        video.create_accumulation_folder()
                        #set network params for the accumulation model
                        accumulation_network_params = NetworkParams(video.number_of_animals,
                                                    learning_rate = 0.005,
                                                    keep_prob = 1.0,
                                                    scopes_layers_to_optimize = ['fully-connected1','softmax1'],
                                                    save_folder = video._accumulation_folder,
                                                    image_size = video.portrait_size)
                        if video._has_been_pretrained:
                            print("We will restore the network from pretraining: %s\n" %video.pretraining_folder)
                            accumulation_network_params.restore_folder = video.pretraining_folder
                        elif not video._has_been_pretrained:
                            if video.tracking_with_knowledge_transfer:
                                print("We will restore the network from a previous model (knowledge transfer): %s\n" %video.knowledge_transfer_model_folder)
                                accumulation_network_params.restore_folder = video.knowledge_transfer_model_folder
                            else:
                                print("The network will be trained from scracth during accumulation\n")
                                accumulation_network_params.scopes_layers_to_optimize = None

                        if job_config.train_filters_in_accumulation == True:
                            accumulation_network_params.scopes_layers_to_optimize = None
                        #instantiate network object
                        net = ConvNetwork(accumulation_network_params)
                        #restore variables from the pretraining
                        net.restore()
                        net.reinitialize_softmax_and_fully_connected()
                        #instantiate accumulation manager
                        accumulation_manager = AccumulationManager(global_fragments, video.number_of_animals, certainty_threshold = job_config.accumulation_certainty)
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
                                                    save_summaries = False,
                                                    print_flag = False,
                                                    plot_flag = False,
                                                    global_step = global_step,
                                                    first_accumulation_flag = accumulation_manager.counter == 0)
                            # update used_for_training flag to True for fragments used
                            accumulation_manager.update_global_fragments_used_for_training()
                            # update the set of images used for training
                            accumulation_manager.update_used_images_and_labels()
                            # assign identities fo the global fragments that have been used for training
                            accumulation_manager.assign_identities_to_accumulated_global_fragments(blobs)
                            # update the list of individual fragments that have been used for training
                            accumulation_manager.update_individual_fragments_used()
                            # Set accumulation params for rest of the accumulation
                            # net.params.restore_folder = video._accumulation_folder
                            #take images from global fragments not used in training (in the remainder test global fragments)
                            candidates_next_global_fragments = [global_fragment for global_fragment in global_fragments if not global_fragment.used_for_training]
                            print("number of candidate global fragments, ", len(candidates_next_global_fragments))
                            if any([not global_fragment.used_for_training for global_fragment in global_fragments]):
                                images, _, candidate_individual_fragments_indices, indices_to_split = get_images_and_labels_from_global_fragments(candidates_next_global_fragments,[])
                            else:
                                print("All the global fragments have been used for accumulation")
                                break
                            # get predictions for images in test global fragments
                            assigner = assign(net, video, images, print_flag = True)
                            accumulation_manager.split_predictions_after_network_assignment(assigner._predictions, assigner._softmax_probs, indices_to_split)
                            # assign identities to the global fragments based on the predictions
                            accumulation_manager.assign_identities_and_check_eligibility_for_training_global_fragments(candidate_individual_fragments_indices)
                            accumulation_manager.update_counter()
                            if job_config.only_accumulate_one_fragment:
                                print("we only accumulate one fragment")
                                break

                        accumulation_time = time.time() - start
                        #############################################################
                        ###################     Assigner      ######################
                        ####
                        #############################################################
                        start = time.time()
                        print("\n**** Assignation ****")
                        # Get images from the blob collection
                        images = get_images_from_blobs_in_video(blobs)#, video.episodes_start_end)
                        if len(images) != 0:
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
                            # assign identity to the extremes of the fragments
                            # for blobs_in_frame in blobs:
                            #     for blob in blobs_in_frame:
                            #         #if a blob has not been assigned but it is a fish and overlaps with one fragment
                            #         #assign it!
                            #         if blob.identity == 0 and blob.is_an_individual:
                            #             if len(blob.next) == 1: blob.identity = blob.next[0].identity
                            #             elif len(blob.previous) == 1: blob.identity = blob.previous[0].identity
                        else:
                            print("All the global fragments have been used in the accumulation")
                        assignation_time = time.time() - start

                        #############################################################
                        ###################     Accuracies     ######################
                        ####
                        #############################################################
                        print("\n**** Accuracies ****")
                        number_of_possible_assignations = 0
                        number_of_correct_assignations = [0] * group_size
                        number_assignations = [0]*group_size
                        number_of_identity_repetitions = 0
                        number_of_frames_with_repetitions = 0
                        number_of_identity_shifts_in_accumulated_frames = 0
                        number_of_blobs_assigned_in_accumulation = 0
                        number_of_not_assigned_blobs = [0] * group_size
                        individual_fragments_badly_assigned_in_accumulation = []
                        individual_fragments_that_are_repetitions = []
                        individual_fragments = []
                        for frame_number, blobs_in_frame in enumerate(blobs):
                            identities_in_frame = []
                            frame_with_repetition = False
                            for i, blob in enumerate(blobs_in_frame):
                                if blob._fragment_identifier not in individual_fragments:
                                    individual_fragments.append(blob._fragment_identifier)
                                if blob.is_an_individual_in_a_fragment:
                                    number_of_possible_assignations += 1
                                    if blob._assigned_during_accumulation:
                                        number_of_blobs_assigned_in_accumulation += 1
                                    if blob.identity is not None and blob.identity != 0:
                                        number_assignations[blob.user_generated_identity-1] += 1
                                        if blob.identity == blob.user_generated_identity:
                                            number_of_correct_assignations[blob.user_generated_identity-1] += 1
                                        elif blob._assigned_during_accumulation:
                                            if blob._fragment_identifier not in individual_fragments_badly_assigned_in_accumulation:
                                                individual_fragments_badly_assigned_in_accumulation.append(blob._fragment_identifier)
                                            number_of_identity_shifts_in_accumulated_frames += 1
                                        if blob.identity in identities_in_frame:
                                            number_of_identity_repetitions += 1
                                            if blob._fragment_identifier not in individual_fragments_that_are_repetitions:
                                                individual_fragments_that_are_repetitions.append(blob._fragment_identifier)
                                            frame_with_repetition = True

                                        identities_in_frame.append(blob.identity)
                                    elif blob.identity is None or blob.identity == 0:
                                        number_of_not_assigned_blobs[i] += 1
                            if frame_with_repetition:
                                number_of_frames_with_repetitions += 1

                        number_of_acceptable_fragments = sum([global_fragment._acceptable_for_training for global_fragment in global_fragments])
                        number_of_unique_fragments = sum([global_fragment.is_unique for global_fragment in global_fragments])
                        number_of_certain_fragments = sum([global_fragment._is_certain for global_fragment in global_fragments])

                        individual_accuracies = np.asarray(number_of_correct_assignations)/np.asarray(number_assignations)
                        accuracy = np.sum(number_of_correct_assignations)/np.sum(number_assignations)
                        accuracy_overall = np.sum(number_of_correct_assignations)/number_of_possible_assignations
                        print("\n\ngroup_size: ", group_size)
                        print("individual_accuracies(assigned): ", individual_accuracies)
                        print("accuracy(assigned): ", accuracy)
                        print("accuracy: ", accuracy_overall)
                        print("\nnumber of global fragments: ", len(global_fragments))
                        print("number of accumulated fragments:", sum([global_fragment.used_for_training for global_fragment in global_fragments]))
                        print("number of candidate global fragments:", len(candidates_next_global_fragments))
                        print("number of acceptable fragments: ", number_of_acceptable_fragments)
                        print("number of unique fragments: ", number_of_unique_fragments + 1)
                        print("number of certain fragments: ", number_of_certain_fragments + 1)
                        print("\nnumber of individual fragments: ", len(individual_fragments))
                        print("number of individual fragments badly assigned in acumulation: ", len(individual_fragments_badly_assigned_in_accumulation))
                        print("number of fragments that are repetitions: ", len(individual_fragments_that_are_repetitions))
                        print("\nframes in video: ", frames_in_video)
                        print("number of frames with repetitions: ", number_of_frames_with_repetitions)
                        print("number of assignation: ", number_assignations)
                        print("number correct assignations: ", number_of_correct_assignations)
                        print("number of identity repetitions: ", number_of_identity_repetitions)
                        print("number of identity shifts in accumulated frames: ", number_of_identity_shifts_in_accumulated_frames)
                        print("****************************************************************************************************\n\n")

                        #############################################################
                        ###################  Duplications removal ###################
                        ####
                        #############################################################
                        start = time.time()
                        if job_config.solve_duplications:
                            print("\n**** Solving duplicated identities ****")
                            solve_duplications(blobs,group_size)


                        duplications_removal_time = time.time() - start

                        #############################################################
                        ###################     Accuracies     ######################
                        ####
                        #############################################################
                        print("\n**** Accuracies ****")
                        number_of_possible_assignations = 0
                        number_of_correct_assignations = [0] * group_size
                        number_assignations = [0]*group_size
                        number_of_identity_repetitions = 0
                        number_of_frames_with_repetitions = 0
                        number_of_identity_shifts_in_accumulated_frames = 0
                        number_of_blobs_assigned_in_accumulation = 0
                        number_of_not_assigned_blobs = [0] * group_size
                        individual_fragments_badly_assigned_in_accumulation = []
                        individual_fragments_that_are_repetitions = []
                        individual_fragments = []
                        for frame_number, blobs_in_frame in enumerate(blobs):
                            identities_in_frame = []
                            frame_with_repetition = False
                            for i, blob in enumerate(blobs_in_frame):
                                if blob._fragment_identifier not in individual_fragments:
                                    individual_fragments.append(blob._fragment_identifier)
                                if blob.is_an_individual_in_a_fragment:
                                    number_of_possible_assignations += 1
                                    if blob._assigned_during_accumulation:
                                        number_of_blobs_assigned_in_accumulation += 1
                                    if blob.identity is not None and blob.identity != 0:
                                        number_assignations[blob.user_generated_identity-1] += 1
                                        if blob.identity == blob.user_generated_identity:
                                            number_of_correct_assignations[blob.user_generated_identity-1] += 1
                                        elif blob._assigned_during_accumulation:
                                            if blob._fragment_identifier not in individual_fragments_badly_assigned_in_accumulation:
                                                individual_fragments_badly_assigned_in_accumulation.append(blob._fragment_identifier)
                                            number_of_identity_shifts_in_accumulated_frames += 1
                                        if blob.identity in identities_in_frame:
                                            print([blob.identity for blob in blobs_in_frame])
                                            raise ValueError("Duplication after removing duplications")
                                            number_of_identity_repetitions += 1
                                            if blob._fragment_identifier not in individual_fragments_that_are_repetitions:
                                                individual_fragments_that_are_repetitions.append(blob._fragment_identifier)
                                            frame_with_repetition = True

                                        identities_in_frame.append(blob.identity)
                                    elif blob.identity is None or blob.identity == 0:
                                        number_of_not_assigned_blobs[i] += 1
                            if frame_with_repetition:
                                number_of_frames_with_repetitions += 1

                        number_of_acceptable_fragments = sum([global_fragment._acceptable_for_training for global_fragment in global_fragments])
                        number_of_unique_fragments = sum([global_fragment.is_unique for global_fragment in global_fragments])
                        number_of_certain_fragments = sum([global_fragment._is_certain for global_fragment in global_fragments])

                        individual_accuracies = np.asarray(number_of_correct_assignations)/np.asarray(number_assignations)
                        accuracy = np.sum(number_of_correct_assignations)/np.sum(number_assignations)
                        accuracy_overall = np.sum(number_of_correct_assignations)/number_of_possible_assignations
                        print("\n\ngroup_size: ", group_size)
                        print("individual_accuracies(assigned): ", individual_accuracies)
                        print("accuracy(assigned): ", accuracy)
                        print("accuracy: ", accuracy_overall)
                        print("\nnumber of global fragments: ", len(global_fragments))
                        print("number of accumulated fragments:", sum([global_fragment.used_for_training for global_fragment in global_fragments]))
                        print("number of candidate global fragments:", len(candidates_next_global_fragments))
                        print("number of acceptable fragments: ", number_of_acceptable_fragments)
                        print("number of unique fragments: ", number_of_unique_fragments + 1)
                        print("number of certain fragments: ", number_of_certain_fragments + 1)
                        print("\nnumber of individual fragments: ", len(individual_fragments))
                        print("number of individual fragments badly assigned in acumulation: ", len(individual_fragments_badly_assigned_in_accumulation))
                        print("number of fragments that are repetitions: ", len(individual_fragments_that_are_repetitions))
                        print("\nframes in video: ", frames_in_video)
                        print("number of frames with repetitions: ", number_of_frames_with_repetitions)
                        print("number of assignation: ", number_assignations)
                        print("number correct assignations: ", number_of_correct_assignations)
                        print("number of identity repetitions: ", number_of_identity_repetitions)
                        print("number of identity shifts in accumulated frames: ", number_of_identity_shifts_in_accumulated_frames)
                        print("****************************************************************************************************\n\n")

                        #############################################################
                        ###################  Update data-frame   ####################
                        #############################################################
                        results_data_frame = results_data_frame.append({'date': time.strftime("%c"),
                                                                        'cluster': int(job_config.cluster) ,
                                                                        'test_name': job_config.test_name,
                                                                        'CNN_model': job_config.CNN_model,
                                                                        'knowledge_transfer_flag': job_config.knowledge_transfer_flag,
                                                                        'knowledge_transfer_folder': job_config.knowledge_transfer_folder,
                                                                        'pretraining_flag': job_config.pretraining_flag,
                                                                        'percentage_of_frames_in_pretaining': job_config.percentage_of_frames_in_pretaining,
                                                                        'only_accumulate_one_fragment': job_config.only_accumulate_one_fragment,
                                                                        'train_filters_in_accumulation': bool(job_config.train_filters_in_accumulation),
                                                                        'accumulation_certainty': job_config.accumulation_certainty,
                                                                        'solve_duplications': job_config.solve_duplications,
                                                                        'preprocessing_type': job_config.preprocessing_type,
                                                                        'IMDB_codes': job_config.IMDB_codes,
                                                                        'ids_codes': job_config.ids_codes,
                                                                        'group_size': int(group_size),
                                                                        'frames_in_video': int(frames_in_video),
                                                                        'frames_per_fragment': mean_frames_per_fragment,
                                                                        'repetition': int(repetition),
                                                                        'individual_accuracies': individual_accuracies,
                                                                        'accuracy': accuracy,
                                                                        'number_of_fragments': len(global_fragments),
                                                                        'number_of_candidate_fragments': len(candidates_next_global_fragments),
                                                                        'number_of_unique_fragments': number_of_unique_fragments,
                                                                        'number_of_certain_fragments': number_of_certain_fragments,
                                                                        'number_of_acceptable_fragments': number_of_acceptable_fragments,
                                                                        'number_of_frames_with_repetitions_after_assignation': int(number_of_frames_with_repetitions),
                                                                        'number_of_blobs_assigned_in_accumulation': number_of_blobs_assigned_in_accumulation,
                                                                        'number_of_not_assigned_blobs': number_of_not_assigned_blobs,
                                                                        'number_of_individual_fragments':len(individual_fragments),
                                                                        'number_of_missassigned_individual_fragments_in_accumulation':len(individual_fragments_badly_assigned_in_accumulation),
                                                                        'number_of_individual_fragments_that_are_repetitions':individual_fragments_that_are_repetitions,
                                                                        'number_of_assignation': number_assignations,
                                                                        'number_of_correct_assignations': number_of_correct_assignations,
                                                                        'number_of_identity_repetitions': number_of_identity_repetitions,
                                                                        'number_of_identity_shifts_in_accumulated_frames': number_of_identity_shifts_in_accumulated_frames,
                                                                        'pretraining_time': pretraining_time,
                                                                        'accumulation_time': accumulation_time,
                                                                        'assignation_time': assignation_time,
                                                                        'duplications_removal_time': duplications_removal_time,
                                                                        'total_time': pretraining_time + accumulation_time + assignation_time + duplications_removal_time,
                                                                         }, ignore_index=True)


                        results_data_frame.to_pickle('./library/results_data_frame.pkl')

                        blobs = None
                        global_fragments = None
