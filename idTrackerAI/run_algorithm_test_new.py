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
from correct_duplications import solve_duplications, mark_fragments_as_duplications, check_for_duplications_last_pass
from compute_groundtruth_statistics import get_accuracy_wrt_groundtruth
from generate_groundtruth import GroundTruth, generate_groundtruth

from library_utils import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs, LibraryJobConfig, check_if_repetition_has_been_computed

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

def plot_fragments_generated(fragments):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1,1)
    from py_utils import get_spaced_colors_util
    colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=False)
    for fragment in fragments:
        if fragment.is_an_individual:
            blob_index = fragment.blob_hierarchy_in_starting_frame
            (start, end) = fragment.start_end
            ax.add_patch(
                patches.Rectangle(
                    (start, blob_index - 0.5),   # (x,y)
                    end - start - 1,  # width
                    1.,          # height
                    fill=True,
                    edgecolor=None,
                    facecolor=colors[blob_index],
                )
            )
    plt.show()

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

                for scale_parameter in job_config.scale_parameter:

                    for shape_parameter in job_config.shape_parameter:

                        repetition_path = os.path.join(job_config.condition_path,'group_size_' + str(group_size),
                                                                'num_frames_' + str(frames_in_video),
                                                                'scale_parameter_' + str(scale_parameter),
                                                                'shape_parameter_' + str(shape_parameter),
                                                                'repetition_' + str(repetition))


                        print("\n********** group size %i - frames_in_video %i - scale_parameter %s -  shape_parameter %s - repetition %i ********"
                                %(group_size,frames_in_video,str(scale_parameter), str(shape_parameter), repetition))
                        mean_number_of_frames_per_fragment = shape_parameter * scale_parameter
                        sigma_number_of_frames_per_fragment = np.sqrt(shape_parameter * scale_parameter ** 2)
                        print("mean_number_of_frames_per_fragment %.2f" %mean_number_of_frames_per_fragment)
                        print("sigma_number_of_frames_per_fragment %.2f" %sigma_number_of_frames_per_fragment)
                        already_computed = False
                        if os.path.isfile('./library/results_data_frame.pkl'):
                            already_computed = check_if_repetition_has_been_computed(results_data_frame,
                                                                                    job_config, group_size,
                                                                                    frames_in_video,
                                                                                    scale_parameter,
                                                                                    shape_parameter,
                                                                                    repetition)
                            print("already_computed flag: ", already_computed)
                        if already_computed:
                            print("The algorithm with this comditions has been already tested")
                        else:

                            video = Video() #instantiate object video
                            video.video_path = os.path.join(repetition_path,'_fake_0.avi') #set path
                            video.create_session_folder()
                            logger = setup_logging(path_to_save_logs = video.session_folder, video_object = video)
                            logger.info("Starting working on session %s" %video.session_folder)
                            logger.info("Log files saved in %s" %video.logs_folder)
                            video._number_of_animals = group_size #int: number of animals in the video
                            video._maximum_number_of_blobs = group_size #int: the maximum number of blobs detected in the video
                            video._number_of_frames = frames_in_video
                            video.tracking_with_knowledge_transfer = job_config.knowledge_transfer_flag
                            video.knowledge_transfer_model_folder = job_config.knowledge_transfer_folder
                            video._identification_image_size = (imsize, imsize, 1) #NOTE: this can change if the library changes. BUILD next library with new preprocessing.

                            #############################################################
                            ####################   Preprocessing   ######################
                            #### prepare blobs list and global fragments from the    ####
                            #### library                                             ####
                            #############################################################
                            video.create_preprocessing_folder()
                            list_of_blobs_config = BlobsListConfig(number_of_animals = group_size,
                                                    scale_parameter = scale_parameter,
                                                    shape_parameter = shape_parameter,
                                                    number_of_frames = frames_in_video,
                                                    repetition = repetition)
                            identification_images, centroids = subsample_dataset_by_individuals(dataset, list_of_blobs_config)
                            blobs = generate_list_of_blobs(identification_images, centroids, list_of_blobs_config)
                            video._has_been_segmented = True
                            list_of_blobs = ListOfBlobs(blobs_in_video = blobs)
                            logger.info("Computing maximum number of blobs detected in the video")
                            list_of_blobs.check_maximal_number_of_blob(video.number_of_animals)
                            logger.info("Computing a model of the area of the individuals")
                            list_of_blobs.video = video
                            #assign an identifier to each blob belonging to an individual fragment
                            list_of_blobs.compute_fragment_identifier_and_blob_index(video.number_of_animals)
                            #assign an identifier to each blob belonging to a crossing fragment
                            list_of_blobs.compute_crossing_fragment_identifier()
                            #create list of fragments
                            fragments = create_list_of_fragments(list_of_blobs.blobs_in_video,
                                                                video.number_of_animals)
                            # plot_fragments_generated(fragments)
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
                                                                                                list_of_fragments = list_of_fragments,
                                                                                                list_of_global_fragments = list_of_global_fragments,
                                                                                                save = True, plot = False)
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
                            video._has_been_preprocessed = True
                            logger.info("Blobs detection and fragmentation finished succesfully.")

                            #############################################################
                            ##################   Protocols cascade   ####################
                            #############################################################
                            video.protocol = 1
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
                                                        save_folder = video._accumulation_folder,
                                                        image_size = video.identification_image_size)

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
                            #restore variables from the pretraining
                            net.restore()
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
                            video._first_frame_first_global_fragment.append(list_of_global_fragments.set_first_global_fragment_for_accumulation(video, accumulation_trial = 0))
                            list_of_global_fragments.video = video
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
                                                                video.knowledge_transfer_from_same_animals)
                            logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
                            video._first_accumulation_finished = True
                            ### NOTE: save all the accumulation statistics
                            video.save()
                            logger.info("Saving fragments")
                            # list_of_fragments.save(video.fragments_path)
                            list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
                            video.first_accumulation_time = time.time() - video.first_accumulation_time
                            list_of_fragments.save_light_list(video._accumulation_folder)
                            if video.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                                if isinstance(video.first_frame_first_global_fragment, list):
                                    video.protocol = 1 if video.accumulation_step <= 1 else 2
                                    video._first_frame_first_global_fragment = video.first_frame_first_global_fragment[video.accumulation_trial]
                                    list_of_global_fragments.video = video
                                video.assignment_time = time.time()
                                #### Assigner ####
                                print('\nAssignment ---------------------------------------------------------')
                                assigner(list_of_fragments, video, net)
                                video._has_been_assigned = True
                                ### NOTE: save all the assigner statistics
                                video.assignment_time = time.time() - video.assignment_time
                                video.pretraining_time = 0
                                video.second_accumulation_time = 0
                                video.save()
                            else:
                                video.protocol = 3
                                percentage_of_accumulated_images = [video._ratio_accumulated_images]
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

                                #### Pre-trainer ####
                                list_of_fragments.reset(roll_back_to = 'fragmentation')
                                list_of_global_fragments.order_by_distance_travelled()
                                pre_trainer(None, video, list_of_fragments, list_of_global_fragments, pretrain_network_params)
                                logger.info("Pretraining ended")
                                #save changes
                                logger.info("Saving changes in video object")
                                video._has_been_pretrained = True
                                video.save()
                                ### NOTE: save pre-training statistics
                                video.pretraining_time = time.time() - video.pretraining_time
                                #### Accumulation ####
                                #Last accumulation after pretraining
                                video.second_accumulation_time = time.time()

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
                                    video._first_frame_first_global_fragment.append(list_of_global_fragments.set_first_global_fragment_for_accumulation(video, accumulation_trial = i - 1))
                                    list_of_global_fragments.video = video
                                    list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(video, accumulation_trial = i - 1)
                                    accumulation_manager = AccumulationManager(video,
                                                                                list_of_fragments, list_of_global_fragments,
                                                                                threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
                                    #set global epoch counter to 0
                                    logger.info("Start accumulation")
                                    global_step = 0
                                    video._ratio_accumulated_images = accumulate(accumulation_manager,
                                                                                video,
                                                                                global_step,
                                                                                net,
                                                                                video.knowledge_transfer_from_same_animals)
                                    logger.info("Accumulation finished. There are no more acceptable global_fragments for training")
                                    percentage_of_accumulated_images.append(video.ratio_accumulated_images)
                                    list_of_fragments.save_light_list(video._accumulation_folder)
                                    if video.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                                        break
                                    else:

                                        logger.info("This accumulation was not satisfactory. Try to start from a different global fragment")


                                video.accumulation_trial = np.argmax(percentage_of_accumulated_images)
                                video._first_frame_first_global_fragment = video.first_frame_first_global_fragment[video.accumulation_trial]
                                video._ratio_accumulated_images = percentage_of_accumulated_images[video.accumulation_trial]
                                list_of_global_fragments.video = video
                                accumulation_folder_name = 'accumulation_' + str(video.accumulation_trial)
                                video._accumulation_folder = os.path.join(video.session_folder, accumulation_folder_name)
                                list_of_fragments.load_light_list(video._accumulation_folder)
                                video._second_accumulation_finished = True
                                logger.info("Saving global fragments")
                                # list_of_fragments.save(video.fragments_path)
                                list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
                                ### NOTE: save second_accumulation statistics
                                video.save()
                                video.second_accumulation_time = time.time() - video.second_accumulation_time
                                video.assignment_time = time.time()
                                #### Assigner ####
                                print('\n---------------------------------------------------------')
                                assigner(list_of_fragments, video, net)
                                video._has_been_assigned = True
                                ### NOTE: save all the assigner statistics
                                video.assignment_time = time.time() - video.assignment_time
                                video.save()

                            # finish and save
                            logger.debug("Saving list of fragments, list of global fragments and video object")
                            # list_of_fragments.save(video.fragments_path)
                            list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
                            video.save()

                            #############################################################
                            ##############   Update list of blobs   #####################
                            ####
                            #############################################################
                            list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments, video.fragment_identifier_to_index)

                            #############################################################
                            ############   Generate generate_groundtruth_file ###########
                            ####
                            #############################################################
                            groundtruth = generate_groundtruth(video, list_of_blobs.blobs_in_video, start = 0, end = video.number_of_frames-1)

                            #############################################################
                            ##########  Accuracies before solving duplications ##########
                            ####
                            #############################################################
                            blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
                            blobs_in_video = list_of_blobs.blobs_in_video[groundtruth.start:groundtruth.end]

                            print("computting groundtrugh")
                            accuracies, _ = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth,
                                                                            blobs_in_video,
                                                                            video.first_frame_first_global_fragment)

                            if accuracies is not None:
                                print("saving accuracies in video")
                                video.gt_start_end = (groundtruth.start,groundtruth.end)
                                video.gt_accuracy_before_duplications = accuracies

                            #############################################################
                            ###################   Solve duplications      ###############
                            ####
                            #############################################################
                            video.solve_duplications_time = time.time()
                            logger.info("Start checking for and solving duplications")
                            list_of_fragments.reset(roll_back_to = 'assignment')
                            # mark fragments as duplications
                            mark_fragments_as_duplications(list_of_fragments.fragments)
                            # solve duplications
                            solve_duplications(list_of_fragments, video.first_frame_first_global_fragment)
                            video._has_duplications_solved = True
                            logger.info("Done")
                            # finish and save
                            logger.info("Saving")
                            # list_of_blobs = ListOfBlobs(video, blobs_in_video = blobs)
                            # list_of_blobs.save(video.blobs_path, NUM_CHUNKS_BLOB_SAVING)
                            logger.info("Done")
                            video.solve_duplications_time = time.time() - video.solve_duplications_time

                            #############################################################
                            ##############   Invididual fragments stats #################
                            ####
                            #############################################################
                            video.individual_fragments_stats = list_of_fragments.get_stats(list_of_global_fragments)
                            video.compute_overall_P2(list_of_fragments.fragments)
                            print("individual overall_P2 ", video.individual_P2)
                            print("overall_P2 ", video.overall_P2)
                            # list_of_fragments.plot_stats(video)
                            list_of_fragments.save_light_list(video._accumulation_folder)

                            #############################################################
                            ##############   Update list of blobs   #####################
                            ####
                            #############################################################
                            list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments, video.fragment_identifier_to_index)
                            list_of_blobs.save(video, video.blobs_path, number_of_chunks = video.number_of_frames)

                            #############################################################
                            ##########  Accuracies after solving duplications ##########
                            ####
                            #############################################################

                            blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
                            blobs_in_video = list_of_blobs.blobs_in_video[groundtruth.start:groundtruth.end]

                            print("computting groundtrugh")
                            accuracies, _ = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth,
                                                                            blobs_in_video,
                                                                            video.first_frame_first_global_fragment)

                            if accuracies is not None:
                                print("saving accuracies in video")
                                video.gt_accuracy = accuracies
                                video.save()

                            video.total_time = sum([video.solve_duplications_time,
                                                    video.assignment_time,
                                                    video.pretraining_time,
                                                    video.second_accumulation_time,
                                                    video.assignment_time,
                                                    video.first_accumulation_time])

                            #############################################################
                            ###################  Update data-frame   ####################
                            #############################################################
                            results_data_frame = \
                                results_data_frame.append({'date': time.strftime("%c"),
                                    'cluster': int(job_config.cluster) ,
                                    'test_name': job_config.test_name,
                                    'CNN_model': job_config.CNN_model,
                                    'knowledge_transfer_flag': job_config.knowledge_transfer_flag,
                                    'knowledge_transfer_folder': job_config.knowledge_transfer_folder,
                                    'IMDB_codes': job_config.IMDB_codes,
                                    'ids_codes': job_config.ids_codes,
                                    'group_size': int(group_size),
                                    'frames_in_video': int(frames_in_video),
                                    'scale_parameter': scale_parameter,
                                    'shape_parameter': shape_parameter,
                                    'mean_number_of_frames_per_fragment': mean_number_of_frames_per_fragment,
                                    'sigma_number_of_frames_per_fragment': sigma_number_of_frames_per_fragment,
                                    'repetition': int(repetition),
                                    'protocol': video.protocol,
                                    'overall_P2': video.overall_P2,
                                    'individual_accuracy_before_duplications': video.gt_accuracy_before_duplications['individual_accuracy'],
                                    'accuracy_before_duplications': video.gt_accuracy_before_duplications['accuracy'],
                                    'individual_accuracy_assigned_before_duplications': video.gt_accuracy_before_duplications['individual_accuracy_assigned'],
                                    'accuracy_assigned_before_duplications': video.gt_accuracy_before_duplications['accuracy_assigned'],
                                    'individual_accuracy_after_accumulation_before_duplications': video.gt_accuracy_before_duplications['individual_accuracy_after_accumulation'],
                                    'accuracy_after_accumulation_before_duplications': video.gt_accuracy_before_duplications['accuracy_after_accumulation'],
                                    'individual_accuracy': video.gt_accuracy['individual_accuracy'],
                                    'accuracy': video.gt_accuracy['accuracy'],
                                    'individual_accuracy_assigned': video.gt_accuracy['individual_accuracy_assigned'],
                                    'accuracy_assigned': video.gt_accuracy['accuracy_assigned'],
                                    'individual_accuracy_in_accumulation': video.gt_accuracy['individual_accuracy_in_accumulation'],
                                    'accuracy_in_accumulation': video.gt_accuracy['accuracy_in_accumulation'],
                                    'individual_accuracy_after_accumulation': video.gt_accuracy['individual_accuracy_after_accumulation'],
                                    'accuracy_after_accumulation': video.gt_accuracy['accuracy_after_accumulation'],
                                    'crossing_detector_accuracy': video.gt_accuracy['crossing_detector_accuracy'],
                                    'individual_fragments_lengths': video.individual_fragments_lenghts,
                                    'number_of_global_fragments': list_of_global_fragments.number_of_global_fragments,
                                    'number_of_fragments': video.individual_fragments_stats['number_of_fragments'],
                                    'number_of_crossing_fragments': video.individual_fragments_stats['number_of_crossing_fragments'],
                                    'number_of_individual_fragments': video.individual_fragments_stats['number_of_individual_fragments'],
                                    'number_of_individual_fragments_not_in_a_global_fragment': video.individual_fragments_stats['number_of_individual_fragments_not_in_a_global_fragment'],
                                    'number_of_not_accumulable_individual_fragments': video.individual_fragments_stats['number_of_not_accumulable_individual_fragments'],
                                    'number_of_globally_accumulated_individual_blobs': video.individual_fragments_stats['number_of_globally_accumulated_individual_blobs'],
                                    'number_of_partially_accumulated_individual_fragments': video.individual_fragments_stats['number_of_partially_accumulated_individual_fragments'],
                                    'number_of_blobs': video.individual_fragments_stats['number_of_blobs'],
                                    'number_of_crossing_blobs': video.individual_fragments_stats['number_of_crossing_blobs'],
                                    'number_of_individual_blobs': video.individual_fragments_stats['number_of_individual_blobs'],
                                    'number_of_individual_blobs_not_in_a_global_fragment': video.individual_fragments_stats['number_of_individual_blobs_not_in_a_global_fragment'],
                                    'number_of_not_accumulable_individual_blobs': video.individual_fragments_stats['number_of_not_accumulable_individual_blobs'],
                                    'number_of_globally_accumulated_individual_fragments': video.individual_fragments_stats['number_of_globally_accumulated_individual_fragments'],
                                    'number_of_partially_accumulated_individual_blobs': video.individual_fragments_stats['number_of_partially_accumulated_individual_blobs'],
                                    'first_accumulation_time': video.first_accumulation_time,
                                    'pretraining_time': video.pretraining_time,
                                    'second_accumulation_time': video.second_accumulation_time,
                                    'assignment_time': video.assignment_time,
                                    'solve_duplications_time': video.solve_duplications_time,
                                    'total_time': video.total_time
                                     }, ignore_index=True)

                                                                            #  'number_of_frames_with_repetitions_after_assignation': int(number_of_frames_with_repetitions),
                                                                            #  'number_of_not_assigned_blobs': [blob.assigned_identity == 0 for blobs_in_frame in list_of_blobs.blobs in video for blob in blobs_in_frame],
                                                                            #  'number_of_identity_shifts_in_accumulated_frames': number_of_identity_shifts_in_accumulated_frames,
                                                                            #  'number_of_missassigned_individual_fragments_in_accumulation': ,
                                                                            #  'number_of_individual_fragments_that_are_repetitions':individual_fragments_that_are_repetitions,

                            results_data_frame.to_pickle('./library/results_data_frame.pkl')
                            blobs_in_video = None
                            blobs_in_video_groundtruth = None
                            groundtruth = None
                            list_of_blobs = None
                            list_of_fragments = None
                            list_of_global_fragments = None
                            video = None
                            accuracies = None
                            percentage_of_accumulated_images = None
