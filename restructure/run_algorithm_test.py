from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys
sys.setrecursionlimit(100000)
import glob
import numpy as np
import pandas as pd
import cPickle as pickle
import time

# Import third party libraries
import cv2
from pprint import pprint

# Import application/library specifics
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./library')
# sys.path.append('IdTrackerDeep/tracker')

from video import Video
from blob import reset_blobs_fragmentation_parameters, compute_fragment_identifier_and_blob_index, connect_blob_list, apply_model_area_to_video, ListOfBlobs, get_images_from_blobs_in_video
from globalfragment import give_me_list_of_global_fragments, ModelArea, give_me_pre_training_global_fragments
from globalfragment import get_images_and_labels_from_global_fragments
from globalfragment import subsample_images_for_last_training, order_global_fragments_by_distance_travelled
from segmentation import segment
from GUI_utils import selectFile, getInput, selectOptions, ROISelectorPreview, selectPreprocParams, fragmentation_inspector, frame_by_frame_identity_inspector
from py_utils import getExistentFiles
from video_utils import checkBkg
from pre_trainer import pre_train
from accumulation_manager import AccumulationManager
from network_params import NetworkParams
from trainer import train
from assigner import assign, assign_identity_to_blobs_in_video, compute_P1_for_blobs_in_video, assign_identity_to_blobs_in_video_by_fragment
from visualize_embeddings import visualize_embeddings_global_fragments
from id_CNN import ConvNetwork

from library_utils import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs, LibraryJobConfig, check_if_repetition_has_been_computed

NUM_CHUNKS_BLOB_SAVING = 50 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
NUMBER_OF_SAMPLES = 30000
RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = 0.1 # threshold to select a individual fragment as eligible for training
###
np.random.seed(0)
###

if __name__ == '__main__':
    '''
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: path to test_data_frame.pkl
    argv[3]: test_number

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

    dataset = Dataset(IMDB_codes = job_config.IMDB_codes, ids_codes = job_config.ids_codes)
    dataset.loadIMDBs()
    print(dataset.images.shape)

    for group_size in job_config.group_sizes:

        for frames_in_video in job_config.frames_in_video:

            for i, frames_in_fragment in enumerate(job_config.frames_per_individual_fragment):
                for repetition in job_config.repetitions:
                    frames_in_fragment_path = os.path.join(job_config.condition_path,'group_size_' + str(group_size),
                                                            'num_frames_' + str(frames_in_video),
                                                            'frames_in_fragment_' + str(frames_in_fragment),
                                                            'repetition_' + str(repetition))


                    print("\n********** group size %i - frames_in_video %i - frames_in_fragment %i - repetition %i ********" %(group_size,frames_in_video,frames_in_fragment,repetition))
                    already_computed = False
                    if os.path.isfile('./library/results_data_frame.pkl'):
                        already_computed = check_if_repetition_has_been_computed(results_data_frame, job_config, group_size, frames_in_video, frames_in_fragment, repetition)
                        print("already_computed flag: ", already_computed)
                    if already_computed:
                        print("The algorithm with this comditions has been already tested")
                    else:

                        video = Video() #instantiate object video
                        video.video_path = os.path.join(frames_in_fragment_path,'fake_0.avi') #set path
                        video.create_session_folder()
                        video._animal_type = 'fish' #string: type of animals to be tracked in the video
                        video._number_of_animals = group_size #int: number of animals in the video
                        video._maximum_number_of_blobs = group_size #int: the maximum number of blobs detected in the video
                        video._num_frames = frames_in_video
                        video.tracking_with_knowledge_transfer = job_config.knowledge_transfer_flag
                        video.knowledge_transfer_model_folder = job_config.knowledge_transfer_folder
                        video.portrait_size = (32, 32, 1) #NOTE: this can change if the library changes. BUILD next library with new preprocessing.

                        #############################################################
                        ####################   Preprocessing   ######################
                        #### prepare blobs list and global fragments from the    ####
                        #### library                                             ####
                        #############################################################

                        config = BlobsListConfig(number_of_animals = group_size,
                                                number_of_frames_per_fragment = frames_in_fragment,
                                                std_number_of_frames_per_fragment = job_config.std_frames_per_individual_fragment,
                                                number_of_frames = frames_in_video,
                                                repetition = repetition)
                        portraits, centroids = subsample_dataset_by_individuals(dataset, config)
                        blobs = generate_list_of_blobs(portraits, centroids, config)
                        compute_fragment_identifier_and_blob_index(blobs, config.number_of_animals)
                        global_fragments = give_me_list_of_global_fragments(blobs, config.number_of_animals)
                        global_fragments_ordered = order_global_fragments_by_distance_travelled(global_fragments)
                        video._has_been_segmented = True
                        video._has_been_preprocessed = True

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
                            print("We will restore the network from pretraining: %s\n" %video._pretraining_folder)
                            accumulation_network_params.restore_folder = video._pretraining_folder
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
                        images = get_images_from_blobs_in_video(blobs)#, video._episodes_start_end)
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
                            #         if blob.identity == 0 and blob.is_a_fish:
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
                        number_correct_assignations = [0] * group_size
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
                                if blob.is_a_fish_in_a_fragment:
                                    number_assignations[i] += 1
                                    if blob._assigned_during_accumulation:
                                        number_of_blobs_assigned_in_accumulation += 1
                                    if blob.identity is not None and blob.identity != 0:
                                        if blob.identity == blob.user_generated_identity:
                                            number_correct_assignations[i] += 1
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

                        individual_accuracies = np.asarray(number_correct_assignations)/np.asarray(number_assignations)
                        accuracy = np.sum(number_correct_assignations)/np.sum(number_assignations)
                        print("\n\ngroup_size: ", group_size)
                        print("individual_accuracies: ", individual_accuracies)
                        print("accuracy: ", accuracy)
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
                                                                        'animal_type': job_config.animal_type,
                                                                        'IMDB_codes': job_config.IMDB_codes,
                                                                        'ids_codes': job_config.ids_codes,
                                                                        'group_size': int(group_size),
                                                                        'frames_in_video': int(frames_in_video),
                                                                        'frames_per_fragment': int(frames_in_fragment),
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
                                                                        'total_time': pretraining_time + accumulation_time + assignation_time,
                                                                         }, ignore_index=True)


                        results_data_frame.to_pickle('./library/results_data_frame.pkl')

                        blobs = None
                        global_fragments = None
