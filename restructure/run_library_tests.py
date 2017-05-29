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
from globalfragment import compute_model_area, give_me_list_of_global_fragments, ModelArea, give_me_pre_training_global_fragments
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

from blobs_list_generator import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs

NUM_CHUNKS_BLOB_SAVING = 50 #it is necessary to split the list of connected blobs to prevent stack overflow (or change sys recursionlimit)
NUMBER_OF_SAMPLES = 30000
RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = 0.1 # threshold to select a individual fragment as eligible for training
###
np.random.seed(0)
###

class LibraryJobConfig(object):
    def __init__(self,cluster = None,
                        job_number = None,
                        condition = None,
                        train_filters = False,
                        percentage_of_fragments_in_pretraining = None,
                        accumulation_certainty = None,
                        IMDB_codes = None,
                        ids_codes = None,
                        repetitions = None):
        self.cluster = int(cluster)
        self.job_number = int(job_number)
        self.condition = condition
        self.IMDB_codes = IMDB_codes
        self.ids_codes = ids_codes
        self.group_sizes = [2, 5, 10, 20]
        self.frames_in_video = [1000]
        self.frames_per_individual_fragment = [50, 100, 250]
        self.repetitions = map(int,repetitions.split('_'))
        if 'Pretraining' in condition:
            self.pretraining_flag = True
            self.knowledge_transfer_flag = False
        elif 'KnowledgeTransfer' in condition:
            self.pretraining_flag = False
            self.knowledge_transfer_flag = True
        self.train_filters = int(train_filters)
        self.percentage_of_fragments_in_pretraining = float(percentage_of_fragments_in_pretraining)
        self.accumulation_certainty = float(accumulation_certainty)

    def create_folders_structure(self):
        #create main condition folder
        self.condition_path = os.path.join('./library','library_test_' + self.condition)
        if not os.path.exists(self.condition_path):
            os.makedirs(self.condition_path)
        #create subfolders for group sizes
        for group_size in self.group_sizes:
            group_size_path = os.path.join(self.condition_path,'group_size_' + str(group_size))
            if not os.path.exists(group_size_path):
                os.makedirs(group_size_path)
            #create subfolders for frames_in_video
            for num_frames in self.frames_in_video:
                num_frames_path = os.path.join(group_size_path,'num_frames_' + str(num_frames))
                if not os.path.exists(num_frames_path):
                    os.makedirs(num_frames_path)
                #create subfolders for frames_in_fragment
                for frames_in_fragment in self.frames_per_individual_fragment:
                    frames_in_fragment_path = os.path.join(num_frames_path, 'frames_in_fragment_' + str(frames_in_fragment))
                    if not os.path.exists(frames_in_fragment_path):
                        os.makedirs(frames_in_fragment_path)
                    for repetition in self.repetitions:
                        repetition_path = os.path.join(frames_in_fragment_path, 'repetition_' + str(repetition))
                        if not os.path.exists(repetition_path):
                            os.makedirs(repetition_path)

if __name__ == '__main__':
    '''
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: job number
    argv[3]: condition: 'P'-pretraining, 'KT'-knowledge transfer
    argv[4]: train_filters:
    argv[5]: percentage_of_fragments_in_pretraining:
    argv[6]: accumulation_certainty:
    argv[7]: IMDB_codes (A, B, C, D, E, F)
    argv[8]: ids_codes: a - all, f - first half, s - second half
    argv[9]: repetitions

    e.g.
    run_library_tests.py 1 1 P 0 .5 .1 DEF afs 1_2 (running in the cluster, job1, pretraining, libraries DEF, all individuals in library D and first half obf E second half of F, repetitions[1 2])
    '''
    print("cluster:", sys.argv[1])
    print("job_number:", sys.argv[2])
    print("condition:", sys.argv[3])
    print("train_filters:", sys.argv[4])
    print("percentage_of_fragments_in_pretraining:", sys.argv[5])
    print("accumulation_certainty:", sys.argv[6])
    print("IMDB_codes:", sys.argv[7])
    print("ids_codes:", sys.argv[8])
    print("repetitions:", sys.argv[9])
    job_config = LibraryJobConfig(cluster = sys.argv[1],
                                    job_number = sys.argv[2],
                                    condition = sys.argv[3],
                                    train_filters = sys.argv[4],
                                    percentage_of_fragments_in_pretraining = sys.argv[5],
                                    accumulation_certainty = sys.argv[6],
                                    IMDB_codes = sys.argv[7],
                                    ids_codes = sys.argv[8],
                                    repetitions = sys.argv[9])

    print("cluster:", job_config.cluster)
    print("job_number:", job_config.job_number)
    print("condition:", job_config.condition)
    print("train_filters:", job_config.train_filters)
    print("percentage_of_fragments_in_pretraining:", job_config.percentage_of_fragments_in_pretraining)
    print("accumulation_certainty:", job_config.accumulation_certainty)
    print("IMDB_codes:", job_config.IMDB_codes)
    print("ids_codes:", job_config.ids_codes)
    print("repetitions:", job_config.repetitions)

    job_config.create_folders_structure()
    dataset = Dataset(IMDB_codes = job_config.IMDB_codes, ids_codes = job_config.ids_codes)
    dataset.loadIMDBs()
    results_data_frame = pd.DataFrame()

    for group_size in job_config.group_sizes:

        for num_frames in job_config.frames_in_video:

            for frames_in_fragment in job_config.frames_per_individual_fragment:

                for repetition in job_config.repetitions:
                    frames_in_fragment_path = os.path.join(
                                                os.path.join(
                                                os.path.join(
                                                os.path.join(
                                                job_config.condition_path,'group_size_' + str(group_size)),
                                                'num_frames_' + str(num_frames)),
                                                'frames_in_fragment_' + str(frames_in_fragment)),
                                                'repetition_' + str(repetition))

                    print("\n********** group size %i - num_frames %i - frames_in_fragment %i - repetition %i ********" %(group_size,num_frames,frames_in_fragment,repetition))

                    video = Video() #instantiate object video
                    video.video_path = os.path.join(frames_in_fragment_path,'fake_0.avi') #set path
                    video.create_session_folder()
                    video._animal_type = 'fish' #string: type of animals to be tracked in the video
                    video._number_of_animals = group_size #int: number of animals in the video
                    video._maximum_number_of_blobs = group_size #int: the maximum number of blobs detected in the video
                    video._num_frames = num_frames

                    #############################################################
                    ####################   Preprocessing   ######################
                    #### prepare blobs list and global fragments from the    ####
                    #### library                                             ####
                    #############################################################

                    config = BlobsListConfig(number_of_animals = group_size, number_of_frames_per_fragment = frames_in_fragment, number_of_frames = num_frames, repetition = repetition)
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
                        if job_config.percentage_of_fragments_in_pretraining != 1.:
                            number_of_pretraining_global_fragments = int(len(global_fragments)*job_config.percentage_of_fragments_in_pretraining)
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
                                                                save_folder = video._pretraining_folder)
                        #start pretraining
                        net = pre_train(video, blobs,
                                        pretraining_global_fragments,
                                        pretrain_network_params,
                                        store_accuracy_and_error = False,
                                        check_for_loss_plateau = True,
                                        save_summaries = False,
                                        print_flag = False,
                                        plot_flag = False)

                    elif job_config.knowledge_transfer_flag:
                        print("Loading a net for KT needs to be implemented")
                    pretraining_time = time.time()
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
                                                restore_folder = video._pretraining_folder)
                    if job_config.train_filters == True:
                        accumulation_network_params.scopes_layers_to_optimize = []
                    #instantiate network object
                    net = ConvNetwork(accumulation_network_params)
                    #restore variables from the pretraining
                    net.restore()
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
                        global_step, net = train(video, blobs,
                                                global_fragments,
                                                net, images, labels,
                                                store_accuracy_and_error = False,
                                                check_for_loss_plateau = True,
                                                save_summaries = False,
                                                print_flag = False,
                                                plot_flag = False,
                                                global_step = global_step,
                                                first_accumulation_flag = accumulation_manager == 0)
                        # update used_for_training flag to True for fragments used
                        accumulation_manager.update_global_fragments_used_for_training()
                        accumulation_manager.update_used_images_and_labels()
                        accumulation_manager.update_individual_fragments_used()
                        # update the identity of the accumulated global fragments to their labels during training
                        accumulation_manager.assign_identities_to_accumulated_global_fragments(blobs)
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
                    else:
                        print("All the global fragments have been used in the accumulation")
                    assignation_time = time.time() - start
                    #############################################################
                    ###################     Accuracies     ######################
                    ####
                    #############################################################
                    print("\n**** Accuracies ****")
                    number_correct_assignations = [0] * group_size
                    number_of_identity_repetitions = 0
                    number_of_identity_shifts_in_accumulated_frames = 0
                    number_of_none_assigned_blobs = [0] * group_size
                    for frame_number, blobs_in_frame in enumerate(blobs):
                        identities_in_frame = []
                        for i, blob in enumerate(blobs_in_frame):
                            if blob.identity is not None:
                                if blob.identity == blob.user_generated_identity:
                                    number_correct_assignations[i] += 1
                                elif blob._assigned_during_accumulation:
                                    number_of_identity_shifts_in_accumulated_frames += 1
                                if blob.identity in identities_in_frame:
                                    number_of_identity_repetitions += 1
                                    print(blob._assigned_during_accumulation)
                                    if blob._assigned_during_accumulation:
                                        raise ValueError("duplications during accumulation in frame %i" %frame_number)

                                identities_in_frame.append(blob.identity)
                            elif blob.identity is None:
                                number_of_none_assigned_blobs[i] += 1


                    individual_accuracies_assigned_frames = np.asarray(number_correct_assignations)/(num_frames - np.asarray(number_of_none_assigned_blobs))
                    accuracy_assigned_frames = np.sum(number_correct_assignations)/(num_frames * group_size - sum(number_of_none_assigned_blobs))
                    individual_accuracies = np.asarray(number_correct_assignations)/num_frames
                    accuracy = np.sum(number_correct_assignations)/(num_frames * group_size)
                    print("individual_accuracies (assigned frames): ", individual_accuracies_assigned_frames)
                    print("accuracy (assigned frames: ", accuracy_assigned_frames)
                    print("individual_accuracies (assigned frames): ", individual_accuracies)
                    print("accuracy (assigned frames: ", accuracy)

                    #############################################################
                    ###################  Update data-frame   ####################
                    #############################################################
                    results_data_frame = results_data_frame.append({'date': time.strftime("%c"),
                                                                    'cluster': int(job_config.cluster) ,
                                                                    'condition_string': job_config.condition,
                                                                    'pretraining_flag': job_config.pretraining_flag ,
                                                                    'train_filters_flag': bool(job_config.train_filters),
                                                                    'knowledge_transfer_flag': job_config.knowledge_transfer_flag,
                                                                    'number_of_fragments': int(len(global_fragments)),
                                                                    'proportion_of_fragments_in_pretraining': number_of_pretraining_global_fragments/len(global_fragments),
                                                                    'certainty_in_accumulation': job_config.accumulation_certainty,
                                                                    'IMDB_codes': job_config.IMDB_codes,
                                                                    'ids_codes': job_config.ids_codes,
                                                                    'group_size': int(group_size),
                                                                    'frames_in_video': int(num_frames),
                                                                    'frames_per_fragment': int(frames_in_fragment),
                                                                    'repetition': int(repetition),
                                                                    'proportion_of_accumulated_fragments': len([global_fragment.used_for_training for global_fragment in global_fragments])/len(global_fragments),
                                                                    'number_of_none_assigned_blobs': number_of_none_assigned_blobs,
                                                                    'individual_accuracies': individual_accuracies,
                                                                    'individual_accuracies(assigned)': individual_accuracies_assigned_frames,
                                                                    'accuracy': accuracy,
                                                                    'accuracy(assigned)': accuracy_assigned_frames,
                                                                    'proportion_of_identity_repetitions': number_of_identity_repetitions/(num_frames * group_size - sum(number_of_none_assigned_blobs)) ,
                                                                    'proportion_of_identity_shifts_in_accumulated_frames': number_of_identity_shifts_in_accumulated_frames/(num_frames * group_size - sum(number_of_none_assigned_blobs)) ,
                                                                    'pretraining_time': pretraining_time,
                                                                    'accumulation_time': accumulation_time,
                                                                    'assignation_time': assignation_time
                                                                     }, ignore_index=True)
