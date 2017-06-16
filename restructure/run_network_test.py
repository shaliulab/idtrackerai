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
sys.path.append('./library')
sys.path.append('./network')
# sys.path.append('IdTrackerDeep/tracker')

from network_params import NetworkParams
from trainer import train
from id_CNN import ConvNetwork

from library_utils import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs, LibraryJobConfig, check_if_repetition_has_been_computed

def slice_data_base(images, labels, indicesIndiv):
    ''' Select images and labels relative to a subset of individuals'''
    print('\nSlicing database...')
    images = np.array(np.concatenate([images[labels==ind] for ind in indicesIndiv], axis=0))
    labels = np.array(np.concatenate([i*np.ones(sum(labels==ind)).astype(int) for i,ind in enumerate(indicesIndiv)], axis=0))
    return images, labels

def getUncorrelatedImages(images,labels,num_images, minimum_number_of_images_per_individual):
    print('\n *** Getting uncorrelated images')
    print("number of images: ", num_images)
    new_images = []
    new_labels= []

    for i in np.unique(labels):
        # print('individual, ', i)
        # Get images of this individual
        thisIndivImages = images[labels==i]
        thisIndivLabels = labels[labels==i]
        # print('num images of this individual, ', thisIndivImages.shape[0])

        # Get train, validation and test, images and labels
        new_images.append(thisIndivImages[:num_images])
        new_labels.append(thisIndivLabels[:num_images])

    return np.concatenate(new_images, axis=0), np.concatenate(new_labels, axis=0)

if __name__ == '__main__':
    '''
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: path to test_data_frame.pkl
    argv[3]: test_number
    '''
    print("cluster:", sys.argv[1])
    print("test_data_frame:", sys.argv[2])
    print("test_number:", sys.argv[3])

    tests_data_frame = pd.read_pickle(sys.argv[2])
    test_dictionary = tests_data_frame.loc[int(sys.argv[3])].to_dict()
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

    for group_size in job_config.group_sizes:

        for frames_in_video in job_config.frames_in_video:

            for frames_in_fragment in job_config.frames_per_individual_fragment:

                for repetition in job_config.repetitions:
                    save_folder = os.path.join(job_config.condition_path,'group_size_' + str(group_size),
                                                            'num_frames_' + str(frames_in_video),
                                                            'frames_in_fragment_' + str(frames_in_fragment),
                                                            'repetition_' + str(repetition))


                    print("\n********** group size %i - frames_in_video %i - frames_in_fragment %i - repetition %i ********" %(group_size,frames_in_video,frames_in_fragment,repetition))
                    already_computed = False
                    if os.path.isfile('./library/results_data_frame.pkl'):
                        already_computed = check_if_repetition_has_been_computed(results_data_frame, job_config, group_size, frames_in_video, frames_in_fragment, repetition)
                    if already_computed:
                        print("The network with this comditions has been already tested")
                    else:

                        # Get individuals indices for this repetition
                        print('Seeding the random generator...')
                        np.random.seed(repetition)
                        permutation_individual_indices = np.random.permutation(dataset.number_of_animals)
                        individual_indices = permutation_individual_indices[:group_size]
                        print('individual indices, ', individual_indices)

                        # Get current individuals images
                        images, labels = slice_data_base(dataset.images, dataset.labels, individual_indices)
                        print("images shape: ", images.shape)
                        print("labels shape: ", labels.shape)
                        print("labels: ", np.unique(labels))

                        images, labels = getUncorrelatedImages(images, labels, frames_in_video, dataset.minimum_number_of_images_per_animal)
                        print("images shape: ", images.shape)
                        print("labels shape: ", labels.shape)

                        train_network_params = NetworkParams(group_size,
                                                                learning_rate = 0.01,
                                                                keep_prob = 1.0,
                                                                use_adam_optimiser = False,
                                                                scopes_layers_to_optimize = None,
                                                                restore_folder = None,
                                                                save_folder = save_folder)
                        net = ConvNetwork(train_network_params)
                        net.restore()
                        start_time = time.time()
                        _, _, store_training_accuracy_and_loss_data = train(None, None, None,
                                                                            net, images, labels,
                                                                            store_accuracy_and_error = True,
                                                                            check_for_loss_plateau = True,
                                                                            save_summaries = False,
                                                                            print_flag = True,
                                                                            plot_flag = False,
                                                                            global_step = 0,
                                                                            first_accumulation_flag = True)
                        total_time = time.time() - start_time
                        print("accuracy: ", store_training_accuracy_and_loss_data.accuracy[-1])
                        print("individual_accuracies: ", store_training_accuracy_and_loss_data.individual_accuracy[-1])

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
                                                                        'IMDB_codes': job_config.IMDB_codes,
                                                                        'ids_codes': job_config.ids_codes,
                                                                        'group_size': int(group_size),
                                                                        'frames_in_video': int(frames_in_video),
                                                                        'frames_per_fragment': int(frames_in_fragment),
                                                                        'repetition': int(repetition),
                                                                        'number_of_fragments': None,
                                                                        'proportion_of_accumulated_fragments': None,
                                                                        'number_of_not_assigned_blobs': None,
                                                                        'individual_accuracies': store_training_accuracy_and_loss_data.individual_accuracy[-1],
                                                                        'individual_accuracies(assigned)': store_training_accuracy_and_loss_data.individual_accuracy[-1],
                                                                        'accuracy': store_training_accuracy_and_loss_data.accuracy[-1],
                                                                        'accuracy(assigned)': store_training_accuracy_and_loss_data.accuracy[-1],
                                                                        'proportion_of_identity_repetitions': None,
                                                                        'proportion_of_identity_shifts_in_accumulated_frames': None,
                                                                        'pretraining_time': None,
                                                                        'accumulation_time': None,
                                                                        'assignation_time': None,
                                                                        'total_time': total_time,
                                                                         }, ignore_index=True)

                        results_data_frame.to_pickle('./library/results_data_frame.pkl')

                        blobs = None
                        global_fragments = None
