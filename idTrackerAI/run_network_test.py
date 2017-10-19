from __future__ import absolute_import, division, print_function
# Import standard libraries
import os
from os.path import isdir, isfile
import sys

import glob
import numpy as np
import pandas as pd
import cPickle as pickle
import time

# Import third party libraries
import cv2
from pprint import pprint
import logging.config
import yaml

# Import application/library specifics
sys.path.append('./utils')
sys.path.append('./library')
sys.path.append('./network/identification_model')

from network_params import NetworkParams
from trainer import train
from id_CNN import ConvNetwork

from library_utils import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs, LibraryJobConfig, check_if_repetition_has_been_computed

def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG',
    path_to_save_logs = './'):
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        if os.path.exists(path_to_save_logs):
            logs_folder = os.path.join(path_to_save_logs, 'log_files')
            if not os.path.isdir(logs_folder):
                os.makedirs(logs_folder)
            config['handlers']['info_file_handler']['filename'] = os.path.join(logs_folder, 'info.log')
            config['handlers']['error_file_handler']['filename'] = os.path.join(logs_folder, 'error.log')
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

    logger = logging.getLogger(__name__)
    logger.propagate = True
    logger.setLevel("DEBUG")
    return logger

def slice_data_base(images, labels, indicesIndiv):
    ''' Select images and labels relative to a subset of individuals'''
    print('\nSlicing database...')
    images = np.array(np.concatenate([images[labels==ind] for ind in indicesIndiv], axis=0))
    labels = np.array(np.concatenate([i*np.ones(sum(labels==ind)).astype(int) for i,ind in enumerate(indicesIndiv)], axis=0))
    return images, labels

def getUncorrelatedImages(images,labels,num_images, minimum_number_of_images_per_individual):
    print('\n *** Getting uncorrelated images')
    print("number of images: ", num_images)
    print("minimum number of images in IMDB: ", minimum_number_of_images_per_individual)
    new_images = []
    new_labels= []
    if num_images > 15000: # on average each animal has 20000 frames from one camera and 20000 frames from the other camera. We select only 15000 images to get only images from one camera
        raise ValueError('The number of images per individual is larger than the minimum number of images per individua in the IMDB')

    for i in np.unique(labels):
        # print('individual, ', i)
        # Get images of this individual
        thisIndivImages = images[labels==i][:15000]
        thisIndivLabels = labels[labels==i][:15000]

        perm = np.random.permutation(len(thisIndivImages))
        thisIndivImages = thisIndivImages[perm]
        thisIndivLabels = thisIndivLabels[perm]

        # Get train, validation and test, images and labels
        new_images.append(thisIndivImages[:num_images])
        new_labels.append(thisIndivLabels[:num_images])

    return np.concatenate(new_images, axis=0), np.concatenate(new_labels, axis=0)

if __name__ == '__main__':
    '''
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: path to test_data_frame.pkl
    argv[3]: test_number on the test_data_frame.pkl
    '''

    print("cluster:", sys.argv[1])
    print("test_number:", sys.argv[2])

    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
    test_dictionary = tests_data_frame.loc[int(sys.argv[2])].to_dict()
    pprint(test_dictionary)

    job_config = LibraryJobConfig(cluster = sys.argv[1], test_dictionary = test_dictionary)
    job_config.create_folders_structure()

    # set log config
    logger = setup_logging(path_to_save_logs = job_config.condition_path)
    logger.info("Log files saved in %s" %job_config.condition_path)

    if os.path.isfile('./library/results_data_frame.pkl'):
        print("results_data_frame.pkl already exists \n")
        results_data_frame = pd.read_pickle('./library/results_data_frame.pkl')
    else:
        print("results_data_frame.pkl does not exist \n")
        results_data_frame = pd.DataFrame()

    dataset = Dataset(IMDB_codes = job_config.IMDB_codes, ids_codes = job_config.ids_codes, preprocessing_type = job_config.preprocessing_type)
    dataset.loadIMDBs()
    print("images shape, ", dataset.images.shape)

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
                        image_size = images.shape[1:] + (1,) # NOTE: Channels number added by hand

                        train_network_params = NetworkParams(group_size,
                                                                cnn_model = job_config.CNN_model,
                                                                learning_rate = 0.005,
                                                                keep_prob = 1.0,
                                                                use_adam_optimiser = False,
                                                                scopes_layers_to_optimize = None,
                                                                restore_folder = None,
                                                                save_folder = save_folder,
                                                                image_size = image_size)
                        net = ConvNetwork(train_network_params)
                        net.restore()
                        start_time = time.time()
                        # if repetition == 1:
                        #     save_summaries = True
                        #     store_accuracy_and_error = True
                        # else:
                        #     save_summaries = False
                        #     store_accuracy_and_error = False
                        store_accuracy_and_error = True
                        images = (images - np.expand_dims(np.expand_dims(np.mean(images,axis=(1,2)),axis=1),axis=2))/np.expand_dims(np.expand_dims(np.std(images, axis = (1,2)), axis = 1), axis = 2).astype('float32')
                        _, _, store_validation_accuracy_and_loss_data = train(None, None, None,
                                                                            net, images, labels,
                                                                            store_accuracy_and_error = store_accuracy_and_error,
                                                                            check_for_loss_plateau = True,
                                                                            save_summaries = False,
                                                                            print_flag = False,
                                                                            plot_flag = False,
                                                                            global_step = 0,
                                                                            first_accumulation_flag = True,
                                                                            preprocessing_type = job_config.preprocessing_type,
                                                                            image_size = image_size)
                        total_time = time.time() - start_time
                        if np.isnan(store_validation_accuracy_and_loss_data.loss[-1]):
                            store_validation_accuracy_and_loss_data.loss = store_validation_accuracy_and_loss_data.loss[:-1]
                            store_validation_accuracy_and_loss_data.accuracy = store_validation_accuracy_and_loss_data.accuracy[:-1]
                            store_validation_accuracy_and_loss_data.individual_accuracy = store_validation_accuracy_and_loss_data.individual_accuracy[:-1]

                        print("accuracies (all): ", store_validation_accuracy_and_loss_data.accuracy)
                        print("losses (all): ", store_validation_accuracy_and_loss_data.loss)
                        min_loss_model_index = np.argmin(store_validation_accuracy_and_loss_data.loss)
                        print("accuracy (min loss): ", store_validation_accuracy_and_loss_data.accuracy[min_loss_model_index])
                        print("individual_accuracies (min loss): ", store_validation_accuracy_and_loss_data.individual_accuracy[min_loss_model_index])
                        last_model_index = -1
                        print("accuracy (last): ", store_validation_accuracy_and_loss_data.accuracy[last_model_index])
                        print("individual_accuracies (last): ", store_validation_accuracy_and_loss_data.individual_accuracy[last_model_index])
                        best_accuracy_model_index = np.argmax(store_validation_accuracy_and_loss_data.accuracy)
                        print("accuracy (best): ", store_validation_accuracy_and_loss_data.accuracy[best_accuracy_model_index])
                        print("individual_accuracies (best): ", store_validation_accuracy_and_loss_data.individual_accuracy[best_accuracy_model_index])

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
                                                                        'frames_per_fragment': int(frames_in_fragment),
                                                                        'repetition': int(repetition),
                                                                        'number_of_fragments': None,
                                                                        'proportion_of_accumulated_fragments': None,
                                                                        'number_of_not_assigned_blobs': None,
                                                                        'individual_accuracies_min_loss': store_validation_accuracy_and_loss_data.individual_accuracy[min_loss_model_index],
                                                                        'individual_accuracies_last': store_validation_accuracy_and_loss_data.individual_accuracy[last_model_index],
                                                                        'individual_accuracies_best': store_validation_accuracy_and_loss_data.individual_accuracy[best_accuracy_model_index],
                                                                        'individual_accuracies(assigned)': store_validation_accuracy_and_loss_data.individual_accuracy[best_accuracy_model_index],
                                                                        'accuracy_min_loss': store_validation_accuracy_and_loss_data.accuracy[min_loss_model_index],
                                                                        'accuracy_last': store_validation_accuracy_and_loss_data.accuracy[last_model_index],
                                                                        'accuracy_best': store_validation_accuracy_and_loss_data.accuracy[best_accuracy_model_index],
                                                                        'accuracy(assigned)': store_validation_accuracy_and_loss_data.accuracy[best_accuracy_model_index],
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
