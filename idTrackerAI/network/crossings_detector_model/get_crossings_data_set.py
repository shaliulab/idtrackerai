from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./network')
import cv2
import numpy as np
from GUI_utils import selectDir
from blob import Blob
from list_of_blobs import ListOfBlobs
import matplotlib.pyplot as plt
from get_data import duplicate_PCA_images

if sys.argv[0] == 'idtrackeraiApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.get_crossing_data_set")

class CrossingDataset(object):
    def __init__(self, blobs_list, video,
                crossings = [], individual_blobs = [], test = [],
                image_size = None, scope = '', dataset_image_size = 40):
        self.video = video
        self.scope = scope
        self.dataset_image_size = dataset_image_size
        self.blobs = blobs_list
        self.get_video_height_and_width_according_to_resolution_reduction()
        if (scope == 'training' or scope == 'validation'):
            self.get_list_of_crossing_blobs_for_training(crossings, image_size)
            self.get_list_of_individual_blobs_for_training(individual_blobs)
            logger.info("number of sure crossing images used for training: %i" %len(self.crossing_blobs))
            logger.info("number of individual images used for training: %i" %len(self.individual_blobs))
        if scope == 'test':
            self.image_size = image_size
            self.get_list_of_blobs_for_test(test)
            logger.info("number of test images: %i" %len(self.test))


    def get_video_height_and_width_according_to_resolution_reduction(self):
        if self.video.resolution_reduction == 1:
            self.video_height = self.video.height
            self.video_width = self.video.width
        else:
            self.video_height  = int(self.video.height * self.video.resolution_reduction)
            self.video_width  = int(self.video.width * self.video.resolution_reduction)

    def get_list_of_individual_blobs_for_training(self, individual_blobs):
        if len(individual_blobs) == 0:
            self.individual_blobs = [blob for blobs_in_frame in self.blobs for blob in blobs_in_frame
                                    if blob.is_a_sure_individual()
                                    or blob.in_a_global_fragment_core(blobs_in_frame)]
            logger.debug("number of individual blobs (before cut): %i" %len(self.individual_blobs))
            np.random.shuffle(self.individual_blobs)
            ratio = 1
            if len(self.individual_blobs) > ratio * len(self.crossing_blobs):
                self.individual_blobs = self.individual_blobs[:ratio * len(self.crossing_blobs)]

        else:
            self.individual_blobs = individual_blobs

    def get_list_of_crossing_blobs_for_training(self, crossings, image_size):
        if len(crossings) == 0 or image_size is None:

            self.crossing_blobs = [blob for blobs_in_frame in self.blobs for blob in blobs_in_frame
                                    if blob.is_a_sure_crossing()]
            logger.debug("number of crossing blobs (in get list): %i" %len(self.crossing_blobs))
            np.random.seed(0)
            np.random.shuffle(self.crossing_blobs)
            self.image_size = np.max([np.max(crossing.bounding_box_image.shape) for crossing in self.crossing_blobs]) + 5
        else:
            self.crossing_blobs = crossings
            self.image_size = image_size

    def get_list_of_blobs_for_test(self, test):
        if len(test) == 0:
            self.test = [blob for blobs_in_frame in self.blobs for blob in blobs_in_frame
                            if (blob.is_an_individual
                                and not blob.in_a_global_fragment_core(blobs_in_frame)
                                and not blob.is_a_sure_individual())
                            or (blob.is_a_crossing
                                and not blob.is_a_sure_crossing())]
        else:
            self.test = test

    def get_data(self, sampling_ratio_start = 0, sampling_ratio_end = 1.):
        logger.info("Generating crossing %s set." %self.scope)
        self.crossings_sliced = self.slice(self.crossing_blobs, sampling_ratio_start, sampling_ratio_end)
        self.crossings_images =  self.generate_crossing_images()
        self.crossing_labels = np.ones(len(self.crossings_images))
        if self.scope == "training":
            self.crossings_images, self.crossing_labels = duplicate_PCA_images(self.crossings_images, self.crossing_labels)
        assert len(self.crossing_labels) == len(self.crossings_images)
        logger.info("Generating single individual %s set" %self.scope)
        self.individual_blobs_sliced = self.slice(self.individual_blobs, sampling_ratio_start, sampling_ratio_end)
        self.individual_blobs_images = self.generate_individual_blobs_images()
        self.individual_blobs_labels = np.zeros(len(self.individual_blobs_images))
        assert len(self.individual_blobs_labels) == len(self.individual_blobs_images)
        logger.info("Preparing images and labels")
        self.images = np.asarray(list(self.crossings_images) + list(self.individual_blobs_images))
        self.images = np.expand_dims(self.images, axis = 3)
        self.labels = np.concatenate([self.crossing_labels, self.individual_blobs_labels], axis = 0)
        self.labels = self.dense_to_one_hot(np.expand_dims(self.labels, axis = 1))
        assert len(self.images) == len(self.labels)
        if self.scope == 'training':
            self.weight_positive = 1 - len(self.crossing_labels)/len(self.labels)
        else:
            self.weight_positive = 1
        np.random.seed(0)
        permutation = np.random.permutation(len(self.labels))
        self.images = self.images[permutation]
        self.labels = self.labels[permutation]

    def compute_resampling_factor(self):
        if not hasattr(self, 'resampling_factor'):
            self.resampling_factor = self.dataset_image_size / self.image_size

    def generate_crossing_images(self):
        crossing_images = []
        self.compute_resampling_factor()
        logger.debug("resampling factor crossings: %s" %str(self.resampling_factor))

        for crossing in self.crossings_sliced:
            _, _, _, crossing_image = crossing.get_image_for_identification(self.video, image_size = self.image_size)
            crossing_image = cv2.resize(crossing_image, None,
                                        fx = self.resampling_factor,
                                        fy = self.resampling_factor,
                                        interpolation = cv2.INTER_CUBIC)
            crossing_image = ((crossing_image - np.mean(crossing_image))/np.std(crossing_image)).astype('float32')
            crossing_images.append(crossing_image)

        return crossing_images

    def generate_individual_blobs_images(self):
        individual_blobs_images = []
        self.compute_resampling_factor()
        logger.debug("resampling factor individual: %s" %str(self.resampling_factor))

        for individual_blobs in self.individual_blobs_sliced:
            _, _, _, individual_blobs_image = individual_blobs.get_image_for_identification(self.video, image_size = self.image_size)
            individual_blobs_image = cv2.resize(individual_blobs_image, None,
                                                fx = self.resampling_factor,
                                                fy = self.resampling_factor,
                                                interpolation = cv2.INTER_CUBIC)
            individual_blobs_image = ((individual_blobs_image - np.mean(individual_blobs_image))/np.std(individual_blobs_image)).astype('float32')
            individual_blobs_images.append(individual_blobs_image)

        return individual_blobs_images

    def generate_test_images(self, interval = None):
        test_images = []
        self.compute_resampling_factor()
        logger.debug("resampling factor test: %s" %str(self.resampling_factor))
        if interval is None:
            blobs = self.test
        else:
            blobs = self.test[interval[0]:interval[1]]
        for blob in blobs:
            _, _, _, test_image = blob.get_image_for_identification(self.video, image_size = self.image_size)
            test_image = cv2.resize(test_image, None,
                                    fx = self.resampling_factor,
                                    fy = self.resampling_factor,
                                    interpolation = cv2.INTER_CUBIC)
            test_image = ((test_image - np.mean(test_image))/np.std(test_image)).astype('float32')
            test_images.append(test_image)

        return np.expand_dims(np.asarray(test_images), axis = 3)

    @staticmethod
    def dense_to_one_hot(labels, n_classes=2):
        """Convert class labels from scalars to one-hot vectors."""
        labels = np.array(labels)
        n_labels = labels.shape[0]
        index_offset = np.arange(n_labels) * n_classes
        labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.int16)
        indices = (index_offset + labels.ravel()).astype('int')
        labels_one_hot.flat[indices] = 1
        return labels_one_hot

    @staticmethod
    def slice(data, sampling_ratio_start, sampling_ratio_end):
        num_examples = len(data)
        start = int(num_examples * sampling_ratio_start)
        end = int(num_examples * sampling_ratio_end)
        return data[start : end]

if __name__ == "__main__":
    ''' select blobs list tracked to compare against ground truth '''
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    logger.info("loading video object...")
    video = np.load(video_path).item(0)
    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(video, blobs_path)
    blobs = list_of_blobs.blobs_in_video

    training_set = CrossingDataset(blobs, video)
    training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .5, scope = 'training')
    np.save(video_folder + '_training_set.npy', training_set)
    validation_set = CrossingDataset(blobs, video)
    validation_set.get_data(sampling_ratio_start = .5, sampling_ratio_end = 1., scope = 'validation')
    np.save(video_folder + '_validation_set.npy', validation_set)
