from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./network')

import cv2
import numpy as np
from GUI_utils import selectDir
from get_portraits import get_body
from list_of_blobs import ListOfBlobs
from blob import Blob
import matplotlib.pyplot as plt
from get_data import duplicate_PCA_images

class CrossingDataset(object):
    def __init__(self, blobs_list, video, crossings = [], fish = [], test = [], image_size = None):
        self.blobs = blobs_list
        self.video_height = video._height
        self.video_width = video._width
        self.video = video
        if len(crossings) == 0 or image_size is None:
            self.crossings = [blob for blobs_in_frame in self.blobs for blob in blobs_in_frame if blob.is_a_crossing and not blob.is_a_ghost_crossing]
            np.random.seed(0)
            np.random.shuffle(self.crossings)
            self.image_size = np.max([np.max(crossing.bounding_box_image.shape) for crossing in self.crossings]) + 5
        else:
            self.crossings = crossings
            self.image_size = image_size
        if len(fish) == 0:
            self.fish = [blob for blobs_in_frame in self.blobs for blob in blobs_in_frame if blob.is_a_fish and blob.in_a_global_fragment_core(blobs_in_frame)]
            ratio = 1
            if len(self.fish) > ratio * len(self.crossings):
                self.fish = self.fish[:ratio * len(self.crossings)]
            np.random.shuffle(self.fish)
        else:
            self.fish = fish
        if len(test) == 0:
            self.test = [blob for blobs_in_frame in self.blobs for blob in blobs_in_frame if blob.is_a_fish and not blob.in_a_global_fragment_core(blobs_in_frame)]
        else:
            self.test = test

    def get_data(self, sampling_ratio_start = 0, sampling_ratio_end = 1., scope = ''):
        self.scope = scope
        # positive examples (crossings)
        print("Generating crossing ", scope, " set.")
        self.crossings_sliced = self.slice(self.crossings, sampling_ratio_start, sampling_ratio_end)
        self.crossings_images =  self.generate_crossing_images()
        self.crossing_labels = np.ones(len(self.crossings_images))
        if self.scope == "training":
            # self.crossings_images, self.crossing_labels = self.data_augmentation_by_rotation(self.crossings_images, self.crossing_labels)
            self.crossings_images, self.crossing_labels = duplicate_PCA_images(self.crossings_images, self.crossing_labels)
        assert len(self.crossing_labels) == len(self.crossings_images)
        print("Done")
        # negative examples (non crossings_images)
        print("Generating single individual ", scope, " set")


        self.fish_sliced = self.slice(self.fish, sampling_ratio_start, sampling_ratio_end)
        self.fish_images = self.generate_fish_images()
        self.fish_labels = np.zeros(len(self.fish_images))
        if self.scope == "training":
            # self.fish_images, self.fish_images_labels = self.data_augmentation_by_rotation(self.fish_images, self.fish_labels)
            self.fish_images, self.fish_labels = duplicate_PCA_images(self.fish_images, self.fish_labels)
        assert len(self.fish_labels) == len(self.fish_images)
        print("Done")
        print("Preparing images and labels")
        self.images = np.asarray(list(self.crossings_images) + list(self.fish_images))
        self.images = np.expand_dims(self.images, axis = 3)
        self.labels = np.concatenate([self.crossing_labels, self.fish_labels], axis = 0)
        self.labels = self.dense_to_one_hot(np.expand_dims(self.labels, axis = 1))
        assert len(self.images) == len(self.labels)
        if self.scope == 'training':
            # self.images, self.labels = duplicate_PCA_images(self.images, self.labels)
            self.weight_positive = 1 - len(self.crossing_labels)/len(self.labels)
        else:
            self.weight_positive = 1
        np.random.seed(0)
        permutation = np.random.permutation(len(self.labels))
        self.images = self.images[permutation]
        self.labels = self.labels[permutation]
        print("Done")

    def generate_crossing_images(self):
        crossing_images = []
        for crossing in self.crossings_sliced:
            crossing_image, _, _ = get_body(self.video_height, self.video_width, crossing.bounding_box_image, crossing.pixels, crossing.bounding_box_in_frame_coordinates, self.image_size , only_blob = True)
            crossing_image = ((crossing_image - np.mean(crossing_image))/np.std(crossing_image)).astype('float32')
            crossing_images.append(crossing_image)
        return crossing_images

    def generate_fish_images(self):
        fish_images = []
        for fish in self.fish_sliced:
            fish_image, _, _ = get_body(self.video_height, self.video_width, fish.bounding_box_image, fish.pixels, fish.bounding_box_in_frame_coordinates, self.image_size , only_blob = True)
            fish_image = ((fish_image - np.mean(fish_image))/np.std(fish_image)).astype('float32')
            fish_images.append(fish_image)
        return fish_images

    def generate_test_images(self):
        test_images = []
        for blob in self.test:
            test_image, _, _ = get_body(self.video_height, self.video_width, blob.bounding_box_image, blob.pixels, blob.bounding_box_in_frame_coordinates, self.image_size , only_blob = True)
            test_image = ((test_image - np.mean(test_image))/np.std(test_image)).astype('float32')
            test_images.append(test_image)

        return np.expand_dims(np.asarray(test_images), axis = 3)

    @staticmethod
    def pad_image(image, size):
        pad_size = int(np.floor((size - image.shape[0]) / 2))
        if pad_size > 0:
            image = cv2.copyMakeBorder( image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
        return image

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
    print("loading video object...")
    video = np.load(video_path).item(0)
    #change this
    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(blobs_path)
    blobs = list_of_blobs.blobs_in_video

    training_set = CrossingDataset(blobs, video)
    training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .5, scope = 'training')
    np.save(video_folder + '_training_set.npy', training_set)
    validation_set = CrossingDataset(blobs, video)
    validation_set.get_data(sampling_ratio_start = .5, sampling_ratio_end = 1., scope = 'validation')
    np.save(video_folder + '_validation_set.npy', validation_set)
