from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')

import cv2
import numpy as np
from GUI_utils import selectDir
from get_portraits import get_body
from blob import ListOfBlobs, Blob

class CrossingDataset(object):
    def __init__(self, blobs_list, video):
        self.blobs = blobs_list
        self.video_height = video._height
        self.video_width = video._width

    def get_data(self, sampling_ratio_start = 0, sampling_ratio_end = 1.):
        # positive examples (crossings)
        np.random.seed(0)
        crossings = [blob for frame in self.blobs for blob in frame if blob.is_a_crossing and not blob.is_a_ghost_crossing]
        np.random.shuffle(crossings)
        print("num crossings ", len(crossings))
        self.crossings = self.slice(crossings, sampling_ratio_start, sampling_ratio_end)
        self.crossings =  self.generate_crossing_images()
        self.crossing_labels = np.ones(len(self.crossings))
        assert len(self.crossing_labels) == len(self.crossings)
        # negative examples (non crossings)
        fish = [blob.portrait for frame in self.blobs for blob in frame if blob.is_a_fish and blob.user_generated_identity != -1]
        np.random.shuffle(fish)
        self.fish = self.slice(fish, sampling_ratio_start, sampling_ratio_end)
        self.fish = self.generate_fish_images()
        self.fish_labels = np.zeros(len(self.fish))
        assert len(self.fish_labels) == len(self.fish)

    def generate_crossing_images(self):
        crossing_images = []
        self.image_size = np.max([np.max(crossing.bounding_box_image.shape) for crossing in self.crossings]) + 5
        for crossing in self.crossings:
            crossing_image, _, _ = get_body(self.video_height, self.video_width, crossing.bounding_box_image, crossing.pixels, crossing.bounding_box_in_frame_coordinates, self.image_size , only_blob = True)
            crossing_image = ((crossing_image - np.mean(crossing_image))/np.std(crossing_image)).astype('float32')
            crossing_images.append(crossing_image)
        return crossing_images

    def generate_fish_images(self):
        return [self.pad_image(fish_image, self.image_size) for fish_image in self.fish]

    @staticmethod
    def pad_image(image, size):
        pad_size = int(np.floor((size - image.shape[0]) / 2))
        if pad_size > 0:
            image = cv2.copyMakeBorder( image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT)
        return image

    def generate_test_set(self):
        # test generalisation on ambiguous examples
        self.test = np.random.shuffle([blob for frame in self.blobs for blob in frame if blob.user_generated_identity == -1])

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
    training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .9)

    validation_set = CrossingDataset(blobs, video)
    validation_set.get_data(sampling_ratio_start = .9, sampling_ratio_end = 1.)
