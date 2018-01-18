from __future__ import absolute_import, division, print_function
import itertools
import logging
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')

import numpy as np
from tqdm import tqdm

from blob import Blob
from model_area import ModelArea
from erosion import get_eroded_blobs, get_new_blobs_in_frame_after_erosion

logger = logging.getLogger("__main__.list_of_blobs")

class ListOfBlobs(object):
    def __init__(self, blobs_in_video = None, number_of_frames = None):
        self.blobs_in_video = blobs_in_video
        self.number_of_frames = len(self.blobs_in_video)
        self.blobs_are_connected = False

    def disconnect(self):
        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                blob.next, blob.previous = [], []

    def connect(self):
        logger.info("Connecting list of blob objects")
        for frame_i in tqdm(range(1,self.number_of_frames), desc = 'connecting blobs'):
            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)

    def reconnect(self):
        logger.info("re-Connecting list of blob objects")
        for frame_i in tqdm(range(1,self.number_of_frames), desc = 're-connecting blobs'):
            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.fragment_identifier == blob_1.fragment_identifier:
                    blob_0.now_points_to(blob_1)
        self.blobs_are_connected = True

    def save(self, video, path_to_save = None, number_of_chunks = 1):
        """save instance"""
        self.disconnect()
        logger.info("saving blobs list at %s" %path_to_save)
        np.save(path_to_save, self)
        if video.has_been_segmented and not video.has_been_preprocessed:
            self.connect()
        self.blobs_are_connected = True

    @classmethod
    def load(cls, video, path_to_load_blob_list_file):
        logger.info("loading blobs list from %s" %path_to_load_blob_list_file)
        list_of_blobs = np.load(path_to_load_blob_list_file).item()
        list_of_blobs.blobs_are_connected = False
        return list_of_blobs

    def compute_fragment_identifier_and_blob_index(self, number_of_animals):
        counter = 0
        possible_blob_indices = range(number_of_animals)

        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'assigning fragment identifier'):
            used_blob_indices = [blob.blob_index for blob in blobs_in_frame if blob.blob_index is not None]
            missing_blob_indices =  list(set(possible_blob_indices).difference(set(used_blob_indices)))
            for blob in blobs_in_frame:
                if blob.fragment_identifier is None and blob.is_an_individual:
                    blob._fragment_identifier = counter
                    blob_index = missing_blob_indices.pop(0)
                    blob._blob_index = blob_index
                    blob.non_shared_information_with_previous = 1.
                    if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_an_individual:
                        blob.next[0]._fragment_identifier = counter
                        blob.next[0]._blob_index = blob_index
                        # blob.next[0].compute_overlapping_with_previous_blob()
                        if blob.next[0].is_an_individual_in_a_fragment:
                            blob = blob.next[0]

                            while len(blob.next) == 1 and blob.next[0].is_an_individual_in_a_fragment:
                                blob = blob.next[0]
                                blob._fragment_identifier = counter
                                blob._blob_index = blob_index
                                # compute_overlapping_with_previous_blob
                                # blob.compute_overlapping_with_previous_blob()

                            if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_an_individual:
                                blob.next[0]._fragment_identifier = counter
                                blob.next[0]._blob_index = blob_index
                                # blob.next[0].compute_overlapping_with_previous_blob()
                    counter += 1

        self.number_of_individual_fragments = counter
        print("number_of_individual_fragments, ", counter)

    def compute_crossing_fragment_identifier(self):
        """we define a crossing fragment as a crossing that in subsequent frames
        involves the same individuals"""
        def propagate_crossing_identifier(blob, fragment_identifier):
            assert blob.fragment_identifier is None
            blob._fragment_identifier = fragment_identifier
            cur_blob = blob

            while len(cur_blob.next) == 1 and len(cur_blob.next[0].previous) == 1 and cur_blob.next[0].is_a_crossing:
                cur_blob = cur_blob.next[0]
                cur_blob._fragment_identifier = fragment_identifier

            cur_blob = blob

            while len(cur_blob.previous) == 1 and len(cur_blob.previous[0].next) == 1 and cur_blob.previous[0].is_a_crossing:
                cur_blob = cur_blob.previous[0]
                cur_blob._fragment_identifier = fragment_identifier

        fragment_identifier = self.number_of_individual_fragments

        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                if blob.is_a_crossing and blob.fragment_identifier is None:
                    propagate_crossing_identifier(blob, fragment_identifier)
                    fragment_identifier += 1
        print("number_of_crossing_fragments, ", fragment_identifier - self.number_of_individual_fragments)
        print("total number of fragments, ", fragment_identifier)

    def compute_overlapping_between_subsequent_frames(self):
        def set_frame_number_to_blobs_in_frame(blobs_in_frame, frame_number):
            for blob in blobs_in_frame:
                blob.frame_number = frame_number
        self.disconnect()
        for frame_i in tqdm(xrange(1, self.number_of_frames), desc = 'Connecting blobs '):
            set_frame_number_to_blobs_in_frame(self.blobs_in_video[frame_i-1], frame_i-1)

            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)

        set_frame_number_to_blobs_in_frame(self.blobs_in_video[frame_i], frame_i)

    def compute_model_area_and_body_length(self, number_of_animals):
        """computes the median and standard deviation of all the blobs of the video
        and the median_body_length estimated from the diagonal of the bounding box.
        These values are later used to discard blobs that are not fish and potentially
        belong to a crossing.
        """
        #areas are collected throughout the entire video in the cores of the global fragments
        areas_and_body_length = np.asarray([(blob.area,blob.estimated_body_length) for blobs_in_frame in self.blobs_in_video
                                                                                    for blob in blobs_in_frame
                                                                                    if len(blobs_in_frame) == number_of_animals])
        if areas_and_body_length.shape[0] == 0:
            raise ValueError('There is not part in the video where all the animals are visible. Try a different segmentation or check the number of animals in the video.')
        median_area = np.median(areas_and_body_length[:,0])
        mean_area = np.mean(areas_and_body_length[:,0])
        std_area = np.std(areas_and_body_length[:,0])
        median_body_length = np.median(areas_and_body_length[:,1])
        return ModelArea(mean_area, median_area, std_area), median_body_length

    def apply_model_area_to_video(self, video, model_area, identification_image_size, number_of_animals):
        def apply_model_area_to_blobs_in_frame(video, number_of_animals, blobs_in_frame, model_area, identification_image_size):
            number_of_blobs = len(blobs_in_frame)
            for blob in blobs_in_frame:
                blob.apply_model_area(video, number_of_animals, model_area, identification_image_size, number_of_blobs)
        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'Applying model area'):
            apply_model_area_to_blobs_in_frame(video, number_of_animals, blobs_in_frame, model_area, identification_image_size)

    def get_data_plot(self):
        return [blob.area for blobs_in_frame in self.blobs_in_video for blob in blobs_in_frame]


    def check_maximal_number_of_blob(self, number_of_animals):
        frames_with_more_blobs_than_animals = []
        for frame_number, blobs_in_frame in enumerate(self.blobs_in_video):

            if len(blobs_in_frame) > number_of_animals:
                frames_with_more_blobs_than_animals.append(frame_number)

        if len(frames_with_more_blobs_than_animals) > 0:
            logger.error('There are frames with more blobs than animals, this can be detrimental for the proper functioning of the system.')
            logger.error("Frames with more blobs than animals: %s" %str(frames_with_more_blobs_than_animals))
        return frames_with_more_blobs_than_animals

    def update_from_list_of_fragments(self, fragments, fragment_identifier_to_index):
        attributes = ['identity',
                        'P2_vector',
                        'identity_corrected_solving_duplication',
                        'user_generated_identity', 'used_for_training',
                        'accumulation_step']

        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'updating list of blobs from list of fragments'):
            for blob in blobs_in_frame:
                fragment = fragments[fragment_identifier_to_index[blob.fragment_identifier]]
                [setattr(blob, '_' + attribute, getattr(fragment, attribute)) for attribute in attributes if hasattr(fragment, attribute)]

    def compute_nose_and_head_coordinates(self):
        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                blob.get_nose_and_head_coordinates()

    def erode(self, video):
        for frame_number, blobs_in_frame in enumerate(tqdm(self.blobs_in_video, desc = 'eroding blobs')):
            eroded_blobs_in_frame = get_eroded_blobs(video, blobs_in_frame)
            if len(eroded_blobs_in_frame) <= video.number_of_animals:
                self.blobs_in_video[frame_number] = get_new_blobs_in_frame_after_erosion(video, blobs_in_frame, eroded_blobs_in_frame)
