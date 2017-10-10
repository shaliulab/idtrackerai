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

logger = logging.getLogger("__main__.list_of_blobs")

class ListOfBlobs(object):
    def __init__(self, video, blobs_in_video = None):
        self.video = video
        self.blobs_in_video = blobs_in_video
        self.number_of_frames_with_blobs = len(self.blobs_in_video)
        self.check_consistency_number_of_frames_and_number_of_frames_with_blobs()

    def check_consistency_number_of_frames_and_number_of_frames_with_blobs(self):
        frame_diff = int(self.video.number_of_frames - self.number_of_frames_with_blobs)
        if frame_diff != 0:
            logger.warning("%i frames do not contain blobs" %frame_diff)
        self.number_of_frames_without_blobs = frame_diff

    def disconect(self):
        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                if hasattr(blob, 'next'): setattr(blob, 'next', [])
                if hasattr(blob, 'previous'): setattr(blob, 'previous', [])

    def reconnect(self):
        if self.video._has_been_segmented:
            logger.info("Reconnecting list of blob objects")
            for frame_i in self.cutting_points:
                for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                    if blob_0.overlaps_with(blob_1):
                        blob_0.now_points_to(blob_1)

    def save(self, path_to_save = None, number_of_chunks = 1):
        """save instance"""
        if path_to_save is None:
            path_to_save = self.video.blobs_path
        self.disconnect()
        logger.info("saving blobs list at %s" %path_to_save)
        np.save(path_to_save, self)
        self.compute_overlapping_between_subsequent_frames()

    @classmethod
    def load(cls, path_to_load_blob_list_file):
        logger.info("loading blobs list from %s" %path_to_load_blob_list_file)

        list_of_blobs = np.load(path_to_load_blob_list_file).item()
        logging.debug("cutting points %s" %list_of_blobs.cutting_points)
        list_of_blobs.compute_overlapping_between_subsequent_frames()
        return list_of_blobs

    def compute_fragment_identifier_and_blob_index(self):
        counter = 0
        possible_blob_indices = range(self.video.maximum_number_of_blobs)

        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'assigning fragment identifier'):
            used_blob_indices = [blob.blob_index for blob in blobs_in_frame if blob.blob_index is not None]
            missing_blob_indices =  list(set(possible_blob_indices).difference(set(used_blob_indices)))
            for blob in blobs_in_frame:
                if blob.fragment_identifier is None and blob.is_a_fish:
                    blob._fragment_identifier = counter
                    blob_index = missing_blob_indices.pop(0)
                    blob._blob_index = blob_index
                    blob.non_shared_information_with_previous = 1.
                    if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_a_fish:
                        blob.next[0]._fragment_identifier = counter
                        blob.next[0]._blob_index = blob_index
                        blob.next[0].compute_overlapping_with_previous_blob()
                        if blob.next[0].is_a_fish_in_a_fragment:
                            blob = blob.next[0]

                            while len(blob.next) == 1 and blob.next[0].is_a_fish_in_a_fragment:
                                blob = blob.next[0]
                                blob._fragment_identifier = counter
                                blob._blob_index = blob_index
                                # compute_overlapping_with_previous_blob
                                blob.compute_overlapping_with_previous_blob()

                            if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_a_fish:
                                blob.next[0]._fragment_identifier = counter
                                blob.next[0]._blob_index = blob_index
                                blob.next[0].compute_overlapping_with_previous_blob()
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

        for frame_i in tqdm(xrange(1, self.number_of_frames_with_blobs), desc = 'Connecting blobs '):
            set_frame_number_to_blobs_in_frame(self.blobs_in_video[frame_i-1], frame_i-1)

            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)
        set_frame_number_to_blobs_in_frame(self.blobs_in_video[frame_i], frame_i)

    def compute_model_area_and_body_length(self):
        """computes the median and standard deviation of all the blobs of the video
        and the median_body_length estimated from the diagonal of the bounding box.
        These values are later used to discard blobs that are not fish and potentially
        belong to a crossing.
        """
        #areas are collected throughout the entire video in the cores of the global fragments
        areas_and_body_length = np.asarray([(blob.area,blob.estimated_body_length) for blobs_in_frame in self.blobs_in_video
                                                                                    for blob in blobs_in_frame
                                                                                    if len(blobs_in_frame) == self.video.number_of_animals])
        median_area = np.median(areas_and_body_length[:,0])
        mean_area = np.mean(areas_and_body_length[:,0])
        std_area = np.std(areas_and_body_length[:,0])
        median_body_length = np.median(areas_and_body_length[:,1])
        return ModelArea(mean_area, median_area, std_area), median_body_length

    def apply_model_area_to_video(self, model_area, portrait_size):
        def apply_model_area_to_blobs_in_frame(video, blobs_in_frame, model_area, portrait_size):
            number_of_blobs = len(blobs_in_frame)
            for blob in blobs_in_frame:
                blob.apply_model_area(video, model_area, portrait_size, number_of_blobs)
        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'Applying model area'):
            apply_model_area_to_blobs_in_frame(self.video, blobs_in_frame, model_area, portrait_size)

    def check_maximal_number_of_blob(self):
        frames_with_more_blobs_than_animals = []
        for frame_number, blobs_in_frame in enumerate(self.blobs_in_video):

            if len(blobs_in_frame) > self.video.number_of_animals:
                frames_with_more_blobs_than_animals.append(frame_number)

        if len(frames_with_more_blobs_than_animals) > 0:
            logger.error('There are frames with more blobs than animals, this can be detrimental for the proper functioning of the system.')
            logger.error("Frames with more blobs than animals: %s" %str(frames_with_more_blobs_than_animals))
            raise ValueError('Please check your segmentaion')
        return frames_with_more_blobs_than_animals

    def update_from_list_of_fragments(self, fragments):
        attributes = ['_identity',
                        '_identity_corrected_solving_duplication',
                        '_user_generated_identity', '_used_for_training']

        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'updating list of blobs from list of fragments'):
            for blob in blobs_in_frame:
                fragment = fragments[self.video.fragment_identifier_to_index[blob.fragment_identifier]]
                values = [fragment.identity,
                            fragment.identity_corrected_solving_duplication,
                            fragment.user_generated_identity,
                            fragment.used_for_training]
                [setattr(blob, attribute, value) for attribute, value in zip(attributes, values)]
