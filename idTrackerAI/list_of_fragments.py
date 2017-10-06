from __future__ import absolute_import, division, print_function
import os
import sys
import random
import logging

sys.path.append('./utils')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from tqdm import tqdm

from fragment import Fragment
from py_utils import set_attributes_of_object_to_value, append_values_to_lists

logger = logging.getLogger("__main__.list_of_fragments")

class ListOfFragments(object):
    def __init__(self, video, fragments):
        self.video = video
        self.fragments = fragments
        self.number_of_fragments = len(self.fragments)

    def get_fragment_identifier_to_index_list(self):
        fragments_identifiers = [fragment.identifier for fragment in self.fragments]
        fragment_identifier_to_index = np.arange(len(fragments_identifiers))
        fragments_identifiers_argsort = np.argsort(fragments_identifiers)
        return fragment_identifier_to_index[fragments_identifiers_argsort]

    def reset(self, roll_back_to = None):
        [fragment.reset(roll_back_to) for fragment in self.fragments]

    def get_images_from_fragments_to_assign(self):
        return np.concatenate([np.asarray(fragment.images) for fragment in self.fragments
                                if not fragment.used_for_training and fragment.is_a_fish], axis = 0)

    def get_data_plot(self):
        number_of_images_in_individual_fragments = []
        distance_travelled_individual_fragments = []
        number_of_images_in_crossing_fragments = []
        for fragment in self.fragments:
            if fragment.is_a_fish:
                number_of_images_in_individual_fragments.append(fragment.number_of_images)
                distance_travelled_individual_fragments.append(fragment.distance_travelled)
            elif fragment.is_a_crossing:
                number_of_images_in_crossing_fragments.append(fragment.number_of_images)
        return np.asarray(number_of_images_in_individual_fragments),\
                np.asarray(distance_travelled_individual_fragments),\
                number_of_images_in_crossing_fragments

    def update_from_list_of_blobs(self, blobs_in_video):
        [setattr(self.fragments[self.video.fragment_identifier_to_index[blob.fragment_identifier]], '_user_generated_identity', blob.user_generated_identity)
            for blobs_in_frame in blobs_in_video for blob in blobs_in_frame if blob.user_generated_identity is not None ]

    def save(self):
        logger.info("saving list of fragments at %s" %self.video.fragments_path)
        [setattr(fragment, 'coexisting_individual_fragments', None) for fragment in self.fragments]
        np.save(self.video.fragments_path,self)
        [fragment.get_coexisting_individual_fragments_indices(self.fragments) for fragment in self.fragments]

    @classmethod
    def load(self, path_to_load):
        logger.info("loading list of fragments from %s" %path_to_load)
        list_of_fragments = np.load(path_to_load).item()
        [fragment.get_coexisting_individual_fragments_indices(list_of_fragments.fragments) for fragment in list_of_fragments.fragments]
        return list_of_fragments

def create_list_of_fragments(blobs_in_video, number_of_animals):
    attributes_to_set = ['_portrait', 'bounding_box_image', 'bounding_box_in_frame_coordinates'
                                        '_area', '_next', '_previous',]
    fragments = []
    used_fragment_identifiers = set()

    for blobs_in_frame in tqdm(blobs_in_video, desc = 'creating list of fragments'):
        for blob in blobs_in_frame:
            current_fragment_identifier = blob.fragment_identifier
            if current_fragment_identifier not in used_fragment_identifiers:
                images = [blob.portrait]
                centroids = [blob.centroid]
                areas = [blob.area]
                pixels = [blob.pixels]
                start = blob.frame_number
                current = blob

                while len(current.next) > 0 and current.next[0].fragment_identifier == current_fragment_identifier:
                    current = current.next[0]
                    images, centroids, areas, pixels = append_values_to_lists([current.portrait,
                                                                current.centroid,
                                                                current.area,
                                                                current.pixels],
                                                                [images,
                                                                centroids,
                                                                areas,
                                                                pixels])

                end = current.frame_number
                fragment = Fragment(current_fragment_identifier,
                                    (start, end + 1), # it is not inclusive to follow Python convention
                                    blob.blob_index,
                                    images,
                                    centroids,
                                    areas,
                                    pixels,
                                    blob.is_a_fish,
                                    blob.is_a_crossing,
                                    blob.is_a_jump,
                                    blob.is_a_jumping_fragment,
                                    blob.is_a_ghost_crossing,
                                    number_of_animals)
                if fragment.is_a_ghost_crossing:
                    fragment.next_blobs_fragment_identifier = [next_blob.fragment_identifier for next_blob in blob.next if len(blob.next) > 0]
                    fragment.previous_blobs_fragment_identifier = [previous_blob.fragment_identifier for previous_blob in blob.previous if len(blob.previous) > 0]
                used_fragment_identifiers.add(current_fragment_identifier)
                fragments.append(fragment)

            set_attributes_of_object_to_value(blob, attributes_to_set, value = None)

    [fragment.get_coexisting_individual_fragments_indices(fragments) for fragment in fragments]
    return fragments
