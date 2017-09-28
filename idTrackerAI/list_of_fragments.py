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

from fragment import Fragment
from py_utils import set_attributes_of_object_to_value

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

    def reset_fragments(list_of_fragments, roll_back_to = None):
        [fragment.reset(roll_back_to) for fragment in self.fragments]

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

    def save(self):
        logger.info("saving list of fragments at %s" %self.video.fragments_path)
        np.save(self.video.fragments_path,self)

    @classmethod
    def load(self, path_to_load):
        logger.info("loading list of fragments from %s" %path_to_load)
        list_of_fragments = np.load(path_to_load).item()
        return list_of_fragments

def create_list_of_fragments(blobs_in_video, number_of_animals):
    attributes_to_set_to_value = ['_portrait', 'bounding_box_image', 'bounding_box_in_frame_coordinates'
                                        '_area', '_next', '_previous',]
    fragments = []
    used_fragment_identifiers = []

    for blobs_in_frame in blobs_in_video:
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
                                    (start, end+1), # it is not inclusive to follow Python convention
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
                used_fragment_identifiers.append(current_fragment_identifier)
                fragments.append(fragment)

            set_attributes_of_object_to_value(blob, attributes_to_set_to_value, value = None)

    [fragment.get_coexisting_individual_fragments_indices(fragments) for fragment in fragments]
    return fragments
