from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')

import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from math import sqrt

from statistics_for_assignment import compute_P1_individual_fragment_from_frequencies
from get_portraits import get_portrait, get_body

# STD_TOLERANCE = 1 # tolerance to select a blob as being a single fish according to the area model
### NOTE set to 1 because we changed the model area to work with the median.
logger = logging.getLogger("__main__.fragment")

class Fragment(object):
    def __init__(self, fragment_identifier,\
                        start_end,\
                        blob_hierarchy_in_starting_frame,\
                        images,\
                        centroids,\
                        areas,\
                        pixels,\
                        is_a_fish,\
                        is_a_crossing,\
                        is_a_jump,\
                        is_a_jumping_fragment,\
                        is_a_ghost_crossing,\
                        number_of_animals):

        self.identifier = fragment_identifier
        self.start_end = start_end
        self.blob_hierarchy_in_starting_frame = blob_hierarchy_in_starting_frame
        self.images = np.asarray(images)
        self.centroids = np.asarray(centroids)
        self.set_distance_travelled()
        self.areas = np.asarray(areas)
        self.pixels = pixels
        self.is_a_fish = is_a_fish
        self.is_a_crossing = is_a_crossing
        self.is_a_jump = is_a_jump
        self.is_a_jumping_fragment = is_a_jumping_fragment
        self.is_a_ghost_crossing = is_a_ghost_crossing
        self.number_of_animals = number_of_animals

    @property
    def number_of_images(self):
        return len(self.images)

    def set_distance_travelled(self):
        if self.centroids.shape[0] > 1:
            self.distance_travelled = np.sum(np.sqrt(np.sum(np.diff(self.centroids, axis = 0)**2, axis = 1)))
        else:
            self.distance_travelled = 0.

    def are_overlapping(self, other):
        (s1,e1), (s2,e2) = self.start_end, other.start_end
        # print("**********")
        # print((s1,e1), (s2,e2))
        # print(s1 < e2 and e1 > s2)
        return s1 < e2 and e1 > s2

    def get_coexisting_individual_fragments_indices(self, list_of_fragments):
        self.coexisting_individual_fragments = [fragment.identifier for fragment in list_of_fragments
                                            if fragment.is_a_fish and self.are_overlapping(fragment)
                                            and fragment is not self
                                            and self.is_a_fish]

def append_values_to_lists(values, list_of_lists):
    list_of_lists_updated = []

    for l, value in zip(list_of_lists, values):
        l.append(value)
        list_of_lists_updated.append(l)

    return list_of_lists_updated

def delete_attributes(blob, attributes_list):
    for attribute in attributes_list:
        setattr(blob, attribute, None)

def create_list_of_fragments(blobs_in_video, number_of_animals):
    attributes_to_delete_from_blob = ['_portrait', 'bounding_box_image', 'bounding_box_in_frame_coordinates'
                                        '_area', '_next', '_previous',]
    list_of_fragments = []
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
                list_of_fragments.append(fragment)

            delete_attributes(blob,attributes_to_delete_from_blob)

    [fragment.get_coexisting_individual_fragments_indices(list_of_fragments) for fragment in list_of_fragments]
    return list_of_fragments
