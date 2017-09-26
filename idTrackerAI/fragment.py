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
    def __init__(self, fragment_identifier = None,\
                        start_end = None,\
                        blob_hierarchy_in_starting_frame = None,\
                        images = None,\
                        centroids = None,\
                        areas = None,\
                        pixels = None,\
                        is_a_fish = None,\
                        is_a_crossing = None,\
                        is_a_jump = None,\
                        is_a_jumping_fragment = None,\
                        is_a_ghost_crossing = None,\
                        number_of_animals = None):

        self.identifier = fragment_identifier
        self.start_end = start_end
        self.blob_hierarchy_in_starting_frame = blob_hierarchy_in_starting_frame
        self.images = images
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
        self._used_for_training = False
        self._acceptable_for_training = True
        self._is_certain = False
        self._temporary_id = None

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def acceptable_for_training(self):
        return self._acceptable_for_training

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def P1_vector(self):
        return self._P1_vector

    @property
    def certainty(self):
        return self._certainty

    @property
    def is_certain(self):
        return self._is_certain

    @property
    def temporary_id(self):
        return self._temporary_id

    # def reset(self):
    #     self._frequencies_in_fragment = np.zeros(self.number_of_animals).astype('int')
    #     self.P1_vector = np.zeros(self.number_of_animals)
    #     self.P2_vector = np.zeros(self.number_of_animals)
    #     self.assigned_during_accumulation = False
    #     self.used_for_training = False
    #     self.user_generated_identity = None #in the validation part users can correct manually the identities
    #     self.identity = None
    #     self.identity_corrected_solving_duplication = None
    #     self.user_generated_centroids = []
    #     self.user_generated_identities = []

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

    def compute_identification_statistics(self, predictions, softmax_probs):
        self._frequencies = self.compute_identification_frequencies_individual_fragment(predictions, self.number_of_animals)
        self._P1_vector = self.compute_P1_individual_fragment_from_frequencies(self.frequencies)
        median_softmax = self.compute_median_softmax(softmax_probs, self.number_of_animals)
        self._certainty = self.compute_certainty_of_individual_fragment(self._P1_vector,median_softmax)

    @staticmethod
    def compute_identification_frequencies_individual_fragment(predictions, number_of_animals):
        return np.asarray([np.sum(predictions == i)
                            for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1

    @staticmethod
    def compute_P1_individual_fragment_from_frequencies(frequencies):
        """Given the frequencies of a individual fragment
        computer the P1 vector. P1 is the softmax of the frequencies with base 2
        for each identity.
        """
        # Compute numerator of P1 and check that it is not inf
        numerator = 2.**frequencies
        if np.any(numerator == np.inf):
            numerator[numerator == np.inf] = MAX_FLOAT
        # Compute denominator of P1
        denominator = np.sum(numerator)
        # Compute P1 and check that it is not 0. for any identity
        P1_of_fragment = numerator / denominator
        if np.all(P1_of_fragment == 0.):
            P1_of_fragment[P1_of_fragment == 0.] = 1/len(P1_of_fragment) #if all the frequencies are very high then the denominator is very big and all the P1 are 0. so we set then to random.
        else:
            P1_of_fragment[P1_of_fragment == 0.] = MIN_FLOAT
        # Change P1 that are 1. for 0.9999 so that we do not have problems when computing P2
        # P1_of_fragment[P1_of_fragment == 1.] = 1. - MIN_FLOAT
        # P1_of_fragment = P1_of_fragment / np.sum(P1_of_fragment)
        P1_of_fragment[P1_of_fragment == 1.] = 0.999999999999
        return P1_of_fragment

    @staticmethod
    def compute_median_softmax(softmax_probs, number_of_animals):
        softmax_probs = np.asarray(softmax_probs)
        max_softmax_probs = np.max(softmax_probs, axis = 1)
        argmax_softmax_probs = np.argmax(softmax_probs, axis = 1)
        softmax_median = np.zeros(number_of_animals)
        for i in np.unique(argmax_softmax_probs):
            softmax_median[i] = np.median(max_softmax_probs[argmax_softmax_probs==i])
        return softmax_median

    @staticmethod
    def compute_certainty_of_individual_fragment(P1_vector, median_softmax):
        argsort_p1_vector = np.argsort(P1_vector)
        sorted_p1_vector = P1_vector[argsort_p1_vector]
        sorted_softmax_probs = median_softmax[argsort_p1_vector]
        certainty = np.diff(np.multiply(sorted_p1_vector,sorted_softmax_probs)[-2:])/np.sum(sorted_p1_vector[-2:])
        return certainty

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
    fragment_identifier_to_index = np.argsort([fragment.identifier for fragment in list_of_fragments])
    return list_of_fragments, fragment_identifier_to_index

def reset_fragments(list_of_fragments, recovering_from = 'accumulation'):
    [fragment.reset(recovering_from) for fragment in list_of_fragments]
