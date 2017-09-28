from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')

import itertools
import numpy as np
from tqdm import tqdm
import logging
from math import sqrt

from py_utils import append_values_to_lists, delete_attributes_from_object

logger = logging.getLogger("__main__.fragment")

MAX_FLOAT = sys.float_info[0]
MIN_FLOAT = sys.float_info[3]

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
        self._used_for_pretraining = False
        self._acceptable_for_training = True


    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def used_for_pretraining(self):
        return self._used_for_pretraining

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

    @property
    def identity(self):
        return self._identity

    @property
    def potentially_randomly_assigned(self):
        return self._potentially_randomly_assigned

    @property
    def non_consistent(self):
        return self._non_consistent

    def reset(self, roll_back_to = None):
        if roll_back_to == 'fragmentation':
            self._used_for_training = False
            self._used_for_pretraining = False
            self._acceptable_for_training = True
            attributes_to_delete = ['_frequencies_in_fragment',
                                    '_P1_vector', '_certainty',
                                    '_temporary_id', '_identity',
                                    '_is_certain', '_P1_below_random',
                                    '_non_consistent',
                                    'assigned_during_accumulation']
            delete_attributes_from_object(self, attributes_to_delete)

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
        return s1 < e2 and e1 > s2

    def get_coexisting_individual_fragments_indices(self, list_of_fragments):
        self.coexisting_individual_fragments = [fragment for fragment in list_of_fragments
                                            if fragment.is_a_fish and self.are_overlapping(fragment)
                                            and fragment is not self
                                            and self.is_a_fish]

    def check_consistency_with_coexistent_individual_fragments(self, temporary_id):
        for coexisting_fragment in self.coexisting_individual_fragments:
            if coexisting_fragment.temporary_id == temporary_id:
                return False
        return True

    def compute_identification_statistics(self, predictions, softmax_probs):
        assert self.is_a_fish
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
        print("********************* certainty ", certainty)
        return certainty[0]
