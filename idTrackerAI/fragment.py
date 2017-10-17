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
FIXED_IDENTITY_THRESHOLD = .9

class Fragment(object):
    def __init__(self, fragment_identifier = None,\
                        start_end = None,\
                        blob_hierarchy_in_starting_frame = None,\
                        images = None,\
                        centroids = None,\
                        areas = None,\
                        pixels = None,\
                        is_an_individual = None,\
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
        self.is_an_individual = is_an_individual
        self.is_a_crossing = is_a_crossing
        self.is_a_jump = is_a_jump
        self.is_a_jumping_fragment = is_a_jumping_fragment
        self.is_a_ghost_crossing = is_a_ghost_crossing
        self.number_of_animals = number_of_animals
        self.possible_identities = range(1, self.number_of_animals + 1)
        self._is_in_a_global_fragment = False
        self._used_for_training = False
        self._used_for_pretraining = False
        self._acceptable_for_training = None
        self._temporary_id = None
        self._identity = None
        self._identity_corrected_solving_duplication = None
        self._user_generated_identity = None
        self._identity_is_fixed = False
        self._accumulated_globally = False
        self._accumulated_partially = False
        self._accumulation_step = None

    def reset(self, roll_back_to = None):
        if roll_back_to == 'fragmentation' or roll_back_to == 'pretraining':
            self._used_for_training = False
            if roll_back_to == 'fragmentation': self._used_for_pretraining = False
            self._acceptable_for_training = None
            self._temporary_id = None
            self._identity = None
            self._user_generated_identity = None
            self._identity_corrected_solving_duplication = None
            self._identity_is_fixed = False
            self._accumulated_globally = False
            self._accumulated_partially = False
            self._accumulation_step = None
            attributes_to_delete = ['_frequencies',
                                    '_P1_vector', '_certainty',
                                    '_is_certain',
                                    '_P1_below_random', '_non_consistent',
                                    'assigned_during_accumulation']
            delete_attributes_from_object(self, attributes_to_delete)
        elif roll_back_to == 'accumulation':
            self._identity_is_fixed = False
            attributes_to_delete = []
            if not self.used_for_training:
                self._identity = None
                self._user_generated_identity = None
                self._identity_corrected_solving_duplication = None
                attributes_to_delete = ['_frequencies', '_P1_vector']
            attributes_to_delete.extend(['_P2_vector', '_ambiguous_identities',
                                        '_is_a_duplication'])
            delete_attributes_from_object(self, attributes_to_delete)
        elif roll_back_to == 'assignment':
            self._user_generated_identity = None
            self._identity_corrected_solving_duplication = None
            attributes_to_delete = ['_is_a_duplication']
            delete_attributes_from_object(self, attributes_to_delete)

    @property
    def is_in_a_global_fragment(self):
        return self._is_in_a_global_fragment

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def accumulated_globally(self):
        return self._accumulated_globally

    @property
    def accumulated_partially(self):
        return self._accumulated_partially

    @property
    def accumulation_step(self):
        return self._accumulation_step

    @property
    def accumulable(self):
        return self._accumulable

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
    def P2_vector(self):
        return self._P2_vector

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
    def identity_is_fixed(self):
        return self._identity_is_fixed

    @property
    def user_generated_identity(self):
        return self._user_generated_identity

    @property
    def final_identity(self):
        if hasattr(self, 'user_generated_identity') and self.user_generated_identity is not None:
            return self.user_generated_identity
        elif hasattr(self, 'identity_corrected_solving_duplication') and self.identity_corrected_solving_duplication is not None:
            return self.identity_corrected_solving_duplication
        else:
            return self.identity

    @property
    def identity_corrected_solving_duplication(self):
        return self._identity_corrected_solving_duplication

    @property
    def ambiguous_identities(self):
        return self._ambiguous_identities

    @property
    def potentially_randomly_assigned(self):
        return self._potentially_randomly_assigned

    @property
    def non_consistent(self):
        return self._non_consistent

    @property
    def is_a_duplication(self):
        return self._is_a_duplication

    def set_duplication_flag(self):
        if any([fragment.identity == self.identity for fragment in self.coexisting_individual_fragments
                if (fragment.is_an_individual and fragment.identity != 0)]):
            self._is_a_duplication = True
        else:
            self._is_a_duplication = False

    def get_attribute_of_coexisting_fragments(self, attribute):
        return [getattr(fragment,attribute) for fragment in self.coexisting_individual_fragments]

    def get_missing_identities_in_coexisting_fragments(self, fixed_identities):
        identities = self.get_attribute_of_coexisting_fragments('final_identity')
        identities = [identity for identity in identities if identity != 0]
        if not self.identity in fixed_identities:
            return list((set(self.possible_identities) - set(identities)) | set([self.identity]))
        else:
            return list(set(self.possible_identities) - set(identities))

    def get_fixed_identities_of_coexisting_fragments(self):
        return [fragment.final_identity for fragment in self.coexisting_individual_fragments
                if fragment.used_for_training
                or not fragment.is_a_duplication
                or fragment.user_generated_identity is not None
                or (fragment.identity_corrected_solving_duplication is not None
                and fragment.identity_corrected_solving_duplication != 0)]

    @property
    def number_of_images(self):
        return len(self.images)

    def set_distance_travelled(self):
        if self.centroids.shape[0] > 1:
            self.distance_travelled = np.sum(np.sqrt(np.sum(np.diff(self.centroids, axis = 0)**2, axis = 1)))
        else:
            self.distance_travelled = 0.

    def frame_by_frame_velocity(self):
        return np.sqrt(np.sum(np.diff(self.centroids, axis = 0)**2, axis = 1))

    def compute_border_velocity(self, other):
        centroids = np.asarray([self.centroids[0], other.centroids[-1]])
        if not self.start_end[0] > other.start_end[1]:
            centroids = np.asarray([self.centroids[-1],other.centroids[0]])
        return np.sqrt(np.sum(np.diff(centroids, axis = 0)**2, axis = 1))[0]

    def are_overlapping(self, other):
        (s1,e1), (s2,e2) = self.start_end, other.start_end
        return s1 < e2 and e1 > s2

    def get_coexisting_individual_fragments_indices(self, fragments):
        self.coexisting_individual_fragments = [fragment for fragment in fragments
                                            if fragment.is_an_individual and self.are_overlapping(fragment)
                                            and fragment is not self
                                            and self.is_an_individual]
        self.number_of_coexisting_individual_fragments = len(self.coexisting_individual_fragments)

    @property
    def has_enough_accumulated_coexisting_fragments(self):
        return sum([fragment.used_for_training
                    for fragment in self.coexisting_individual_fragments]) >= self.number_of_coexisting_individual_fragments/2

    def check_consistency_with_coexistent_individual_fragments(self, temporary_id):
        for coexisting_fragment in self.coexisting_individual_fragments:
            if coexisting_fragment.temporary_id == temporary_id:
                return False
        return True

    def compute_identification_statistics(self, predictions, softmax_probs):
        assert self.is_an_individual
        self._frequencies = self.compute_identification_frequencies_individual_fragment(predictions, self.number_of_animals)
        self._P1_vector = self.compute_P1_from_frequencies(self.frequencies)
        median_softmax = self.compute_median_softmax(softmax_probs, self.number_of_animals)
        self._certainty = self.compute_certainty_of_individual_fragment(self._P1_vector,median_softmax)

    @staticmethod
    def get_possible_identities(P2_vector):
        """Check if P2 has two identical maxima. In that case returns the indices.
        Else return false.
        """
        maxima_indices = np.where(P2_vector == np.max(P2_vector))[0]
        return maxima_indices + 1, np.max(P2_vector)

    def assign_identity(self, recompute = True):
        assert self.is_an_individual
        self.compute_P2_vector()
        if self.used_for_training and not self._identity_is_fixed:
            self._identity_is_fixed = True
        elif not self._identity_is_fixed:
            possible_identities, max_P2 = self.get_possible_identities(self.P2_vector)
            if len(possible_identities) > 1:
                self._identity = 0
                self._ambiguous_identities = possible_identities
            else:
                if max_P2 > FIXED_IDENTITY_THRESHOLD:
                    self._identity_is_fixed = True
                self._identity = possible_identities[0]
        if recompute:
            self.recompute_P2_of_coexisting_fragments()

    def recompute_P2_of_coexisting_fragments(self):
        # The P2 of fragments with fixed identity won't be recomputed
        # due to the condition in assign_identity() (second line)
        [fragment.assign_identity(recompute = False) for fragment in self.coexisting_individual_fragments]

    def compute_P2_vector(self):
        coexisting_P1_vectors = np.asarray([fragment.P1_vector for fragment in self.coexisting_individual_fragments])
        numerator = np.asarray(self.P1_vector) * np.prod(1. - coexisting_P1_vectors, axis = 0)
        denominator = np.sum(numerator)
        if denominator == 0:
            self._P2_vector = self.P1_vector
        else:
            self._P2_vector = numerator / denominator

    @staticmethod
    def compute_identification_frequencies_individual_fragment(predictions, number_of_animals):
        return np.asarray([np.sum(predictions == i)
                            for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1

    @staticmethod
    def compute_P1_from_frequencies(frequencies):
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
        #jumps are fragment composed by a single image, thus:
        if len(softmax_probs.shape) == 1:
            softmax_probs = np.expand_dims(softmax_probs, axis = 1)
        max_softmax_probs = np.max(softmax_probs, axis = 1)
        argmax_softmax_probs = np.argmax(softmax_probs, axis = 1)
        softmax_median = np.zeros(number_of_animals)
        for i in np.unique(argmax_softmax_probs):
            softmax_median[i] = np.median(max_softmax_probs[argmax_softmax_probs == i])
        return softmax_median

    @staticmethod
    def compute_certainty_of_individual_fragment(P1_vector, median_softmax):
        argsort_p1_vector = np.argsort(P1_vector)
        sorted_p1_vector = P1_vector[argsort_p1_vector]
        sorted_softmax_probs = median_softmax[argsort_p1_vector]
        certainty = np.diff(np.multiply(sorted_p1_vector,sorted_softmax_probs)[-2:])/np.sum(sorted_p1_vector[-2:])
        return certainty[0]

    def get_neighbour_fragment(self, fragments, scope):
        if scope == 'to_the_past':
            neighbour = [fragment for fragment in fragments
                            if fragment.final_identity == self.final_identity
                            and self.start_end[0] - fragment.start_end[1] == 1]
        elif scope == 'to_the_future':
            neighbour = [fragment for fragment in fragments
                            if fragment.final_identity == self.final_identity
                            and fragment.start_end[0] - self.start_end[1] == 1]

        assert len(neighbour) < 2
        return neighbour[0] if len(neighbour) == 1 else None

    def set_partially_or_globally_accumualted(self, accumulation_strategy):
        if accumulation_strategy == 'global':
            self._accumulated_globally = True
        elif accumulation_strategy == 'partial':
            self._accumulated_partially = True
