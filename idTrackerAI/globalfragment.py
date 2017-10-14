from __future__ import absolute_import, division, print_function
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import logging

logger = logging.getLogger("__main__.globalfragment")

class GlobalFragment(object):
    def __init__(self, list_of_blobs, fragments, index_beginning_of_fragment, number_of_animals):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.individual_fragments_identifiers = [blob.fragment_identifier for blob in list_of_blobs[index_beginning_of_fragment]]
        self.get_list_of_attributes_from_individual_fragments(fragments)
        self.set_minimum_distance_travelled()
        self.number_of_animals = number_of_animals
        self.reset(roll_back_to = 'fragmentation')
        self._is_unique = False
        self._is_certain = False

    def reset(self, roll_back_to):
        if roll_back_to == 'fragmentation':
            self._ids_assigned = np.nan * np.ones(self.number_of_animals)
            self._temporary_ids = np.arange(self.number_of_animals) # I initialize the _temporary_ids like this so that I can use the same function to extract images in pretraining and training
            self._score = None
            self._is_unique = False
            self._uniqueness_score = None
            self._repeated_ids = []
            self._missing_ids = []
            self.predictions = [] #stores predictions per portrait in self, organised according to individual fragments.
            self.softmax_probs_median = [] #stores softmax median per individual, per individual fragment

    def get_individual_fragments_of_global_fragment(self, fragments):
        self.individual_fragments = [fragment for fragment in fragments
                                        if fragment.identifier in self.individual_fragments_identifiers]

    def get_list_of_attributes_from_individual_fragments(self, fragments, list_of_attributes = ['distance_travelled', 'number_of_images']):
        [setattr(self,attribute + '_per_individual_fragment',[]) for attribute in list_of_attributes]
        for fragment in fragments:
            if fragment.identifier in self.individual_fragments_identifiers:
                assert fragment.is_a_fish
                setattr(fragment, '_is_in_a_global_fragment', True)
                for attribute in list_of_attributes:
                    getattr(self, attribute + '_per_individual_fragment').append(getattr(fragment, attribute))


    def set_minimum_distance_travelled(self):
        self.minimum_distance_travelled = min(self.distance_travelled_per_individual_fragment)

    def get_total_number_of_images(self):
        return sum([fragment.number_of_images for fragment in self.individual_fragments])

    # @property
    # def used_for_training(self):
    #     return self._used_for_training

    @property
    def uniqueness_score(self):
        return self._uniqueness_score

    def compute_uniqueness_score(self, P1_individual_fragments):
        """ Computes the distance of the assignation probabilities (P2) per
        individual fragment in the global fragment to the identity matrix of
        dimension number of animals.
        uniqueness_score = 0.0 means that every individual is assigned with
        certainty 1.0 once and only once in the global fragment
        """
        if not self._used_for_training and self.is_unique:
            identity = np.identity(self.number_of_animals)
            P1_mat = np.vstack(P1_individual_fragments) # stack P1 of each individual fragment in the global fragment into a matrix
            perm = np.argmax(P1_mat,axis=1) # get permutation that orders the matrix to match the identity matrix
            P1_mat = P1_mat[:,perm] # apply permutation
            print(P1_mat)
            self._uniqueness_score = np.linalg.norm(P1_mat - identity)

    @property
    def score(self):
        return self._score

    def compute_score(self, P1_individual_fragments, max_distance_travelled):
        if not self._used_for_training and self.is_unique:
            self.compute_uniqueness_score(P1_individual_fragments)
            self._score = self.uniqueness_score**2 + (max_distance_travelled - self.minimum_distance_travelled)**2

    @property
    def used_for_training(self):
        return all([fragment.used_for_training for fragment in self.individual_fragments])

    def acceptable_for_training(self, accumulation_strategy):
        if accumulation_strategy == 'global':
            return all([fragment.acceptable_for_training for fragment in self.individual_fragments])
        else:
            return any([fragment.acceptable_for_training for fragment in self.individual_fragments])

    @property
    def is_unique(self):
        self.check_uniqueness(scope = 'global')
        return self._is_unique

    @property
    def is_partially_unique(self):
        self.check_uniqueness(scope = 'partial')
        return self._is_partially_unique

    def check_uniqueness(self, scope):
        all_identities = range(self.number_of_animals)
        if scope == 'global':
            if len(set(all_identities) - set([fragment.temporary_id for fragment in self.individual_fragments])) > 0:
                self._is_unique = False
            else:
                self._is_unique = True
        elif scope == 'partial':
            identities_acceptable_for_training = [fragment.temporary_id for fragment in self.individual_fragments
                                                    if fragment.acceptable_for_training]
            self.duplicated_identities = set([x for x in identities_acceptable_for_training if identities_acceptable_for_training.count(x) > 1])
            if len(self.duplicated_identities) > 0:
                self._is_partially_unique = False
            else:
                self._is_partially_unique = True

    def get_total_number_of_images(self):
        if not hasattr(self,'total_number_of_images'):
            self.total_number_of_images = sum([fragment.number_of_images for fragment in self.individual_fragments])
        return self.total_number_of_images

    def get_images_and_labels(self):
        images = []
        labels = []

        for fragment in self.individual_fragments:
            images.extend(fragment.images)
            labels.extend([fragment.blob_hierarchy_in_starting_frame] * fragment.number_of_images)

        return images, labels

    def compute_start_end_frame_indices_of_individual_fragments(self, blobs_in_video):
        self.starts_ends_individual_fragments = [blob.compute_fragment_start_end()
            for blob in blobs_in_video[self.index_beginning_of_fragment]]

    def update_individual_fragments_attribute(self, attribute, value):
        [setattr(fragment, attribute, value) for fragment in self.individual_fragments]
