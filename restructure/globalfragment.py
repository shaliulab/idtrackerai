from __future__ import absolute_import, division, print_function
import numpy as np

from blob import is_a_global_fragment, check_global_fragments

STD_TOLERANCE = 4

def detect_beginnings(boolean_array):
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def compute_model_area(blobs_in_video, number_of_animals, std_tolerance = STD_TOLERANCE):
    blobs_in_core_global_fragments = [blobs_in_frame for blobs_in_frame in blobs_in_video if is_a_global_fragment(blobs_in_frame, number_of_animals)]
    areas = [blob.area for blob in blobs_in_frame for blobs_in_frame in blobs_in_core_global_fragments]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    return ModelArea(mean_area, std_area)

class ModelArea():
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, area, std_tolerance = STD_TOLERANCE):
    return (area - self.mean) < std_tolerance * self.std

class GlobalFragment(object):
    def __init__(self, list_of_blobs, index_beginning_of_fragment, number_of_animals):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.min_distance_travelled = np.min([blob.distance_travelled_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ])

        self.portraits = [blob.portraits_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ]
        self.number_of_animals = number_of_animals
        self._used_for_training = False
        self._ids_assigned = [None] * self.number_of_animals
        self._score = None
        self._is_unique = False
        self._uniqueness_score = None
        self._repeated_ids = []
        self._missing_ids = []


    @property
    def uniqueness_score(self):
        return self._uniqueness_score

    @uniqueness_score.setter
    def uniqueness_score(self):
        """ Computes the distance of the assignation probabilities (P2) per
        individual fragment in the global fragment to the identity matrix of
        dimension number of animals.
        uniqueness_score = 0.0 means that every individual is assigned with
        certainty 1.0 one and only once in the global fragment
        """
        if self.is_unique and not self._used_for_training:
            identity = np.identity(self.number_of_animals)
            P2_mat = np.vstack(self.P2_list) # stack P2 of each individual fragment in the global fragment into a matrix
            perm = np.argmax(P2_mat,axis=1) # get permutation that orders the matrix to match the identity matrix
            P2_mat = P2_mat[:,perm] # apply permutation
            self._uniqueness_score = np.linalg.norm(matFragment - identity)

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, best_uniqueness_score, max_distance_travelled):
        if not self._used_for_training:
            self._score = (best_uniqueness_score - self.uniqueness_score)**2 + ((max_distance_travelled - self.min_distance_travelled))**2

    @property
    def is_unique(self):
        return self._is_unique

    @is_unique.setter
    def is_unique(self):
        if not self._used_for_training:
            all_identities = range(self.number_of_animals)
            if set(all_identities).difference(set(self._ids_assigned)):
                self._is_unique = False
                self.compute_repeated_and_missing_ids()
            else:
                self._is_unique = True

    def compute_repeated_and_missing_ids(self):
        self._repeated_ids = set([x for x in self._ids_assigned if self._ids_assigned.count(x) > 1])
        self._missing_ids = set(all_identities).difference(set(self._ids_assigned))

def give_me_identities_of_global_fragment(global_fragment,list_of_blobs):
    global_fragment._ids_assigned = [blob.identity
        for blob in list_of_blobs[global_fragment.index_beginning_of_fragment] ]

def give_me_list_of_global_fragments(list_of_blobs, num_animals):
    global_fragments_boolean_array = check_global_fragments(list_of_blobs, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(list_of_blobs,i,num_animals) for i in indices_beginning_of_fragment]

def order_global_fragments_by_distance_travelled(global_fragments):
    global_fragments = sorted(global_fragments, key = lambda x: x.min_distance_travelled, reverse = True)

def give_me_pre_training_global_fragments(global_fragments, number_of_global_fragments = 10):
    step = int(np.floor(len(global_fragments) / number_of_global_fragments))
    split_global_fragments = [global_fragments[i:i + step] for i in range(0, len(global_fragments), step)]
    ordered_split_global_fragments = [order_global_fragments_by_distance_travelled(global_fragments_in_split)[0]
                                    for global_fragments_in_split in split_global_fragments]
