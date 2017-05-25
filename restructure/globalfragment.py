from __future__ import absolute_import, division, print_function
import numpy as np
import random

from blob import is_a_global_fragment, check_global_fragments
from statistics_for_assignment import compute_identification_frequencies_individual_fragment, compute_P1_individual_fragment_from_blob

STD_TOLERANCE = 1 ### NOTE set to 1 because we changed the model area to work with the median.

def detect_beginnings(boolean_array):
    """ detects the frame where the core of a global fragment starts.
    A core of a global fragment is the part of the global fragment where all the
    individuals are visible, i.e. the number of animals in the frame equals the
    number of animals in the video
    :boolean_array: array with True where the number of animals in the frame equals
    the number of animals in the video
    """
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def compute_model_area(blobs_in_video, number_of_animals, std_tolerance = STD_TOLERANCE):
    """computes the median and standard deviation of all the blobs of the video.
    These values are later used to discard blobs that are not fish and potentially
    belong to a crossing.
    """
    areas = [blob.area for blobs_in_frame in blobs_in_video for blob in blobs_in_frame ]
    media_area = np.median(areas)
    std_area = np.std(areas)
    return ModelArea(media_area, std_area)

class ModelArea(object):
  def __init__(self, median, std):
    self.median = median
    self.std = std

  def __call__(self, area, std_tolerance = STD_TOLERANCE):
    return (area - self.median) < std_tolerance * self.std

class GlobalFragment(object):
    def __init__(self, list_of_blobs, index_beginning_of_fragment, number_of_animals):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.min_distance_travelled = np.min([blob.distance_travelled_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ])
        self.individual_fragments_identifiers = [blob.fragment_identifier for blob in list_of_blobs[index_beginning_of_fragment]]
        self.portraits = [blob.portraits_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment]]
        self.number_of_animals = number_of_animals
        self._used_for_training = False
        self._acceptable_for_training = True
        self._ids_assigned = np.nan * np.ones(self.number_of_animals)
        self._temporary_ids = np.arange(self.number_of_animals) # I initialize the _ids_assigned like this so that I can use the same function to extract images in pretraining and training
        self._score = None
        self._is_unique = False
        self._uniqueness_score = None
        self._repeated_ids = []
        self._missing_ids = []
        self._number_of_portraits_per_individual_fragment = [len(portraits_in_individual_fragment)
                        for portraits_in_individual_fragment in self.portraits] # length of the portraits contained in each individual fragment part of the global fragment
        self._total_number_of_portraits = np.sum(self._number_of_portraits_per_individual_fragment) #overall number of portraits
        self.predictions = [] #stores predictions per portrait in self, organised according to individual fragments.
        self.softmax_probs_median = [] #stores softmax median per individual, per individual fragment

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def acceptable_for_training(self):
        return self._acceptable_for_training

    @property
    def uniqueness_score(self):
        return self._uniqueness_score

    def compute_uniqueness_score(self, P1_individual_fragments):
        """ Computes the distance of the assignation probabilities (P2) per
        individual fragment in the global fragment to the identity matrix of
        dimension number of animals.
        uniqueness_score = 0.0 means that every individual is assigned with
        certainty 1.0 one and only once in the global fragment
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
            self._score = self.uniqueness_score**2 + (max_distance_travelled - self.min_distance_travelled)**2

    @property
    def is_unique(self):
        self.check_uniqueness()
        return self._is_unique

    def check_uniqueness(self):
        if not self._used_for_training:
            all_identities = range(self.number_of_animals)
            if set(all_identities).difference(set(self._temporary_ids)):
                self._is_unique = False
                self.compute_repeated_and_missing_ids(all_identities)
            else:
                self._is_unique = True

    def compute_repeated_and_missing_ids(self, all_identities):
        self._repeated_ids = set([x for x in self._ids_assigned if list(self._ids_assigned).count(x) > 1])
        self._missing_ids = set(all_identities).difference(set(self._ids_assigned))

def order_global_fragments_by_distance_travelled(global_fragments):
    global_fragments = sorted(global_fragments, key = lambda x: x.min_distance_travelled, reverse = True)
    return global_fragments

def give_me_identities_of_global_fragment(global_fragment, blobs_in_video):
    global_fragment._ids_assigned = [blob.identity
        for blob in blobs_in_video[global_fragment.index_beginning_of_fragment]]

def give_me_list_of_global_fragments(blobs_in_video, num_animals):
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video,i,num_animals) for i in indices_beginning_of_fragment]

def give_me_pre_training_global_fragments(global_fragments, number_of_global_fragments = 10):
    step = len(global_fragments) // number_of_global_fragments
    split_global_fragments = [global_fragments[i:i + step] for i in range(0, len(global_fragments)-1, step)]
    ordered_split_global_fragments = [order_global_fragments_by_distance_travelled(global_fragments_in_split)[0]
                                    for global_fragments_in_split in split_global_fragments]
    return ordered_split_global_fragments

def get_images_and_labels_from_global_fragment(global_fragment, individual_fragments_identifiers_already_used = []):
    if not np.isnan(global_fragment._ids_assigned).any() and list(global_fragment._temporary_ids) != list(global_fragment._ids_assigned -1):
        raise ValueError("Temporary ids and assigned ids should match in global fragments used for training")
    images = []
    labels = []
    lengths = []
    individual_fragments_identifiers = []
    for i, portraits in enumerate(global_fragment.portraits):
        if global_fragment.individual_fragments_identifiers[i] not in individual_fragments_identifiers_already_used:
            print("This individual fragment has not been used, we take images")
            images.extend(portraits)
            labels.extend([global_fragment._temporary_ids[i]]*len(portraits))
            lengths.append(len(portraits))
            individual_fragments_identifiers.append(global_fragment.individual_fragments_identifiers[i])
    return images, labels, lengths, individual_fragments_identifiers

def get_images_and_labels_from_global_fragments(global_fragments, individual_fragments_identifiers_already_used = []):
    images = []
    labels = []
    lengths = []
    for global_fragment in global_fragments:
        print("\ngetting images from global fragment")
        # print(individual_fragments_identifiers_already_used)
        images_global_fragment, labels_global_fragment, lengths_global_fragment, individual_fragments_identifiers = get_images_and_labels_from_global_fragment(global_fragment, individual_fragments_identifiers_already_used)
        # print("len(images_global_fragment) ", len(images_global_fragment))
        # print("len(labels_global_fragment) ", len(labels_global_fragment))
        # print("sum(lengths_global_fragment) ", sum(lengths_global_fragment))
        # print("len(lengths_global_fragment) ", len(lengths_global_fragment))
        # print("len(individual_fragments_identifiers) ", len(individual_fragments_identifiers))
        if len(images_global_fragment) != 0:
            images.append(images_global_fragment)
            labels.append(labels_global_fragment)
            lengths.extend(lengths_global_fragment)
            individual_fragments_identifiers_already_used.extend(individual_fragments_identifiers)
    # print("len(images) ", len(np.concatenate(images, axis = 0)))
    # print("len(labels) ", len(np.concatenate(labels, axis = 0)))
    # print("sum(lengths) ", sum(lengths))
    # print("len(lengths) ", len(lengths))
    # print("len(individual_fragments_identifiers_already_used)", len(individual_fragments_identifiers_already_used))

    return np.concatenate(images, axis = 0), np.concatenate(labels, axis = 0), individual_fragments_identifiers_already_used, np.cumsum(lengths)[:-1]

def subsample_images_for_last_training(images, labels, number_of_animals, number_of_samples = 3000):
    """Before assigning identities to the blobs that are not part of the training set we train the network
    a last time with uncorrelated images from the training set of references. This function subsample the
    entire set of training images to get a 'number_of_samples' balanced set.
    """
    subsampled_images = []
    subsampled_labels = []
    for i in np.unique(labels):
        subsampled_images.append(random.sample(images[np.where(labels == i)[0]],number_of_samples))
        subsampled_labels.append([i] * number_of_samples)
    return np.concatenate(subsampled_images, axis = 0), np.concatenate(subsampled_labels, axis = 0)
