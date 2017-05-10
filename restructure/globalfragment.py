from __future__ import absolute_import, division, print_function
import numpy as np

from blob import is_a_global_fragment, check_global_fragments
from statistics_for_assignment import compute_identification_frequencies_individual_fragment, compute_P1_individual_fragment_from_blob

STD_TOLERANCE = 1 ### NOTE set to 1 because we changed the model area to work with the median.

def detect_beginnings(boolean_array):
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def compute_model_area(blobs_in_video, number_of_animals, std_tolerance = STD_TOLERANCE):
    # blobs_in_core_global_fragments = [blobs_in_frame for blobs_in_frame in blobs_in_video if is_a_global_fragment(blobs_in_frame, number_of_animals)]=
    areas = [blob.area for blobs_in_frame in blobs_in_video for blob in blobs_in_frame ]
    media_area = np.median(areas)
    std_area = np.std(areas)
    return ModelArea(media_area, std_area)

class ModelArea():
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

        self.portraits = [blob.portraits_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ]
        self.number_of_animals = number_of_animals
        self._used_for_training = False
        self._ids_assigned = [None] * self.number_of_animals
        self._temporary_ids = range(self.number_of_animals) # I initialize the _ids_assigned like this so that I can use the same function to extract images in pretraining and training
        self._score = None
        self._is_unique = False
        self._uniqueness_score = None
        self._repeated_ids = []
        self._missing_ids = []
        self._number_of_portraits_per_individual_fragment = [len(portraits_in_individual_fragment)
                        for portraits_in_individual_fragment in self.portraits] # length of the portraits contained in each individual fragment part of the global fragment
        self._total_number_of_portraits = np.sum(self._number_of_portraits_per_individual_fragment) #overall number of portraits
        self.predictions = [] #store predictions per portrait in self, organised according to individual fragments.
    @property
    def used_for_training(self):
        return self._used_for_training

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

    def is_unique(self):
        if not self._used_for_training:
            all_identities = range(1,self.number_of_animals+1)
            if set(all_identities).difference(set(self._temporary_ids)):
                self._is_unique = False
                self.compute_repeated_and_missing_ids(all_identities)
            else:
                self._is_unique = True
        return self._is_unique

    def compute_repeated_and_missing_ids(self, all_identities):
        self._repeated_ids = set([x for x in self._ids_assigned if self._ids_assigned.count(x) > 1])
        self._missing_ids = set(all_identities).difference(set(self._ids_assigned))

def give_me_identities_of_global_fragment(global_fragment, blobs_in_video):
    global_fragment._ids_assigned = [blob.identity
        for blob in blobs_in_video[global_fragment.index_beginning_of_fragment] ]

def give_me_list_of_global_fragments(blobs_in_video, num_animals):
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video,i,num_animals) for i in indices_beginning_of_fragment]

def order_global_fragments_by_distance_travelled(global_fragments):
    global_fragments = sorted(global_fragments, key = lambda x: x.min_distance_travelled, reverse = True)
    return global_fragments

def give_me_pre_training_global_fragments(global_fragments, number_of_global_fragments = 10):
    step = len(global_fragments) // number_of_global_fragments
    split_global_fragments = [global_fragments[i:i + step] for i in range(0, len(global_fragments)-1, step)]
    ordered_split_global_fragments = [order_global_fragments_by_distance_travelled(global_fragments_in_split)[0]
                                    for global_fragments_in_split in split_global_fragments]
    return ordered_split_global_fragments

def get_images_and_labels_from_global_fragment(global_fragment):
    images = [global_fragment.portraits[i] for i in range(len(global_fragment.portraits))]
    labels = [[id_]*len(images[i]) for i, id_ in enumerate(global_fragment._temporary_ids)]
    images = [im for ims in images for im in ims]
    labels = [lab for labs in labels for lab in labs]
    return images, labels

def assign_identity_to_global_fragment_used_for_training(global_fragment, blobs_in_video):
    """Assign the identities in identities_list to both global fragment and all its blobs if
    after training on the global fragment"""
    assert global_fragment.used_for_training == True
    global_fragment._ids_assigned = global_fragment._temporary_ids
    [blob.update_identity_in_fragment(identity_in_fragment)
        for blob, identity_in_fragment in zip(blobs_in_video[global_fragment.index_beginning_of_fragment], global_fragment._ids_assigned)]

def get_images_from_test_global_fragments(global_fragments):
    """stack all the images in global fragments if they have not been used
    for training. Optimised for GPU computing"""
    print("\nstacking images of global fragment for the GPU")
    return np.concatenate([np.concatenate(global_fragment.portraits, axis = 0)
            for global_fragment in global_fragments
            if not global_fragment.used_for_training], axis = 0)

def split_predictions_after_network_assignment(global_fragments, predictions):
    """Go back to the CPU"""
    print("\nun-stacking images for the CPU")
    number_of_portraits_per_global_fragment = [global_fragment._total_number_of_portraits
        for global_fragment in global_fragments
        if not global_fragment.used_for_training]
    print("Number of portraits per global fragment, ", number_of_portraits_per_global_fragment)
    print("length, ", len(number_of_portraits_per_global_fragment))
    print("sum, ", np.sum(number_of_portraits_per_global_fragment))
    predictions_per_global_fragments = np.split(predictions, np.cumsum(number_of_portraits_per_global_fragment)[:-1])
    print("\npredictions shape before splitting in ind frags",len(predictions_per_global_fragments))
    print("number of global fragments, ", len(global_fragments))
    c = 0
    for global_fragment in global_fragments:
        if not global_fragment.used_for_training:
            global_fragment.predictions = np.split(predictions_per_global_fragments[c], np.cumsum(global_fragment._number_of_portraits_per_individual_fragment)[:-1])
            print("\nnumber of portraits per individual fragment ", global_fragment._number_of_portraits_per_individual_fragment)
            print("total number of portraits in global fragment ", global_fragment._total_number_of_portraits)
            print("predictions shape per global fragment ", [len(indiv_frag_prediction) for indiv_frag_prediction in global_fragment.predictions])
            # print("portraits shape per global fragment ", global_fragment.portraits)
            # break
            c += 1

def assign_identities_to_test_global_fragment(global_fragment, number_of_animals):
    assert global_fragment.used_for_training == False
    global_fragment._temporary_ids = []
    for individual_fragment_predictions in global_fragment.predictions:
        # compute statistcs
        identities_in_fragment = np.asarray(individual_fragment_predictions)
        frequencies_in_fragment = compute_identification_frequencies_individual_fragment(identities_in_fragment, number_of_animals)
        P1_of_fragment = compute_P1_individual_fragment_from_blob(frequencies_in_fragment)
        # Assign identity to the fragment
        identity_in_fragment = np.argmax(P1_of_fragment) + 1
        global_fragment._temporary_ids.append(identity_in_fragment)
    global_fragment.is_unique()
    print("temp ids ", global_fragment._temporary_ids)
    print("the global fragment is unique: ", global_fragment._is_unique)

def assign_identities_to_all_test_global_fragments(global_fragments, number_of_animals):
    """Assigns identities during test to blobs in global fragments"""
    for global_fragment in global_fragments:
        if global_fragment.used_for_training == False:
            print("\nnumber of portraits per individual fragment ", global_fragment._number_of_portraits_per_individual_fragment)
            print("total number of portraits in global fragment ", global_fragment._total_number_of_portraits)
            assign_identities_to_test_global_fragment(global_fragment, number_of_animals)
