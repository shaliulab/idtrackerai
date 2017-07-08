from __future__ import absolute_import, division, print_function
import numpy as np
import random
from globalfragment import get_images_and_labels_from_global_fragments, order_global_fragments_by_distance_travelled
from statistics_for_assignment import compute_P1_individual_fragment_from_frequencies, compute_identification_frequencies_individual_fragment

RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = 0.1 # threshold to select a individual fragment as eligible for training

###
random.seed(0)
###

class AccumulationManager(object):
    def __init__(self,global_fragments, number_of_animals, certainty_threshold = CERTAINTY_THRESHOLD):
        """ This class manages the selection of global fragments for accumulation,
        the retrieval of images from the new global fragments, the selection of
        of images for training, the final assignment of identities to the global fragments
        used for training, the temporal assignment of identities to the candidates global fragments
        and the computation of the certainty levels of the individual fragments to check the
        eligability of the candidates global fragments for accumulation.

        :global_fragments list: list of global_fragments objects from the video
        :number_of_animals param: number of animals in the video
        :certainty_threshold param: threshold to set a individual fragment as certain for accumulation
        """
        self.number_of_animals = number_of_animals
        self.global_fragments = global_fragments
        self.counter = 0
        self.certainty_threshold = certainty_threshold
        self.individual_fragments_used = [] # list with the individual_fragments_identifiers of the individual fragments used for training
        self.identities_of_individual_fragments_used = [] # identities of the individual fragments used for training
        self.used_images = None # images used for training the network
        self.used_labels = None # labels for the images used for training
        self.new_images = None # set of images that will be added to the new training
        self.new_labels = None # labels for the set of images that will be added for training
        self._continue_accumulation = True # flag to continue_accumulation or not


    @property
    def continue_accumulation(self):
        """ We stop the accumulation when there are not more global fragments
        that are acceptable for training"""
        if not any([global_fragment.acceptable_for_training for global_fragment in self.global_fragments]):
            return False
        else:
            return True

    def update_counter(self):
        """ updates the counter of the accumulation"""
        self.counter += 1

    def get_next_global_fragments(self):
        """ get the global fragments that are going to be added to the current
        list of global fragments used for training"""
        if self.counter == 0:
            print("\nGetting global fragment for the first accumulation...")
            self.next_global_fragments = [order_global_fragments_by_distance_travelled(self.global_fragments)[0]]
        else:
            print("\nGetting global fragments...")
            self.next_global_fragments = [global_fragment for global_fragment in self.global_fragments
                                                if global_fragment.acceptable_for_training == True and global_fragment._used_for_training == False]
        print("Number of global fragments for training, ", len(self.next_global_fragments))

    def get_new_images_and_labels(self):
        """ get the images and labels of the new global fragments that are going
        to be used for training, this function checks whether the images of a individual
        fragment have been added before"""
        self.new_images, self.new_labels, _, _ = get_images_and_labels_from_global_fragments(self.next_global_fragments,list(self.individual_fragments_used))
        print("New images for training:", self.new_images.shape, self.new_labels.shape)
        if self.used_images is not None:
            print("Old images for training:", self.used_images.shape, self.used_labels.shape)

    def get_images_and_labels_for_training(self):
        """ We limit the number of images per animal that are used for training
        to MAXIMAL_IMAGES_PER_ANIMAL. We take a percentage RATIO_NEW of the new
        images and a percentage RATIO_OLD of the images already used. """
        print("Getting images for training...")
        images = []
        labels = []
        for i in range(self.number_of_animals):
            new_images_indices = np.where(self.new_labels == i)[0]
            used_images_indices = np.where(self.used_labels == i)[0]
            number_of_new_images = len(new_images_indices)
            number_of_used_images = len(used_images_indices)
            number_of_images_for_individual = number_of_new_images + number_of_used_images
            if number_of_images_for_individual > MAXIMAL_IMAGES_PER_ANIMAL:
                # we take a proportion of the old images a new images only if the
                # total number of images for this label is bigger than the limit MAXIMAL_IMAGES_PER_ANIMAL
                number_samples_new = int(MAXIMAL_IMAGES_PER_ANIMAL * RATIO_NEW)
                number_samples_used = int(MAXIMAL_IMAGES_PER_ANIMAL * RATIO_OLD)
                if number_of_used_images < number_samples_used:
                    # if the proportion of used images is bigger than the number of
                    # used images we take all the used images for this label and update
                    # the number of new images to reach the MAXIMAL_IMAGES_PER_ANIMAL
                    number_samples_used = number_of_used_images
                    number_samples_new = MAXIMAL_IMAGES_PER_ANIMAL - number_samples_used
                if number_of_new_images < number_samples_new:
                    # if the proportion of new images is bigger than the number of
                    # new images we take all the new images for this label and update
                    # the number of used images to reac the MAXIMAL_IMAGES_PER_ANIMAL
                    number_samples_new = number_of_new_images
                    number_samples_used = MAXIMAL_IMAGES_PER_ANIMAL - number_samples_new

                # we put together a random sample of the new images and the used images
                images.extend(random.sample(self.new_images[new_images_indices],number_samples_new))
                labels.extend([i] * number_samples_new)
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate the variable used_images is None
                    images.extend(random.sample(self.used_images[used_images_indices],number_samples_used))
                    labels.extend([i] * number_samples_used)
            else:
                # if the total number of images for this label does not exceed the MAXIMAL_IMAGES_PER_ANIMAL
                # we take all the new images and all the used images
                images.extend(self.new_images[new_images_indices])
                labels.extend([i] * number_of_new_images)
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate the variable used_images is None
                    images.extend(self.used_images[used_images_indices])
                    labels.extend([i] * number_of_used_images)

        return np.asarray(images), np.asarray(labels)

    def update_global_fragments_used_for_training(self):
        """ Once a global fragment has been used for training we set the flags
        used_for_training to TRUE and acceptable_for_training to FALSE"""
        print("Setting used_for_training to TRUE and acceptable for training to FALSE for the global fragments already used...")
        for global_fragment in self.next_global_fragments:
            global_fragment._used_for_training = True
            global_fragment._acceptable_for_training = False ###NOTE review this flag. This global fragment is technically acceptable for training. Maybe we can work by only setting used_for_training to TRUE

    def update_used_images_and_labels(self):
        """ we add the images that were in the set of new_images to the set of used_images"""
        print("Updating used_images...")
        if self.counter == 0:
            self.used_images = self.new_images
            self.used_labels = self.new_labels
        else:
            self.used_images = np.concatenate([self.used_images, self.new_images], axis = 0)
            self.used_labels = np.concatenate([self.used_labels, self.new_labels], axis = 0)
        print("Used images for training:", self.used_images.shape, self.used_labels.shape)

    def assign_identities_to_accumulated_global_fragments(self, blobs_in_video):
        """ assign the identities to the global fragments used for training and
        to the blobs that belong to these global fragments. This function checks
        that the identities of the individual fragments in the global fragment
        are consistent with the previously assigned identities"""
        print("Assigning identities to global fragments and blobs used in this accumulation step...")
        for global_fragment in self.next_global_fragments:
            assert global_fragment.used_for_training == True
            self.check_consistency_of_assignation(global_fragment)
            global_fragment._ids_assigned = np.asarray(global_fragment._temporary_ids) + 1
            [blob.update_identity_in_fragment(identity_in_fragment, assigned_during_accumulation = True)
                for blob, identity_in_fragment in zip(blobs_in_video[global_fragment.index_beginning_of_fragment], global_fragment._ids_assigned)]

    def check_consistency_of_assignation(self,global_fragment):
        """ this function checks that the identities of the individual fragments in the global
        fragment that is going to be assigned is consistent with the identities of the individual
        fragments that have already been used for training """
        for individual_fragment_identifier, temporal_identity in zip(global_fragment.individual_fragments_identifiers, global_fragment._temporary_ids):
            if individual_fragment_identifier in self.individual_fragments_used:
                index = self.individual_fragments_used.index(individual_fragment_identifier)
                assert self.identities_of_individual_fragments_used[index] == temporal_identity

    def update_individual_fragments_used(self):
        """ Updates the list of individual fragments used in training and their identities.
        If a individual fragment was added before is not added again.
        """
        print("Updating list of individual fragments used for training")
        new_individual_fragments_and_id = [(individual_fragment_identifier, temporal_identity)
                                                for global_fragment in self.next_global_fragments
                                                for individual_fragment_identifier, temporal_identity in zip(global_fragment.individual_fragments_identifiers, global_fragment._temporary_ids)
                                                if global_fragment.used_for_training and not individual_fragment_identifier in self.individual_fragments_used]
        new_individual_fragments = list(np.asarray(list(new_individual_fragments_and_id))[:,0])
        new_ids = list(np.asarray(list(new_individual_fragments_and_id))[:,1])
        self.individual_fragments_used.extend(new_individual_fragments)
        self.identities_of_individual_fragments_used.extend(new_ids)
        print("Individual fragments used for training:", self.individual_fragments_used)
        print("Ids of individual fragments used for training:", self.identities_of_individual_fragments_used)

    # def get_images_from_test_global_fragments(self):
    #     """stack all the images in global fragments if they have not been used
    #     for training. Optimised for GPU computing"""
    #     print("Stacking images of global fragment for the GPU")
    #     return np.concatenate([np.concatenate(global_fragment.portraits, axis = 0)
    #                 for global_fragment in self.global_fragments
    #                 if not global_fragment.used_for_training], axis = 0)

    def split_predictions_after_network_assignment(self,predictions, softmax_probs, indices_to_split):
        """Go back to the CPU"""
        print("Un-stacking predictions for the CPU")
        individual_fragments_predictions = np.split(predictions, indices_to_split)
        individual_fragments_softmax_probs = np.split(softmax_probs, indices_to_split)
        self.frequencies_of_candidate_individual_fragments = []
        self.P1_vector_of_candidate_individual_fragments = []
        self.median_softmax_of_candidate_individual_fragments = [] #used to compute the certainty on the network's assignment
        self.certainty_of_candidate_individual_fragments = []

        for individual_fragment_predictions, individual_fragment_softmax_probs in zip(individual_fragments_predictions, individual_fragments_softmax_probs):

            frequencies_of_candidate_individual_fragment = compute_identification_frequencies_individual_fragment(np.asarray(individual_fragment_predictions), self.number_of_animals)
            self.frequencies_of_candidate_individual_fragments.append(frequencies_of_candidate_individual_fragment)

            P1_of_candidate_individual_fragment = compute_P1_individual_fragment_from_frequencies(frequencies_of_candidate_individual_fragment)
            # print("P1_of_candidate_individual_fragment: ", P1_of_candidate_individual_fragment)
            self.P1_vector_of_candidate_individual_fragments.append(P1_of_candidate_individual_fragment)

            median_softmax_of_candidate_individual_fragment = compute_median_softmax(individual_fragment_softmax_probs)
            # print("individual_fragment_softmax_median: ", median_softmax_of_candidate_individual_fragment)
            self.median_softmax_of_candidate_individual_fragments.append(median_softmax_of_candidate_individual_fragment)

            certainty_of_individual_fragment = compute_certainty_of_individual_fragment(P1_of_candidate_individual_fragment,median_softmax_of_candidate_individual_fragment)
            # print("certainty_of_individual_fragment: ", certainty_of_individual_fragment)
            self.certainty_of_candidate_individual_fragments.append(certainty_of_individual_fragment)

    def assign_identities_and_check_eligibility_for_training_global_fragments(self, candidate_individual_fragments_identifiers):
        """Assigns identities during test to blobs in global fragments and rank them
        according to the score computed from the certainty of identification and the
        minimum distance travelled"""
        self.candidate_individual_fragments_identifiers = candidate_individual_fragments_identifiers
        self.temporal_inividual_fragments_used = list(self.individual_fragments_used)
        self.temporal_identities_of_individual_fragments_used = list(self.identities_of_individual_fragments_used)
        for i, global_fragment in enumerate(self.global_fragments):
            if global_fragment.used_for_training == False:
                self.assign_identities_to_test_global_fragment(global_fragment)

    def assign_identities_to_test_global_fragment(self, global_fragment):
        assert global_fragment.used_for_training == False
        global_fragment._acceptable_for_training = True

        global_fragment._certainties = []
        global_fragment._P1_vector = []
        # Check certainties of the individual fragments in the global fragment
        # print("\nChecking global fragment")
        for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:
            # print("\nindividual_fragment_identifier: ", individual_fragment_identifier)
            if individual_fragment_identifier in self.candidate_individual_fragments_identifiers:
                # if the individual fragment is in the list of candidates we check the certainty
                index_in_candidate_individual_fragments = list(self.candidate_individual_fragments_identifiers).index(individual_fragment_identifier)
                individual_fragment_certainty =  self.certainty_of_candidate_individual_fragments[index_in_candidate_individual_fragments]
                if individual_fragment_certainty <= self.certainty_threshold:
                    # if the certainty of the individual fragment is not high enough
                    # we set the global fragment not to be acceptable for training
                    global_fragment._acceptable_for_training = False
                    # print("it is not certain enough: ", individual_fragment_certainty)
                    break
                else:
                    # if the certainty of the individual fragment is high enough
                    global_fragment._certainties.append(individual_fragment_certainty)
                    individual_fragment_P1_vector = self.P1_vector_of_candidate_individual_fragments[index_in_candidate_individual_fragments]
                    global_fragment._P1_vector.append(individual_fragment_P1_vector)
            elif individual_fragment_identifier in self.individual_fragments_used:
                # if the individual fragment is no in the list of candidates is because it has been assigned
                # and it is in the list of individual_fragments_used. We set the certainty to 1. And we
                global_fragment._certainties.append(1.)
                individual_fragment_P1_vector = np.zeros(self.number_of_animals)
                individual_fragment_identifier_index = list(self.individual_fragments_used).index(individual_fragment_identifier)
                individual_fragment_id = self.identities_of_individual_fragments_used[individual_fragment_identifier_index]
                individual_fragment_P1_vector[individual_fragment_id] = 0.99999999999999
                global_fragment._P1_vector.append(individual_fragment_P1_vector)

        # Compute identities if the global_fragment is acceptable for training after checking the certainty
        # and check uniqueness
        if global_fragment._acceptable_for_training:
            # print("it is certain enough")
            global_fragment._temporary_ids = np.nan * np.ones(self.number_of_animals)
            # get the index position of the individual fragments ordered by certainty from max to min
            index_individual_fragments_sorted_by_certanity_max_to_min = np.argsort(np.squeeze(np.asarray(global_fragment._certainties)))[::-1]
            # get array of P1 values for the global fragment
            P1_array = np.asarray(global_fragment._P1_vector)
            # get the maximum P1 of each individual fragment
            P1_max = np.max(P1_array,axis=1)
            # get the index position of the individual fragments ordered by P1_max from max to min
            index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]

            # first we set the identities of the individual fragments that have been already used
            for index_individual_fragment, individual_fragment_identifier in enumerate(global_fragment.individual_fragments_identifiers):
                if individual_fragment_identifier in self.temporal_inividual_fragments_used:
                    index = self.temporal_inividual_fragments_used.index(individual_fragment_identifier)
                    identity = int(self.temporal_identities_of_individual_fragments_used[index])
                    global_fragment._temporary_ids[index_individual_fragment] = identity
                    P1_array[index_individual_fragment,:] = 0.
                    P1_array[:,identity] = 0.

            ### NOTE: we need to decide whether we want to assign the identities by order of P1 or by order of certainty
            ### NOTE: the certainty now considers the distance to the random assignation for each individual fragment. check the function below
            for index_individual_fragment in index_individual_fragments_sorted_by_certanity_max_to_min:
                # print(global_fragment._temporary_ids[index_individual_fragment])
                if np.isnan(global_fragment._temporary_ids[index_individual_fragment]):
                    temporal_identity = np.argmax(P1_array[index_individual_fragment,:])
                    global_fragment._temporary_ids[index_individual_fragment] = int(temporal_identity)
                    P1_array[index_individual_fragment,:] = 0.
                    P1_array[:,temporal_identity] = 0.
            # Check if the global fragment is unique after assigning the identities
            if not global_fragment.is_unique:
                # print("is not unique")
                global_fragment._acceptable_for_training = False
            else:
                global_fragment._temporary_ids = np.asarray(global_fragment._temporary_ids).astype('int')

            # Check consistenscy with rest of global fragments
            if global_fragment.is_unique:
                # print("is unique")
                if self.check_consistency_of_assignation_tests(global_fragment):
                    # print("is consistent")
                    for individual_fragment_identifier, temporal_identity in zip(global_fragment.individual_fragments_identifiers,global_fragment._temporary_ids):
                        if individual_fragment_identifier not in self.temporal_inividual_fragments_used:
                            self.temporal_inividual_fragments_used.append(individual_fragment_identifier)
                            self.temporal_identities_of_individual_fragments_used.append(temporal_identity)
                else:
                    # print("is not consistent")
                    global_fragment._acceptable_for_training = False


    def check_consistency_of_assignation_tests(self,global_fragment):
        """ this function checks that the identities of the individual fragments in the global
        fragment that is going to be assigned is consistent with the identities of the individual
        fragments that have already been used for training """
        for individual_fragment_identifier, temporal_identity in zip(global_fragment.individual_fragments_identifiers, global_fragment._temporary_ids):
            if individual_fragment_identifier in self.temporal_inividual_fragments_used:
                index = self.temporal_inividual_fragments_used.index(individual_fragment_identifier)
                if not self.temporal_identities_of_individual_fragments_used[index] == temporal_identity:
                    return False
        return True


def sample_images_and_labels(images, labels, ratio):
    subsampled_images = []
    subsampled_labels = []
    print(np.unique(labels))
    for i in np.unique(labels):
        if len(images[np.where(labels == i)[0]]) > NUMBER_OF_SAMPLES:
            subsampled_images.append(random.sample(images[np.where(labels == i)[0]],NUMBER_OF_SAMPLES))
            subsampled_labels.append([i] * NUMBER_OF_SAMPLES)
        else:
            subsampled_images.append(images[np.where(labels == i)[0]])
            subsampled_labels.append([i] * len(images[np.where(labels == i)[0]]))

    return np.concatenate(subsampled_images, axis = 0), np.concatenate(subsampled_labels, axis = 0)

def compute_median_softmax(individual_fragment_softmax_probs):
    individual_fragment_softmax_probs = np.asarray(individual_fragment_softmax_probs)
    # print("individual_fragment_softmax_probs: ", individual_fragment_softmax_probs.shape)
    max_softmax_probs = np.max(individual_fragment_softmax_probs, axis = 1)
    # print("max_softmax_probs: ", max_softmax_probs.shape)
    argmax_softmax_probs = np.argmax(individual_fragment_softmax_probs, axis = 1)
    # print("argmax_softmax_probs: ", argmax_softmax_probs.shape)
    num_animals = individual_fragment_softmax_probs.shape[1]
    individual_fragment_softmax_median = np.zeros(num_animals)
    for i in np.unique(argmax_softmax_probs):
        individual_fragment_softmax_median[i] = np.median(max_softmax_probs[argmax_softmax_probs==i])

    return individual_fragment_softmax_median



"""functions used during accumulation
but belong to the assign part of the accumulation that's why are here
"""
def compute_certainty_of_individual_fragment(p1_vector_individual_fragment,median_softmax_of_candidate_individual_fragment):
    argsort_p1_vector = np.argsort(p1_vector_individual_fragment)
    sorted_p1_vector = p1_vector_individual_fragment[argsort_p1_vector]
    sorted_softmax_probs = median_softmax_of_candidate_individual_fragment[argsort_p1_vector]
    certainty = np.diff(np.multiply(sorted_p1_vector,sorted_softmax_probs)[-2:])/np.sum(sorted_p1_vector[-2:])
    return certainty
