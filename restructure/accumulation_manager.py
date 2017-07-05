from __future__ import absolute_import, division, print_function
import numpy as np
import random
from globalfragment import get_images_and_labels_from_global_fragments, order_global_fragments_by_distance_travelled
from statistics_for_assignment import compute_P1_individual_fragment_from_blob, compute_identification_frequencies_individual_fragment

RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = 0.1 # threshold to select a individual fragment as eligible for training

###
random.seed(0)
###

class AccumulationManager(object):
    def __init__(self,global_fragments, number_of_animals, accumulation_counter = 0, certainty_threshold = CERTAINTY_THRESHOLD):
        self.counter = accumulation_counter
        self.number_of_animals = number_of_animals
        self.global_fragments = global_fragments
        self.individual_fragments_used = []
        self.used_images = None
        self.used_labels = None
        self.new_images = None
        self.new_labels = None
        self._continue_accumulation = True
        self.certainty_threshold = certainty_threshold

    @property
    def continue_accumulation(self):
        if not any([global_fragment.acceptable_for_training for global_fragment in self.global_fragments]):
            return False
        else:
            return True

    def update_counter(self):
        self.counter += 1

    def get_next_global_fragments(self):
        if self.counter == 0:
            print("\nGetting global fragment for the first accumulation")
            self.next_global_fragments = [order_global_fragments_by_distance_travelled(self.global_fragments)[0]]
        else:
            print("\nGetting global fragments")
            self.next_global_fragments = [global_fragment for global_fragment in self.global_fragments
                                                if global_fragment.acceptable_for_training == True]
        print("Number of global fragments for training, ", len(self.next_global_fragments))

    def get_new_images_and_labels(self):
        self.new_images, self.new_labels, _, _ = get_images_and_labels_from_global_fragments(self.next_global_fragments,list(self.individual_fragments_used))
        print("New images for training:", self.new_images.shape, self.new_labels.shape)
        if self.used_images is not None:
            print("Old images for training:", self.used_images.shape, self.used_labels.shape)

    def get_images_and_labels_for_training(self):
        images = []
        labels = []

        for i in range(self.number_of_animals):
            # print("\nTaking images for individual %i" %i)
            new_images_indices = np.where(self.new_labels == i)[0]
            used_images_indices = np.where(self.used_labels == i)[0]
            number_of_new_images = len(new_images_indices)
            number_of_used_images = len(used_images_indices)
            number_of_images_for_individual = number_of_new_images + number_of_used_images
            if number_of_images_for_individual > MAXIMAL_IMAGES_PER_ANIMAL:
                # print("The total number of images for this individual is greater than %i " %MAXIMAL_IMAGES_PER_ANIMAL)
                # print("we sample %f from the old ones and %f from the new ones" %(RATIO_OLD, RATIO_NEW))
                number_samples_new = int(MAXIMAL_IMAGES_PER_ANIMAL * RATIO_NEW)
                number_samples_used = int(MAXIMAL_IMAGES_PER_ANIMAL * RATIO_OLD)
                if number_of_used_images < number_samples_used:
                    number_samples_used = number_of_used_images
                    number_samples_new = MAXIMAL_IMAGES_PER_ANIMAL - number_samples_used
                if number_of_new_images < number_samples_new:
                    number_samples_new = number_of_new_images
                    number_samples_used = MAXIMAL_IMAGES_PER_ANIMAL - number_samples_new

                images.extend(random.sample(self.new_images[new_images_indices],number_samples_new))
                labels.extend([i] * number_samples_new)
                if self.used_images is not None:
                    images.extend(random.sample(self.used_images[used_images_indices],number_samples_used))
                    labels.extend([i] * number_samples_used)
            else:
                # print("The total number of images for this individual is %i " %number_of_images_for_individual)
                # print("we take all the new images")
                images.extend(self.new_images[new_images_indices])
                labels.extend([i] * number_of_new_images)
                if self.used_images is not None:
                    images.extend(self.used_images[used_images_indices])
                    labels.extend([i] * number_of_used_images)

        return np.asarray(images), np.asarray(labels)

    def update_global_fragments_used_for_training(self):
        print("Setting used_for_training to TRUE and acceptable for training to FALSE for the global fragments already used")
        for global_fragment in self.next_global_fragments:
            global_fragment._used_for_training = True
            global_fragment._acceptable_for_training = False

    def update_individual_fragments_used(self):
        print("Updating list of individual fragments used for training")
        self.individual_fragments_used = np.unique(np.concatenate([global_fragment.individual_fragments_identifiers
                                                                    for global_fragment in self.global_fragments
                                                                    if global_fragment.used_for_training]))
        print("Individual fragments used for training:", self.individual_fragments_used)

    def update_used_images_and_labels(self):
        print("Updating used_images")
        if self.counter == 0:
            self.used_images = self.new_images
            self.used_labels = self.new_labels
        else:
            self.used_images = np.concatenate([self.used_images, self.new_images], axis = 0)
            self.used_labels = np.concatenate([self.used_labels, self.new_labels], axis = 0)
        print("Old images for training:", self.used_images.shape, self.used_labels.shape)

    def assign_identities_to_accumulated_global_fragments(self, blobs_in_video):
        """Assign the identities in identities_list to both global fragment and all its blobs
        after training on the global fragment"""
        print("Assigning identities to global fragments and blobs used in this accumulation step")
        for global_fragment in self.next_global_fragments:
            assert global_fragment.used_for_training == True
            global_fragment._ids_assigned = np.asarray(global_fragment._temporary_ids) + 1
            [blob.update_identity_in_fragment(identity_in_fragment, assigned_during_accumulation = True)
                for blob, identity_in_fragment in zip(blobs_in_video[global_fragment.index_beginning_of_fragment], global_fragment._ids_assigned)]

    def get_images_from_test_global_fragments(self):
        """stack all the images in global fragments if they have not been used
        for training. Optimised for GPU computing"""
        print("Stacking images of global fragment for the GPU")
        return np.concatenate([np.concatenate(global_fragment.portraits, axis = 0)
                    for global_fragment in self.global_fragments
                    if not global_fragment.used_for_training], axis = 0)

    def split_predictions_after_network_assignment(self,predictions, softmax_probs, indices_to_split):
        """Go back to the CPU"""
        print("Un-stacking predictions for the CPU")
        individual_fragments_predictions = np.split(predictions, indices_to_split)
        individual_fragments_softmax_probs = np.split(softmax_probs, indices_to_split)
        self.frequencies_in_candidate_individual_fragments = []
        self.P1_of_candidate_individual_fragments = []
        self.median_softmax_candidate_individual_fragments = [] #used to compute the certainty on the network's assignment
        self.identity_of_candidate_individual_fragments = []

        for individual_fragment_predictions, individual_fragment_softmax_probs in zip(individual_fragments_predictions, individual_fragments_softmax_probs):
            frequencies_in_candidate_individual_fragment = compute_identification_frequencies_individual_fragment(np.asarray(individual_fragment_predictions), self.number_of_animals)
            P1_of_fragment = compute_P1_individual_fragment_from_blob(frequencies_in_candidate_individual_fragment)
            self.frequencies_in_candidate_individual_fragments.append(frequencies_in_candidate_individual_fragment)
            self.P1_of_candidate_individual_fragments.append(P1_of_fragment)
            self.identity_of_candidate_individual_fragments.append(np.argmax(P1_of_fragment))
            self.median_softmax_candidate_individual_fragments.append(np.median(individual_fragment_softmax_probs, axis = 0))

    def assign_identities_and_check_eligibility_for_training_global_fragments(self, candidate_individual_fragments_identifiers):
        """Assigns identities during test to blobs in global fragments and rank them
        according to the score computed from the certainty of identification and the
        minimum distance travelled"""
        self.candidate_individual_fragments_identifiers = candidate_individual_fragments_identifiers
        for i, global_fragment in enumerate(self.global_fragments):
            # print("Analysing whether global fragment %i is good for training" %i)
            if global_fragment.used_for_training == False:
                self.assign_identities_to_test_global_fragment(global_fragment)
            # else:
                # print("global fragment %i has been used for training" %i)

    def assign_identities_to_test_global_fragment(self, global_fragment):
        assert global_fragment.used_for_training == False
        global_fragment._temporary_ids = []
        global_fragment._P1_vector = []
        global_fragment._frequencies_in_fragment = []
        global_fragment._median_softmax_probs = []
        global_fragment._acceptable_for_training = True
        global_fragment._certainties = []
        for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:
            # print("candidate frag identifiers, ", self.candidate_individual_fragments_identifiers)
            # print("index ", list(self.candidate_individual_fragments_identifiers).index(individual_fragment_identifier))
            index_in_candidate_individual_fragments = list(self.candidate_individual_fragments_identifiers).index(individual_fragment_identifier)
            # print(len(self.identity_of_candidate_individual_fragments))
            # print(len(self.candidate_individual_fragments_identifiers))
            cur_id = self.identity_of_candidate_individual_fragments[index_in_candidate_individual_fragments]
            cur_P1 = self.P1_of_candidate_individual_fragments[index_in_candidate_individual_fragments]
            cur_frequencies = self.frequencies_in_candidate_individual_fragments[index_in_candidate_individual_fragments]
            cur_softmax_median = self.median_softmax_candidate_individual_fragments[index_in_candidate_individual_fragments]

            global_fragment._temporary_ids.append(cur_id)
            global_fragment._P1_vector.append(cur_P1)
            global_fragment._frequencies_in_fragment.append(cur_frequencies)
            global_fragment._median_softmax_probs.append(cur_softmax_median)
            acceptable_individual_fragment, certainty = check_certainty_individual_fragment(cur_P1, cur_softmax_median, self.certainty_threshold)
            global_fragment._certainties.append(certainty)
            if not acceptable_individual_fragment:
                # print("This individual fragment is not good for training")
                global_fragment._acceptable_for_training = False
        # print(global_fragment._temporary_ids)
        if not global_fragment.is_unique:
            # print("The global fragment is not unique")
            global_fragment._acceptable_for_training = False
        else:
            global_fragment._temporary_ids = np.asarray(global_fragment._temporary_ids)

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


"""functions used during accumulation
but belong to the assign part of the accumulation that's why are here
"""
def check_certainty_individual_fragment(frequencies_individual_fragment,softmax_probs_median_individual_fragment, certainty_threshold):
    argsort_frequencies = np.argsort(frequencies_individual_fragment)
    sorted_frequencies = frequencies_individual_fragment[argsort_frequencies]
    sorted_softmax_probs = softmax_probs_median_individual_fragment[argsort_frequencies]
    certainty = np.diff(np.multiply(sorted_frequencies,sorted_softmax_probs)[-2:])/np.sum(sorted_frequencies[-2:])
    acceptable_individual_fragment = False
    if certainty > certainty_threshold:
        acceptable_individual_fragment = True
    # else:
        # print("global fragment discarded with certainty ", certainty)
    return acceptable_individual_fragment, certainty
