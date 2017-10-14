from __future__ import absolute_import, division, print_function
from pprint import pprint
import numpy as np
import random
import psutil
import logging

from assigner import assign
from trainer import train

RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = .1 # threshold to select a individual fragment as eligible for training
MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY = .5

logger = logging.getLogger("__main__.accumulation_manager")

class AccumulationManager(object):
    def __init__(self, video,
                list_of_fragments, list_of_global_fragments,
                certainty_threshold = CERTAINTY_THRESHOLD,
                allow_partial_accumulation = False,
                threshold_acceptable_accumulation = None):
        """ This class manages the selection of global fragments for accumulation,
        the retrieval of images from the new global fragments, the selection of
        of images for training, the final assignment of identities to the global fragments
        used for training, the temporary assignment of identities to the candidates global fragments
        and the computation of the certainty levels of the individual fragments to check the
        eligability of the candidates global fragments for accumulation.

        :list_of_fragments: list of individual and crossing fragments
        :list_of_global_fragments:  object
        :number_of_animals param: number of animals in the video
        :certainty_threshold param: threshold to set a individual fragment as certain for accumulation
        """
        self.video = video
        self.number_of_animals = video.number_of_animals
        self.list_of_fragments = list_of_fragments
        self.list_of_global_fragments = list_of_global_fragments
        self.counter = 0
        self.certainty_threshold = certainty_threshold
        self.threshold_acceptable_accumulation = threshold_acceptable_accumulation
        self.accumulation_strategy = 'global'
        self.individual_fragments_used = [] # list with the individual_fragments_identifiers of the individual fragments used for training
        self.used_images = None # images used for training the network
        self.used_labels = None # labels for the images used for training
        self.new_images = None # set of images that will be added to the new training
        self.new_labels = None # labels for the set of images that will be added for training
        self._continue_accumulation = True # flag to continue_accumulation or not

    @property
    def continue_accumulation(self):
        """ We stop the accumulation when there are not more global fragments
        that are acceptable for training"""
        print("******************************************")
        print("global fragments acceptable_for_training: ", sum([global_fragment.acceptable_for_training(self.accumulation_strategy)
                                                                for global_fragment in self.list_of_global_fragments.global_fragments]))
        print("global fragments not used_for_training: ", sum([not global_fragment.used_for_training
                                                                for global_fragment in self.list_of_global_fragments.global_fragments]))
        if not any([(global_fragment.acceptable_for_training(self.accumulation_strategy)
                    and not global_fragment.used_for_training)
                    for global_fragment in self.list_of_global_fragments.global_fragments]):
            return False
        else:
            return True

    def update_counter(self):
        """ updates the counter of the accumulation"""
        self.counter += 1

    def get_new_images_and_labels(self):
        """ get the images and labels of the new global fragments that are going
        to be used for training, this function checks whether the images of a individual
        fragment have been added before"""
        self.new_images, self.new_labels, _, _ = self.list_of_fragments.get_new_images_and_labels_for_training()

        if self.new_images is not None:
            logger.info("New images for training: %s %s"  %(str(self.new_images.shape), str(self.new_labels.shape)))
        else:
            logger.info("There are no new images in this accumulation")
        if self.used_images is not None:
            logger.info("Old images for training: %s %s" %(str(self.used_images.shape), str(self.used_labels.shape)))

    def update_global_fragments_used_for_training(self):
        """ Once a global fragment has been used for training we set the flags
        used_for_training to TRUE and acceptable_for_training to FALSE"""
        logger.debug("Setting used_for_training to TRUE and acceptable for training to FALSE for the global fragments already used...")
        [(setattr(fragment,'_used_for_training',True), setattr(fragment,'_acceptable_for_training',False))
                                                for fragment in self.list_of_fragments.fragments
                                                if fragment.acceptable_for_training == True]

    def get_images_and_labels_for_training(self):
        """ We limit the number of images per animal that are used for training
        to MAXIMAL_IMAGES_PER_ANIMAL. We take a percentage RATIO_NEW of the new
        images and a percentage RATIO_OLD of the images already used. """
        logger.info("Getting images for training...")
        random.seed(0)
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
                if self.new_images is not None:
                    images.extend(random.sample(self.new_images[new_images_indices],number_samples_new))
                    labels.extend([i] * number_samples_new)
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate the variable used_images is None
                    images.extend(random.sample(self.used_images[used_images_indices],number_samples_used))
                    labels.extend([i] * number_samples_used)
            else:
                # if the total number of images for this label does not exceed the MAXIMAL_IMAGES_PER_ANIMAL
                # we take all the new images and all the used images
                if self.new_images is not None:
                    images.extend(self.new_images[new_images_indices])
                    labels.extend([i] * number_of_new_images)
                if self.used_images is not None:
                    # this condition is set because the first time we accumulate the variable used_images is None
                    images.extend(self.used_images[used_images_indices])
                    labels.extend([i] * number_of_used_images)
        return np.asarray(images), np.asarray(labels)

    def update_used_images_and_labels(self):
        """ we add the images that were in the set of new_images to the set of used_images"""
        logger.debug("Updating used_images...")
        if self.counter == 0:
            self.used_images = self.new_images
            self.used_labels = self.new_labels
        elif self.new_images is not None:
            self.used_images = np.concatenate([self.used_images, self.new_images], axis = 0)
            self.used_labels = np.concatenate([self.used_labels, self.new_labels], axis = 0)
        logger.info("number of images used for training: %s %s" %(str(self.used_images.shape), str(self.used_labels.shape)))

    def assign_identities_to_fragments_used_for_training(self):
        """ assign the identities to the global fragments used for training and
        to the blobs that belong to these global fragments. This function checks
        that the identities of the individual fragments in the global fragment
        are consistent with the previously assigned identities"""
        [setattr(fragment, '_identity', getattr(fragment, 'temporary_id') + 1)
            for fragment in self.list_of_fragments.fragments
            if fragment.used_for_training]

    def update_individual_fragments_used_for_training(self):
        return list(set([fragment.identifier for fragment in self.list_of_fragments.fragments
                if fragment.used_for_training
                and fragment.identifier not in self.individual_fragments_used]))

    def update_individual_fragments_used(self):
        """ Updates the list of individual fragments used in training and their identities.
        If a individual fragment was added before is not added again.
        """
        logging.info("Updating list of individual fragments used for training")
        new_individual_fragments_identifiers = self.update_individual_fragments_used_for_training()
        self.individual_fragments_used.extend(new_individual_fragments_identifiers)
        logging.info("number of individual fragments used for training: %i" %sum([fragment.used_for_training for fragment in self.list_of_fragments.fragments]))

    def split_predictions_after_network_assignment(self, predictions, softmax_probs, indices_to_split, candidate_individual_fragments_identifiers):
        """Go back to the CPU"""
        logging.info("Un-stacking predictions for the CPU")
        individual_fragments_predictions = np.split(predictions, indices_to_split)
        individual_fragments_softmax_probs = np.split(softmax_probs, indices_to_split)
        self.frequencies_of_candidate_individual_fragments = []
        self.P1_vector_of_candidate_individual_fragments = []
        self.median_softmax_of_candidate_individual_fragments = [] #used to compute the certainty on the network's assignment
        self.certainty_of_candidate_individual_fragments = []

        for individual_fragment_predictions, \
            individual_fragment_softmax_probs, \
            candidate_individual_fragment_identifier in zip(individual_fragments_predictions,\
                                                                individual_fragments_softmax_probs,\
                                                                candidate_individual_fragments_identifiers):

            index = self.video.fragment_identifier_to_index[candidate_individual_fragment_identifier]
            self.list_of_fragments.fragments[index].compute_identification_statistics(np.asarray(individual_fragment_predictions),\
                                                                            individual_fragment_softmax_probs)

    def reset_accumulation_variables(self):
        self.temporary_individual_fragments_used = []
        if self.accumulation_strategy == 'global' :
            self.number_of_noncertain_global_fragments = 0
            self.number_of_random_assigned_global_fragments = 0
            self.number_of_nonconsistent_global_fragments = 0
            self.number_of_nonunique_global_fragments = 0
        self.number_of_sparse_fragments = 0
        self.number_of_noncertain_fragments = 0
        self.number_of_random_assigned_fragments = 0
        self.number_of_nonconsistent_fragments = 0
        self.number_of_nonunique_fragments = 0
        self.number_of_acceptable_fragments = 0

    def get_acceptable_global_fragments_for_training(self, candidate_individual_fragments_identifiers):
        """Assigns identities during test to blobs in global fragments and rank them
        according to the score computed from the certainty of identification and the
        minimum distance travelled"""
        self.accumulation_strategy = 'global'
        self.candidate_individual_fragments_identifiers = candidate_individual_fragments_identifiers
        self.reset_accumulation_variables()
        logger.debug('Accumulating by global strategy')
        for i, global_fragment in enumerate(self.list_of_global_fragments.global_fragments):
            if global_fragment.used_for_training == False:
                self.check_if_is_acceptable_for_training(global_fragment)
        self.number_of_acceptable_global_fragments = np.sum([global_fragment.acceptable_for_training(self.accumulation_strategy)
                                                                and not global_fragment.used_for_training
                                                                for global_fragment in self.list_of_global_fragments.global_fragments])
        if self.number_of_acceptable_global_fragments == 0\
            and self.ratio_accumulated_images > MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY\
            and self.ratio_accumulated_images < self.threshold_early_stop_accumulation:
            logger.debug('Accumulating by partial strategy')
            self.accumulation_strategy = 'partial'
            self.reset_accumulation_variables()
            for i, global_fragment in enumerate(self.list_of_global_fragments.global_fragments):
                if global_fragment.used_for_training == False:
                    self.check_if_is_acceptable_for_training(global_fragment)
        elif self.ratio_accumulated_images < MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY:
            logger.info('The ratio of accumulated images is too small and a partial accumulation might fail.')

    def reset_non_acceptable_fragment(self, fragment):
        if fragment.identifier not in self.temporary_individual_fragments_used \
        and fragment.identifier not in self.individual_fragments_used:
            fragment._temporary_id = None
            fragment._acceptable_for_training = False

    def reset_non_acceptable_global_fragment(self, global_fragment):
        for fragment in global_fragment.individual_fragments:
            self.reset_non_acceptable_fragment(fragment)

    def check_if_is_acceptable_for_training(self, global_fragment):
        if self.accumulation_strategy == 'global':
            # Check certainties of the individual fragments in the global fragment
            # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:
            [setattr(fragment,'_acceptable_for_training', True) for fragment in global_fragment.individual_fragments]
            for fragment in global_fragment.individual_fragments:
                if fragment.identifier in self.candidate_individual_fragments_identifiers:
                    if fragment.certainty < self.certainty_threshold:
                        # if the certainty of the individual fragment is not high enough
                        # we set the global fragment not to be acceptable for training
                        self.reset_non_acceptable_global_fragment(global_fragment)
                        self.number_of_noncertain_global_fragments += 1

                        fragment._is_certain = False
                        break
                    else:
                        # if the certainty of the individual fragment is high enough
                        fragment._is_certain = True
                elif fragment.identifier in self.individual_fragments_used:
                    # if the individual fragment is not in the list of candidates is because it has been assigned
                    # and it is in the list of individual_fragments_used. We set the certainty to 1. And we
                    fragment._is_certain = True
                else:
                    logging.warn("Individual fragment not in candidates or in used, this should not happen")
            # Compute identities if the global_fragment is certain
            if global_fragment.acceptable_for_training(self.accumulation_strategy):
                # get array of P1 values for the global fragment
                P1_array = np.asarray([fragment.P1_vector for fragment in global_fragment.individual_fragments])
                # get the maximum P1 of each individual fragment
                P1_max = np.max(P1_array, axis = 1)
                # logger.debug("P1 max: %s" %str(P1_max))
                # get the index position of the individual fragments ordered by P1_max from max to min
                index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]
                # set to zero the P1 of the the identities of the individual fragments that have been already used
                for index_individual_fragment, fragment in enumerate(global_fragment.individual_fragments):
                    if fragment.identifier in self.individual_fragments_used or fragment.identifier in self.temporary_individual_fragments_used:
                        P1_array[index_individual_fragment,:] = 0.
                        P1_array[:,fragment.temporary_id] = 0.
                # assign temporal identity to individual fragments by hierarchical P1
                for index_individual_fragment in index_individual_fragments_sorted_by_P1_max_to_min:
                    fragment = global_fragment.individual_fragments[index_individual_fragment]
                    if fragment.temporary_id is None:
                        if np.max(P1_array[index_individual_fragment,:]) < 1./fragment.number_of_images:
                            fragment._P1_below_random = True
                            self.number_of_random_assigned_global_fragments += 1
                            self.reset_non_acceptable_global_fragment(global_fragment)
                            break
                        else:
                            temporary_id = np.argmax(P1_array[index_individual_fragment,:])
                            if not fragment.check_consistency_with_coexistent_individual_fragments(temporary_id):
                                self.reset_non_acceptable_global_fragment(global_fragment)
                                fragment._non_consistent = True
                                self.number_of_nonconsistent_global_fragments += 1
                                break
                            else:
                                fragment._temporary_id = int(temporary_id)
                                P1_array[index_individual_fragment,:] = 0.
                                P1_array[:,temporary_id] = 0.

                # Check if the global fragment is unique after assigning the identities
                if global_fragment.acceptable_for_training(self.accumulation_strategy):
                    if not global_fragment.is_unique:
                        # set acceptable_for_training to False and temporary_id to None for all the individual_fragments
                        # that had not been accumulated before (i.e. not in temporary_individual_fragments_used or individual_fragments_used)
                        self.reset_non_acceptable_global_fragment(global_fragment)
                        self.number_of_nonunique_global_fragments += 1
                    else:
                        global_fragment._accumulation_step = self.counter
                        [self.temporary_individual_fragments_used.append(fragment.identifier)
                            for fragment in global_fragment.individual_fragments
                            if fragment.identifier not in self.temporary_individual_fragments_used
                            and fragment.identifier not in self.individual_fragments_used]
        elif self.accumulation_strategy == 'partial':
            [setattr(fragment,'_acceptable_for_training', False) for fragment in global_fragment.individual_fragments]
            print("\n**** partial")
            print("\nChecking certainty")
            for fragment in global_fragment.individual_fragments:
                print("--fragment ", fragment.identifier)
                # Check certainties of the individual fragme
                if fragment.identifier in self.candidate_individual_fragments_identifiers:
                    print('Is a candidate, id ', fragment.temporary_id)
                    print("acceptable_for_training ", fragment.acceptable_for_training)
                    if fragment.has_enough_accumulated_coexisting_fragments:
                        print('Enough coexisting accumulated fragments, id ', fragment.temporary_id)
                        print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
                        print("acceptable_for_training ", fragment.acceptable_for_training)
                        # Check if the more than half of the individual fragments that coexist with this one have being accumulated
                        if fragment.certainty < self.certainty_threshold:
                            # if the certainty of the individual fragment is not high enough
                            # we set the global fragment not to be acceptable for training
                            self.reset_non_acceptable_fragment(fragment)
                            self.number_of_noncertain_fragments += 1
                            fragment._is_certain = False
                            print('Is not certain, id ', fragment.temporary_id)
                            print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
                            print("acceptable_for_training ", fragment.acceptable_for_training)
                        else:
                            # if the certainty of the individual fragment is high enough
                            fragment._is_certain = True
                            fragment._acceptable_for_training = True
                            print('Is certain, id ', fragment.temporary_id)
                            print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
                            print("acceptable_for_training ", fragment.acceptable_for_training)
                    else:
                        self.reset_non_acceptable_fragment(fragment)
                        self.number_of_sparse_fragments += 1
                        print('Not enough coexisting accuulated fragments, id ', fragment.temporary_id)
                        print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
                        print("acceptable_for_training ", fragment.acceptable_for_training)
                elif fragment.identifier in self.individual_fragments_used:
                    # if the individual fragment is not in the list of candidates is because it has been assigned
                    # and it is in the list of individual_fragments_used. We set the certainty to 1. And we
                    fragment._is_certain = True
                    print('Already used, id ', fragment.temporary_id)
                    print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
                    print("acceptable_for_training ", fragment.acceptable_for_training)
                else:
                    logging.warn("Individual fragment not in candidates or in used, this should not happen")


            # Compute identities if the global_fragment is certain
            # get array of P1 values for the global fragment
            P1_array = np.asarray([fragment.P1_vector for fragment in global_fragment.individual_fragments])
            # get the maximum P1 of each individual fragment
            P1_max = np.max(P1_array, axis = 1)
            # logger.debug("P1 max: %s" %str(P1_max))
            # get the index position of the individual fragments ordered by P1_max from max to min
            index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]
            # set to zero the P1 of the the identities of the individual fragments that have been already used
            print("\nAssigning identities")
            for index_individual_fragment, fragment in enumerate(global_fragment.individual_fragments):
                print('--fragment ', fragment.identifier)
                if fragment.identifier in self.individual_fragments_used or fragment.identifier in self.temporary_individual_fragments_used:
                    print('Already used, id ', fragment.temporary_id)
                    P1_array[index_individual_fragment,:] = 0.
                    P1_array[:,fragment.temporary_id] = 0.
            # assign temporal identity to individual fragments by hierarchical P1
            for index_individual_fragment in index_individual_fragments_sorted_by_P1_max_to_min:
                fragment = global_fragment.individual_fragments[index_individual_fragment]
                print('--fragment ', fragment.identifier)
                if fragment.temporary_id is None:
                    print('Temporary id is None, id ', fragment.temporary_id)
                    if np.max(P1_array[index_individual_fragment,:]) < 1./fragment.number_of_images:
                        fragment._P1_below_random = True
                        self.number_of_random_assigned_fragments += 1
                        self.reset_non_acceptable_fragment(fragment)
                        print('P1 below random, id ', fragment.temporary_id, ', acceptable_for_training ', fragment.acceptable_for_training)
                        print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
                        print("in individual_fragments_used ", fragment.identifier in self.individual_fragments_used)
                        print("used for training ", fragment.used_for_training)
                    else:
                        temporary_id = np.argmax(P1_array[index_individual_fragment,:])
                        if not fragment.check_consistency_with_coexistent_individual_fragments(temporary_id):
                            self.reset_non_acceptable_fragment(fragment)
                            fragment._non_consistent = True
                            self.number_of_nonconsistent_fragments += 1
                            print('P1 above random but no consistent, id ', fragment.temporary_id)
                        else:
                            fragment._acceptable_for_training = True
                            fragment._temporary_id = int(temporary_id)
                            P1_array[index_individual_fragment,:] = 0.
                            P1_array[:,temporary_id] = 0.
                            print('P1 above random and consistent, id ', fragment.temporary_id)

            # Check if the global fragment is unique after assigning the identities
            print("\nChecking uniqueness")
            if not global_fragment.is_partially_unique:
                print("Global fragment not partially unique")
                number_of_duplicated_fragments = len([self.reset_non_acceptable_fragment(fragment)
                                                    for fragment in global_fragment.individual_fragments
                                                    if fragment.temporary_id in global_fragment.duplicated_identities])
                self.number_of_nonunique_fragments += number_of_duplicated_fragments
                for fragment in global_fragment.individual_fragments:
                    print("--fragment ", fragment.identifier)
                    print("acceptable ", fragment.acceptable_for_training, ' id ', fragment.temporary_id)

            else:
                print("Global fragment partially unique")
                global_fragment._accumulation_step = self.counter
                [self.temporary_individual_fragments_used.append(fragment.identifier)
                    for fragment in global_fragment.individual_fragments
                    if fragment.identifier not in self.temporary_individual_fragments_used
                    and fragment.identifier not in self.individual_fragments_used
                    and fragment.acceptable_for_training]
                self.number_of_acceptable_fragments += len([fragment for fragment in global_fragment.individual_fragments
                                                        if fragment.acceptable_for_training and not fragment.used_for_training])
                for fragment in global_fragment.individual_fragments:
                    print("--fragment ", fragment.identifier)
                    print("acceptable ", fragment.acceptable_for_training, ", used_for_training ", fragment.used_for_training, ', id ', fragment.temporary_id)
                    print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)
            # print([(fragment.acceptable_for_training, fragment.identifier) for fragment in global_fragment.individual_fragments])
        if not all([fragment.temporary_id is not None for fragment in global_fragment.individual_fragments if fragment.acceptable_for_training and fragment.is_a_fish]):
            print("\nThere is something wrong")
            print(self.accumulation_strategy)
            for fragment in global_fragment.individual_fragments:
                print('---fragment ', fragment.identifier)
                print("acceptable_for_training ", fragment.acceptable_for_training)
                print("is a fish ", fragment.is_a_fish)
                print("is in a global fragment ", fragment.is_in_a_global_fragment)
                print("temporary_id ", fragment.temporary_id)
                print("used_for_training ", fragment.used_for_training)
                print("in temporary_individual_fragments_used ", fragment.identifier in self.temporary_individual_fragments_used)

        assert all([fragment.temporary_id is not None for fragment in global_fragment.individual_fragments if fragment.acceptable_for_training and fragment.is_a_fish])


def sample_images_and_labels(images, labels, ratio):
    subsampled_images = []
    subsampled_labels = []
    for i in np.unique(labels):
        if len(images[np.where(labels == i)[0]]) > NUMBER_OF_SAMPLES:
            subsampled_images.append(random.sample(images[np.where(labels == i)[0]],NUMBER_OF_SAMPLES))
            subsampled_labels.append([i] * NUMBER_OF_SAMPLES)
        else:
            subsampled_images.append(images[np.where(labels == i)[0]])
            subsampled_labels.append([i] * len(images[np.where(labels == i)[0]]))

    return np.concatenate(subsampled_images, axis = 0), np.concatenate(subsampled_labels, axis = 0)

""" Get predictions of individual fragments in candidates global fragments"""
def get_predictions_of_candidates_fragments(net, video, fragments):
    images = []
    lengths = []
    candidate_individual_fragments_identifiers = []

    for fragment in fragments:
        if fragment.is_a_fish and not fragment.used_for_training:
            images.extend(fragment.images)
            lengths.append(fragment.number_of_images)
            candidate_individual_fragments_identifiers.append(fragment.identifier)

    if len(images) != 0:
        images = np.asarray(images)
        assigner = assign(net, video, images, print_flag = True)

    return assigner._predictions, assigner._softmax_probs, np.cumsum(lengths)[:-1], candidate_individual_fragments_identifiers


def get_predictions_of_candidates_global_fragments(net, video,
                                                    candidates_next_global_fragments,
                                                    individual_fragments_identifiers_already_used = []):

    def get_images_and_labels_from_global_fragment_predictions(video,
                                                                global_fragment,
                                                                individual_fragments_identifiers_already_used = []):
        images = np.ones((video.maximum_number_of_portraits_in_global_fragments, video.portrait_size[0], video.portrait_size[1]))
        lengths = []
        individual_fragments_identifiers = []
        num_images = 0

        for fragment in global_fragment.individual_fragments:
            if fragment.identifier not in individual_fragments_identifiers_already_used:
                if len(np.asarray(fragment.images).shape) == 1:
                    print("fragment.images.shape: ", np.asarray(fragment.images).shape)
                    print("is a fish ", fragment.is_a_fish)
                    print("image shape ", fragment.images[0].shape)
                    print(fragment.identifier)
                    print(fragment.images)
                print("images: ", images.shape)
                images[num_images : fragment.number_of_images + num_images] = np.asarray(fragment.images)
                lengths.append(fragment.number_of_images)
                individual_fragments_identifiers.append(fragment.identifier)
                num_images += fragment.number_of_images

        images = images[:num_images]
        return images, lengths, individual_fragments_identifiers

    predictions = []
    softmax_probs = []
    lengths = []
    candidate_individual_fragments_identifiers = []

    logging.info("Getting images from candidate global fragments for predictions...")
    # compute maximum number of images given the available RAM and SWAP
    image_size_bytes = np.prod(video.portrait_size)*4
    if psutil.virtual_memory().available > video.maximum_number_of_portraits_in_global_fragments * image_size_bytes:
        num_images = psutil.virtual_memory().available//image_size_bytes
        logging.debug("There is enough RAM to host %i images" %num_images)
    elif psutil.swap_memory().free > video.maximum_number_of_portraits_in_global_fragments * image_size_bytes:
        num_images = psutil.swap_memory().free * .8 // image_size_bytes
        logging.debug("There is enough Swap to host %i images" %num_images)
        logging.warn("WARNING: using swap memory, performance reduced")
    else:
        logging.info("Virtual memory %s" %str(psutil.virtual_memory()))
        logging.info("Swap memory %s" %str(psutil.swap_memory()))
        raise MemoryError('There is not enough free RAM and swap to continue with the process')

    # This loop is to get the predictions in batches so that we do not overload the RAM
    while len(candidates_next_global_fragments) > 0:
        images_in_batch = np.ones((video.maximum_number_of_portraits_in_global_fragments, video.portrait_size[0], video.portrait_size[1])) * np.nan
        individual_fragments_identifiers_already_used = list(individual_fragments_identifiers_already_used)
        num_images_to_assign = 0

        for global_fragment in candidates_next_global_fragments:
            logger.debug("Getting images from next candidate global fragment")
            images_global_fragment, \
            lengths_global_fragment, \
            individual_fragments_identifiers = get_images_and_labels_from_global_fragment_predictions(video, global_fragment,
                                                                                            individual_fragments_identifiers_already_used)
            if len(images_global_fragment) != 0\
                and len(images_global_fragment) < video.maximum_number_of_portraits_in_global_fragments - num_images_to_assign:
                logger.debug("Adding images to the batch")
                # The images of this global fragment fit in this batch
                images_in_batch[num_images_to_assign : num_images_to_assign + len(images_global_fragment)] = images_global_fragment
                lengths.extend(lengths_global_fragment)
                candidate_individual_fragments_identifiers.extend(individual_fragments_identifiers)
                individual_fragments_identifiers_already_used.extend(individual_fragments_identifiers)
                num_images_to_assign += len(images_global_fragment)
                # update list of candidates global fragments
                candidates_next_global_fragments = candidates_next_global_fragments[1:]
            elif len(images_global_fragment) > video.maximum_number_of_portraits_in_global_fragments - num_images_to_assign:
                logger.debug("Enough images in the batch")
                # No more images fit in this batch
                break
            elif len(images_global_fragment) == 0:
                logger.debug("No more images to assign in this global fragment")
                # I skip this global fragment because all the images have been already used
                candidates_next_global_fragments = candidates_next_global_fragments[1:]

        if num_images_to_assign != 0:
            images_in_batch = images_in_batch[:num_images_to_assign]
            logging.debug("shape of images in assignment batch: %s" %str(images_in_batch.shape))
            assigner = assign(net, video, images_in_batch, print_flag = True)
            predictions.extend(assigner._predictions)
            softmax_probs.extend(assigner._softmax_probs)

    assert len(predictions) == len(softmax_probs)
    return predictions, softmax_probs, np.cumsum(lengths)[:-1], candidate_individual_fragments_identifiers
