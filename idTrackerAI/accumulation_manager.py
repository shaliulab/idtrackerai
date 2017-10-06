from __future__ import absolute_import, division, print_function
from pprint import pprint
import numpy as np
import random
import psutil
import logging


from list_of_global_fragments import get_images_and_labels_from_global_fragments
# from statistics_for_assignment import compute_P1_individual_fragment_from_frequencies, \
                                        # compute_identification_frequencies_individual_fragment
from assigner import assign
from trainer import train

RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = .1 # threshold to select a individual fragment as eligible for training

logger = logging.getLogger("__main__.accumulation_manager")

class AccumulationManager(object):
    def __init__(self, video, fragments, global_fragments, certainty_threshold = CERTAINTY_THRESHOLD):
        """ This class manages the selection of global fragments for accumulation,
        the retrieval of images from the new global fragments, the selection of
        of images for training, the final assignment of identities to the global fragments
        used for training, the temporary assignment of identities to the candidates global fragments
        and the computation of the certainty levels of the individual fragments to check the
        eligability of the candidates global fragments for accumulation.

        :fragments: list of individual and crossing fragments
        :global_fragments list: list of global_fragments objects from the video
        :number_of_animals param: number of animals in the video
        :certainty_threshold param: threshold to set a individual fragment as certain for accumulation
        """
        self.video = video
        self.number_of_animals = video.number_of_animals
        self.fragments = fragments
        self.global_fragments = global_fragments
        self.counter = 0
        self.certainty_threshold = certainty_threshold
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
        if not any([(global_fragment.acceptable_for_training and not global_fragment.used_for_training) for global_fragment in self.global_fragments]):
            return False
        else:
            return True

    def update_counter(self):
        """ updates the counter of the accumulation"""
        self.counter += 1

    @staticmethod
    def give_me_frequencies_first_fragment_accumulated(i, number_of_animals, fragment):
        frequencies = np.zeros(number_of_animals)
        frequencies[i] = fragment.number_of_images
        return frequencies

    def get_next_global_fragments(self, get_ith_global_fragment = None):
        """ get the global fragments that are going to be added to the current
        list of global fragments used for training"""

        if self.counter == 0:
            logger.info("Getting global fragment for the first accumulation...")
            # At this point global fragments are already ordered according to minmax distance travelled
            self.next_global_fragments = [self.global_fragments[get_ith_global_fragment]]
            [(setattr(fragment, '_temporary_id', i),
                setattr(fragment, '_frequencies', self.give_me_frequencies_first_fragment_accumulated(i, self.number_of_animals, fragment)),
                setattr(fragment, '_is_certain', True),
                setattr(fragment, '_certainty', 1.),
                setattr(fragment, '_P1_vector', fragment.compute_P1_from_frequencies(fragment.frequencies)))
                for i, fragment in enumerate(self.next_global_fragments[0].individual_fragments)]
        else:
            logger.info("Getting global fragments...")
            self.next_global_fragments = [global_fragment for global_fragment in self.global_fragments
                                                if global_fragment.acceptable_for_training == True and global_fragment.used_for_training == False]
        logger.info("Number of global fragments for training: %i" %len(self.next_global_fragments))

    def get_new_images_and_labels(self):
        """ get the images and labels of the new global fragments that are going
        to be used for training, this function checks whether the images of a individual
        fragment have been added before"""
        self.new_images, self.new_labels, _, _ = get_images_and_labels_from_global_fragments(self.fragments,
                                                                    self.next_global_fragments,
                                                                    list(self.individual_fragments_used))

        if self.new_images is not None:
            logger.info("New images for training: %s %s"  %(str(self.new_images.shape), str(self.new_labels.shape)))
        else:
            logger.info("There are no new images in this accumulation")
        if self.used_images is not None:
            logger.info("Old images for training: %s %s" %(str(self.used_images.shape), str(self.used_labels.shape)))

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

    def update_global_fragments_used_for_training(self):
        """ Once a global fragment has been used for training we set the flags
        used_for_training to TRUE and acceptable_for_training to FALSE"""
        logger.debug("Setting used_for_training to TRUE and acceptable for training to FALSE for the global fragments already used...")
        for global_fragment in self.next_global_fragments:
            [(setattr(fragment,'_used_for_training',True),setattr(fragment,'_acceptable_for_training',True))
                                                for fragment in global_fragment.individual_fragments]

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

    def assign_identities_to_accumulated_global_fragments(self):
        """ assign the identities to the global fragments used for training and
        to the blobs that belong to these global fragments. This function checks
        that the identities of the individual fragments in the global fragment
        are consistent with the previously assigned identities"""
        for global_fragment in self.next_global_fragments:
            assert global_fragment.used_for_training == True
            [setattr(fragment, '_identity', getattr(fragment, 'temporary_id') + 1) for fragment in global_fragment.individual_fragments]

    def update_individual_fragments_used(self):
        """ Updates the list of individual fragments used in training and their identities.
        If a individual fragment was added before is not added again.
        """
        logging.info("Updating list of individual fragments used for training")
        new_individual_fragments_identifiers = list(set([fragment.identifier for global_fragment in self.next_global_fragments
                                                            for fragment in global_fragment.individual_fragments
                                                            if global_fragment.used_for_training
                                                            and fragment.identifier not in self.individual_fragments_used]))
        self.individual_fragments_used.extend(new_individual_fragments_identifiers)
        logging.info("number of individual fragments used for training: %i" %sum([fragment.used_for_training for fragment in self.fragments]))

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
            self.fragments[index].compute_identification_statistics(np.asarray(individual_fragment_predictions),\
                                                                            individual_fragment_softmax_probs)

    def assign_identities_and_check_eligibility_for_training_global_fragments(self, candidate_individual_fragments_identifiers):
        """Assigns identities during test to blobs in global fragments and rank them
        according to the score computed from the certainty of identification and the
        minimum distance travelled"""
        self.candidate_individual_fragments_identifiers = candidate_individual_fragments_identifiers
        self.temporary_individual_fragments_used = []
        self.temporary_identities_of_individual_fragments_used = []
        self.number_of_noncertain_global_fragments = 0
        self.number_of_random_assigned_global_fragments = 0
        self.number_of_nonconsistent_global_fragments = 0
        self.number_of_nonunique_global_fragments = 0
        for i, global_fragment in enumerate(self.global_fragments):
            if global_fragment.used_for_training == False:
                self.assign_identities_to_test_global_fragment(global_fragment)

    def reset_identity_of_non_accumulated_individual_fragments(self, global_fragment):
        for fragment in global_fragment.individual_fragments:
            if fragment.identifier not in self.temporary_individual_fragments_used \
            and fragment.identifier not in self.individual_fragments_used:
                fragment._temporary_id = None
                fragment._acceptable_for_training = False

    def assign_identities_to_test_global_fragment(self, global_fragment):
        # Check certainties of the individual fragments in the global fragment
        # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:
        [setattr(fragment,'_acceptable_for_training', True) for fragment in global_fragment.individual_fragments]
        for fragment in global_fragment.individual_fragments:
            if fragment.identifier in self.candidate_individual_fragments_identifiers:
                if fragment.certainty < self.certainty_threshold:
                    # if the certainty of the individual fragment is not high enough
                    # we set the global fragment not to be acceptable for training
                    self.reset_identity_of_non_accumulated_individual_fragments(global_fragment)
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
        if global_fragment.acceptable_for_training:
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
                        self.reset_identity_of_non_accumulated_individual_fragments(global_fragment)
                        break
                    else:
                        temporary_id = np.argmax(P1_array[index_individual_fragment,:])
                        if not fragment.check_consistency_with_coexistent_individual_fragments(temporary_id):
                            self.reset_identity_of_non_accumulated_individual_fragments(global_fragment)
                            fragment._non_consistent = True
                            self.number_of_nonconsistent_global_fragments += 1
                            break
                        else:
                            fragment._temporary_id = int(temporary_id)
                            P1_array[index_individual_fragment,:] = 0.
                            P1_array[:,temporary_id] = 0.

            # Check if the global fragment is unique after assigning the identities
            if global_fragment.acceptable_for_training:
                if not global_fragment.is_unique:
                    # set acceptable_for_training to False and temporary_id to None for all the individual_fragments
                    # that had not been accumulated before (i.e. not in temporary_individual_fragments_used or individual_fragments_used)
                    self.reset_identity_of_non_accumulated_individual_fragments(global_fragment)
                    self.number_of_nonunique_global_fragments += 1
                else:
                    global_fragment._accumulation_step = self.counter

                    for fragment in global_fragment.individual_fragments:
                        if fragment.identifier not in self.temporary_individual_fragments_used \
                        and fragment.identifier not in self.individual_fragments_used:
                            self.temporary_individual_fragments_used.append(fragment.identifier)

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
def get_predictions_of_candidates_global_fragments(net,video,candidates_next_global_fragments,individual_fragments_identifiers_already_used = []):

    def get_images_and_labels_from_global_fragment_predictions(video, global_fragment, individual_fragments_identifiers_already_used = []):
        images = np.ones((video.maximum_number_of_portraits_in_global_fragments, video.portrait_size[0], video.portrait_size[1]))
        lengths = []
        individual_fragments_identifiers = []
        num_images = 0

        for fragment in global_fragment.individual_fragments:
            if fragment.identifier not in individual_fragments_identifiers_already_used:
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
            images_global_fragment, \
            lengths_global_fragment, \
            individual_fragments_identifiers = get_images_and_labels_from_global_fragment_predictions(video, global_fragment,
                                                                                            individual_fragments_identifiers_already_used)

            if len(images_global_fragment) != 0\
                and len(images_global_fragment) < video.maximum_number_of_portraits_in_global_fragments - num_images_to_assign:
                # The images of this global fragment fit in this batch
                images_in_batch[num_images_to_assign : num_images_to_assign + len(images_global_fragment)] = images_global_fragment
                lengths.extend(lengths_global_fragment)
                candidate_individual_fragments_identifiers.extend(individual_fragments_identifiers)
                individual_fragments_identifiers_already_used.extend(individual_fragments_identifiers)
                num_images_to_assign += len(images_global_fragment)
                # update list of candidates global fragments
                candidates_next_global_fragments = candidates_next_global_fragments[1:]
            elif len(images_global_fragment) > video.maximum_number_of_portraits_in_global_fragments - num_images_to_assign:
                # No more images fit in this batch
                break
            elif len(images_global_fragment) == 0:
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
