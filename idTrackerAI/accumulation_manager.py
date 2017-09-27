from __future__ import absolute_import, division, print_function
import numpy as np
import random
import psutil
import logging


from globalfragment import get_images_and_labels_from_global_fragments, \
                            order_global_fragments_by_distance_travelled, \
                            order_global_fragments_by_distance_to_the_first_global_fragment
from statistics_for_assignment import compute_P1_individual_fragment_from_frequencies, \
                                        compute_identification_frequencies_individual_fragment
from assigner import assign
from trainer import train
from globalfragment import check_uniquenss_of_global_fragments

RATIO_OLD = 0.6
RATIO_NEW = 0.4
MAXIMAL_IMAGES_PER_ANIMAL = 3000
CERTAINTY_THRESHOLD = .1 # threshold to select a individual fragment as eligible for training

logger = logging.getLogger("__main__.accumulation_manager")

class AccumulationManager(object):
    def __init__(self, blobs_in_video, global_fragments, number_of_animals, certainty_threshold = CERTAINTY_THRESHOLD):
        """ This class manages the selection of global fragments for accumulation,
        the retrieval of images from the new global fragments, the selection of
        of images for training, the final assignment of identities to the global fragments
        used for training, the temporary assignment of identities to the candidates global fragments
        and the computation of the certainty levels of the individual fragments to check the
        eligability of the candidates global fragments for accumulation.

        :global_fragments list: list of global_fragments objects from the video
        :number_of_animals param: number of animals in the video
        :certainty_threshold param: threshold to set a individual fragment as certain for accumulation
        """
        self.number_of_animals = number_of_animals
        self.global_fragments = global_fragments
        self.blobs_in_video = blobs_in_video
        self.counter = 0
        self.certainty_threshold = certainty_threshold
        self.individual_fragments_used = [] # list with the individual_fragments_identifiers of the individual fragments used for training
        self.identities_of_individual_fragments_used = [] # identities of the individual fragments used for training
        self.P1_vector_of_individual_fragments_used = []
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

    def get_next_global_fragments(self, get_ith_global_fragment = None):
        """ get the global fragments that are going to be added to the current
        list of global fragments used for training"""
        if self.counter == 0:
            logger.info("Getting global fragment for the first accumulation...")
            self.next_global_fragments = [order_global_fragments_by_distance_travelled(self.global_fragments)[get_ith_global_fragment]]
        else:
            logger.info("Getting global fragments...")
            self.next_global_fragments = [global_fragment for global_fragment in self.global_fragments
                                                if global_fragment.acceptable_for_training == True and global_fragment._used_for_training == False]
        logger.info("Number of global fragments for training: %i" %len(self.next_global_fragments))

    def get_new_images_and_labels(self):
        """ get the images and labels of the new global fragments that are going
        to be used for training, this function checks whether the images of a individual
        fragment have been added before"""
        self.new_images, self.new_labels, _, _ = get_images_and_labels_from_global_fragments(self.next_global_fragments,list(self.individual_fragments_used))

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
            global_fragment._used_for_training = True
            global_fragment._acceptable_for_training = False ###NOTE review this flag. This global fragment is technically acceptable for training. Maybe we can work by only setting used_for_training to TRUE

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
        # print("Assigning identities to global fragments and blobs used in this accumulation step...")
        for global_fragment in self.next_global_fragments:
            assert global_fragment.used_for_training == True
            self.check_consistency_of_assignment(global_fragment)
            global_fragment._ids_assigned = np.asarray(global_fragment._temporary_ids) + 1
            [blob.update_identity_in_fragment(identity_in_fragment, assigned_during_accumulation = True, number_of_images_in_fragment = number_of_images_in_fragment)
                for blob, identity_in_fragment, number_of_images_in_fragment in zip(self.blobs_in_video[global_fragment.index_beginning_of_fragment],
                                                        global_fragment._ids_assigned,
                                                        global_fragment._number_of_portraits_per_individual_fragment)]
            global_fragment._P1_vector = [blob._P1_vector for blob in self.blobs_in_video[global_fragment.index_beginning_of_fragment]]

    def check_consistency_of_assignment(self,global_fragment):
        """ this function checks that the identities of the individual fragments in the global
        fragment that is going to be assigned is consistent with the identities of the individual
        fragments that have already been used for training """
        for individual_fragment_identifier, temporary_identity in zip(global_fragment.individual_fragments_identifiers, global_fragment._temporary_ids):
            if individual_fragment_identifier in self.individual_fragments_used:
                index = self.individual_fragments_used.index(individual_fragment_identifier)
                assert self.identities_of_individual_fragments_used[index] == temporary_identity

    def update_individual_fragments_used(self):
        """ Updates the list of individual fragments used in training and their identities.
        If a individual fragment was added before is not added again.
        """
        logging.info("Updating list of individual fragments used for training")
        new_individual_fragments_identifiers_and_id = set([(individual_fragment_identifier, temporary_identity, tuple(P1_vector))
                                                for global_fragment in self.next_global_fragments
                                                for individual_fragment_identifier, temporary_identity, P1_vector in zip(global_fragment.individual_fragments_identifiers, global_fragment._temporary_ids, global_fragment._P1_vector)
                                                    if global_fragment.used_for_training and individual_fragment_identifier not in self.individual_fragments_used])

        new_individual_fragments_identifiers = [out[0] for out in new_individual_fragments_identifiers_and_id]
        new_ids = [out[1] for out in new_individual_fragments_identifiers_and_id]
        new_P1_vectors = [list(out[2]) for out in new_individual_fragments_identifiers_and_id]
        self.individual_fragments_used.extend(new_individual_fragments_identifiers)
        self.identities_of_individual_fragments_used.extend(new_ids)
        self.P1_vector_of_individual_fragments_used.extend(new_P1_vectors)
        logging.info("number of individual fragments used for training: %i" %len(self.individual_fragments_used))
        logging.debug("number of ids of individual fragments used for training: %i" %len(self.identities_of_individual_fragments_used))

    def split_predictions_after_network_assignment(self,predictions, softmax_probs, non_shared_information, indices_to_split):
        """Go back to the CPU"""
        logging.info("Un-stacking predictions for the CPU")
        individual_fragments_predictions = np.split(predictions, indices_to_split)
        individual_fragments_softmax_probs = np.split(softmax_probs, indices_to_split)
        individual_fragments_non_shared_information = np.split(non_shared_information, indices_to_split)
        self.frequencies_of_candidate_individual_fragments = []
        self.P1_vector_of_candidate_individual_fragments = []
        self.median_softmax_of_candidate_individual_fragments = [] #used to compute the certainty on the network's assignment
        self.certainty_of_candidate_individual_fragments = []

        for individual_fragment_predictions, individual_fragment_softmax_probs, individual_fragment_non_shared_information in zip(individual_fragments_predictions, individual_fragments_softmax_probs, individual_fragments_non_shared_information):

            frequencies_of_candidate_individual_fragment = compute_identification_frequencies_individual_fragment(np.asarray(individual_fragment_non_shared_information), np.asarray(individual_fragment_predictions), self.number_of_animals)
            self.frequencies_of_candidate_individual_fragments.append(frequencies_of_candidate_individual_fragment)

            P1_of_candidate_individual_fragment = compute_P1_individual_fragment_from_frequencies(frequencies_of_candidate_individual_fragment)
            self.P1_vector_of_candidate_individual_fragments.append(P1_of_candidate_individual_fragment)

            median_softmax_of_candidate_individual_fragment = compute_median_softmax(individual_fragment_softmax_probs)
            self.median_softmax_of_candidate_individual_fragments.append(median_softmax_of_candidate_individual_fragment)

            certainty_of_individual_fragment = compute_certainty_of_individual_fragment(P1_of_candidate_individual_fragment,median_softmax_of_candidate_individual_fragment)
            self.certainty_of_candidate_individual_fragments.append(certainty_of_individual_fragment)

    def assign_identities_and_check_eligibility_for_training_global_fragments(self, candidate_individual_fragments_identifiers):
        """Assigns identities during test to blobs in global fragments and rank them
        according to the score computed from the certainty of identification and the
        minimum distance travelled"""
        self.candidate_individual_fragments_identifiers = candidate_individual_fragments_identifiers
        self.temporary_individual_fragments_used = []
        self.temporary_identities_of_individual_fragments_used = []
        ordered_global_fragments = order_global_fragments_by_distance_to_the_first_global_fragment(self.global_fragments)
        # ordered_global_fragments = order_global_fragments_by_distance_travelled(self.global_fragments)
        self.number_of_noncertain_global_fragments = 0
        self.number_of_random_assigned_global_fragments = 0
        self.number_of_nonconsistent_global_fragments = 0
        self.number_of_nonunique_global_fragments = 0
        for i, global_fragment in enumerate(ordered_global_fragments):
            if global_fragment.used_for_training == False:
                self.assign_identities_to_test_global_fragment(global_fragment)

    def assign_identities_to_test_global_fragment(self, global_fragment):
        global_fragment._acceptable_for_training = True

        global_fragment._certainties = []
        global_fragment._P1_vector = []
        # Check certainties of the individual fragments in the global fragment
        # logger.debug("*******Checking new global fragment")
        for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:

            if individual_fragment_identifier in self.candidate_individual_fragments_identifiers:
                # if the individual fragment is in the list of candidates we check the certainty
                index_in_candidate_individual_fragments = list(self.candidate_individual_fragments_identifiers).index(individual_fragment_identifier)
                individual_fragment_certainty =  self.certainty_of_candidate_individual_fragments[index_in_candidate_individual_fragments]
                # print("individual_fragment_certainty: ", individual_fragment_certainty)
                if individual_fragment_certainty < self.certainty_threshold:
                    # if the certainty of the individual fragment is not high enough
                    # we set the global fragment not to be acceptable for training
                    global_fragment._acceptable_for_training = False
                    global_fragment._is_certain = False
                    self.number_of_noncertain_global_fragments += 1
                    # logger.debug("The individual fragment %i is not certain enough (certainty %.4f)" %(individual_fragment_identifier, individual_fragment_certainty))
                    break
                else:
                    # if the certainty of the individual fragment is high enough
                    global_fragment._certainties.append(individual_fragment_certainty)
                    individual_fragment_P1_vector = self.P1_vector_of_candidate_individual_fragments[index_in_candidate_individual_fragments]
                    global_fragment._P1_vector.append(individual_fragment_P1_vector)
                    global_fragment._is_certain = True
            elif individual_fragment_identifier in self.individual_fragments_used:
                # if the individual fragment is no in the list of candidates is because it has been assigned
                # and it is in the list of individual_fragments_used. We set the certainty to 1. And we
                global_fragment._certainties.append(1.)
                individual_fragment_identifier_index = list(self.individual_fragments_used).index(individual_fragment_identifier)
                individual_fragment_id = self.identities_of_individual_fragments_used[individual_fragment_identifier_index]
                individual_fragment_P1_vector = self.P1_vector_of_individual_fragments_used[individual_fragment_identifier_index]
                global_fragment._P1_vector.append(individual_fragment_P1_vector)
                global_fragment._is_certain = True
            else:
                logging.warn("Individual fragment not in candidates or in used, this should not happen")

        # Compute identities if the global_fragment is certain
        if global_fragment._acceptable_for_training:
            global_fragment._temporary_ids = np.nan * np.ones(self.number_of_animals)
            # get the index position of the individual fragments ordered by certainty from max to min
            index_individual_fragments_sorted_by_certanity_max_to_min = np.argsort(np.squeeze(np.asarray(global_fragment._certainties)))[::-1]
            # get array of P1 values for the global fragment
            P1_array = np.asarray(global_fragment._P1_vector)
            # print("P1_array_shape: ", P1_array.shape)
            # if P1_array.shape[0] != self.number_of_animals:
            #     print("global_fragment individual fragment identifiers, ", global_fragment.individual_fragments_identifiers)
            # get the maximum P1 of each individual fragment
            P1_max = np.max(P1_array,axis=1)
            # logger.debug("P1 max: %s" %str(P1_max))
            # get the index position of the individual fragments ordered by P1_max from max to min
            index_individual_fragments_sorted_by_P1_max_to_min = np.argsort(P1_max)[::-1]
            # first we set the identities of the individual fragments that have been already used
            for index_individual_fragment, individual_fragment_identifier in enumerate(global_fragment.individual_fragments_identifiers):
                if individual_fragment_identifier in self.individual_fragments_used:
                    index = list(self.individual_fragments_used).index(individual_fragment_identifier)
                    identity = int(self.identities_of_individual_fragments_used[index])
                    global_fragment._temporary_ids[index_individual_fragment] = identity
                    P1_array[index_individual_fragment,:] = 0.
                    P1_array[:,identity] = 0.
                elif individual_fragment_identifier in self.temporary_individual_fragments_used:
                    index = list(self.temporary_individual_fragments_used).index(individual_fragment_identifier)
                    identity = int(self.temporary_identities_of_individual_fragments_used[index])
                    global_fragment._temporary_ids[index_individual_fragment] = identity
                    P1_array[index_individual_fragment,:] = 0.
                    P1_array[:,identity] = 0.

            for index_individual_fragment in index_individual_fragments_sorted_by_P1_max_to_min:
                if np.isnan(global_fragment._temporary_ids[index_individual_fragment]):
                    # if it has not been assigned an identity
                    # logger.debug("-----------------------------------")
                    # logger.debug("index individual fragment")
                    # logger.debug("max of P1 %s" %str(np.max(P1_array[index_individual_fragment,:])))
                    # logger.debug("threshold %s" %str(1./global_fragment._number_of_portraits_per_individual_fragment[index_individual_fragment]))
                    if np.max(P1_array[index_individual_fragment,:]) < 1./global_fragment._number_of_portraits_per_individual_fragment[index_individual_fragment]:
                        global_fragment._acceptable_for_training = False
                        self.number_of_random_assigned_global_fragments += 1
                        # logger.debug("Individual fragment would be assigned randomly")
                        break
                    else:
                        temporary_identity = np.argmax(P1_array[index_individual_fragment,:])
                        if not self.check_consistency_with_coexistent_individual_fragments(global_fragment,index_individual_fragment,temporary_identity):
                            global_fragment._acceptable_for_training = False
                            # logger.debug("Individual fragment is not consistent")
                            self.number_of_nonconsistent_global_fragments += 1
                            break
                        else:
                            global_fragment._temporary_ids[index_individual_fragment] = int(temporary_identity)
                            P1_array[index_individual_fragment,:] = 0.
                            P1_array[:,temporary_identity] = 0.

            # Check if the global fragment is unique after assigning the identities
            if global_fragment._acceptable_for_training:
                if not global_fragment.is_unique:
                    global_fragment._acceptable_for_training = False
                    # logger.debug("The global fragment is not unique")
                    self.number_of_nonunique_global_fragments += 1
                else:
                    global_fragment._temporary_ids = np.asarray(global_fragment._temporary_ids).astype('int')
                    global_fragment._accumulation_step = self.counter
                    # logger.debug("The global fragment will be accumulated")
                    for individual_fragment_identifier, temporary_identity in zip(global_fragment.individual_fragments_identifiers, global_fragment._temporary_ids):
                        if individual_fragment_identifier not in self.temporary_individual_fragments_used and individual_fragment_identifier not in self.individual_fragments_used:
                            self.temporary_individual_fragments_used.append(individual_fragment_identifier)
                            self.temporary_identities_of_individual_fragments_used.append(temporary_identity)

    def get_blob_from_global_fragment_and_individual_fragment_identifier(self, global_fragment, individual_fragment_identifier):
        frame_number = global_fragment.index_beginning_of_fragment
        blobs_in_frame = self.blobs_in_video[frame_number]
        return [blob for blob in blobs_in_frame if blob.fragment_identifier == individual_fragment_identifier][0]


    def check_consistency_with_coexistent_individual_fragments(self, global_fragment, index_individual_fragment, temporary_identity):
        individual_fragment_identifier = global_fragment.individual_fragments_identifiers[index_individual_fragment]
        blob_to_check = self.get_blob_from_global_fragment_and_individual_fragment_identifier(global_fragment,individual_fragment_identifier)
        _, fragment_identifiers_of_coexisting_fragments = blob_to_check.get_coexisting_blobs_in_fragment(self.blobs_in_video)

        for fragment_identifier in fragment_identifiers_of_coexisting_fragments:
            if fragment_identifier in self.individual_fragments_used:
                index = list(self.individual_fragments_used).index(fragment_identifier)
                identity = int(self.identities_of_individual_fragments_used[index])
                if identity == temporary_identity:
                    return False
            elif fragment_identifier in self.temporary_individual_fragments_used:
                index = list(self.temporary_individual_fragments_used).index(fragment_identifier)
                identity = int(self.temporary_identities_of_individual_fragments_used[index])
                if identity == temporary_identity:
                    return False
        return True

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

def compute_median_softmax(individual_fragment_softmax_probs):
    individual_fragment_softmax_probs = np.asarray(individual_fragment_softmax_probs)
    max_softmax_probs = np.max(individual_fragment_softmax_probs, axis = 1)
    argmax_softmax_probs = np.argmax(individual_fragment_softmax_probs, axis = 1)
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
    # print("two best P1_vector values: ", np.multiply(sorted_p1_vector,sorted_softmax_probs)[-2:])
    certainty = np.diff(np.multiply(sorted_p1_vector,sorted_softmax_probs)[-2:])/np.sum(sorted_p1_vector[-2:])
    # print("certainty: ", certainty)
    return certainty

""" Get predictions of individual fragments in candidates global fragments"""
def get_predictions_of_candidates_global_fragments(net,video,candidates_next_global_fragments,individual_fragments_identifiers_already_used = []):

    def get_images_and_labels_from_global_fragment_predictions(video, global_fragment, individual_fragments_identifiers_already_used = []):
        images = np.ones((video.maximum_number_of_portraits_in_global_fragments, video.portrait_size[0], video.portrait_size[1]))
        labels = np.ones((video.maximum_number_of_portraits_in_global_fragments, 1))
        non_shared_information = np.ones((video.maximum_number_of_portraits_in_global_fragments))
        lengths = []
        individual_fragments_identifiers = []
        num_images = 0

        for i, portraits in enumerate(global_fragment.portraits):
            if global_fragment.individual_fragments_identifiers[i] not in individual_fragments_identifiers_already_used :
                assert len(portraits) == len(global_fragment.non_shared_information[i])
                images[num_images : len(portraits) + num_images] = np.asarray(portraits)
                labels[num_images : len(portraits) + num_images] = np.asarray(global_fragment._temporary_ids[i] * len(portraits))
                non_shared_information[num_images : len(portraits) + num_images] = np.asarray(global_fragment.non_shared_information[i])
                lengths.append(len(portraits))
                individual_fragments_identifiers.append(global_fragment.individual_fragments_identifiers[i])
                num_images += len(portraits)

        images = images[:num_images]
        labels = labels[:num_images]
        non_shared_information = non_shared_information[:num_images]
        return images, labels, non_shared_information, lengths, individual_fragments_identifiers

    predictions = []
    softmax_probs = []
    non_shared_information = []
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
        non_shared_information_in_batch = np.ones((video.maximum_number_of_portraits_in_global_fragments)) * np.nan
        individual_fragments_identifiers_already_used = list(individual_fragments_identifiers_already_used)
        num_images_to_assign = 0
        for global_fragment in candidates_next_global_fragments:
            images_global_fragment, \
            _, \
            non_shared_information_global_fragment,\
            lengths_global_fragment, \
            individual_fragments_identifiers = get_images_and_labels_from_global_fragment_predictions(video, global_fragment,
                                                                                            individual_fragments_identifiers_already_used)

            if len(images_global_fragment) != 0\
                and len(images_global_fragment) < video.maximum_number_of_portraits_in_global_fragments - num_images_to_assign:
                # The images of this global fragment fit in this batch
                images_in_batch[num_images_to_assign : num_images_to_assign + len(images_global_fragment)] = images_global_fragment
                non_shared_information_in_batch[num_images_to_assign : num_images_to_assign + len(images_global_fragment)] = non_shared_information_global_fragment
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
            non_shared_information_in_batch = non_shared_information_in_batch[:num_images_to_assign]
            assigner = assign(net, video, images_in_batch, print_flag = True)
            predictions.extend(assigner._predictions)
            softmax_probs.extend(assigner._softmax_probs)
            non_shared_information.extend(non_shared_information_in_batch)
    assert len(predictions) == len(softmax_probs) == len(non_shared_information)
    return predictions, softmax_probs, non_shared_information, np.cumsum(lengths)[:-1], candidate_individual_fragments_identifiers
