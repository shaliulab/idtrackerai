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
from accumulation_manager import AccumulationManager, get_predictions_of_candidates_global_fragments


THRESHOLD_EARLY_STOP_ACCUMULATION = .9995
logger = logging.getLogger('main._accumulator')

def early_stop_criteria_for_accumulation(number_of_accumulated_images, number_of_unique_images_in_global_fragments):
    return number_of_accumulated_images / number_of_unique_images_in_global_fragments

def accumulate(accumulation_manager,
                video,
                blobs,
                global_fragments,
                global_step,
                net,
                knowledge_transfer_from_same_animals,
                get_ith_global_fragment = 0):

    number_of_accumulated_images = 0
    while accumulation_manager.continue_accumulation:
        logger.info("accumulation step %s" %accumulation_manager.counter)
        #get next fragments for accumulation
        accumulation_manager.get_next_global_fragments(get_ith_global_fragment = get_ith_global_fragment)
        logger.debug("Getting %ith global fragment with respect to the distance travelled ordering")
        #get images from the new global fragments
        #(we do not take images from individual fragments already used)
        accumulation_manager.get_new_images_and_labels()
        # update used_for_training flag to True for fragments used
        logger.info("Accumulation step completed. Updating global fragments used for training")
        accumulation_manager.update_global_fragments_used_for_training()
        # update the set of images used for training
        logger.info("Update images and labels used for training")
        accumulation_manager.update_used_images_and_labels()
        # assign identities fo the global fragments that have been used for training
        logger.info("Assigning identities to accumulated global fragments")
        accumulation_manager.assign_identities_to_accumulated_global_fragments()
        # update the list of individual fragments that have been used for training
        logger.info("Update individual fragments used for training")
        accumulation_manager.update_individual_fragments_used()
        # Check uniqueness global_fragments
        logger.info("Check uniqueness of global fragments")
        check_uniquenss_of_global_fragments(global_fragments)
        number_of_accumulated_images += int(accumulation_manager.new_images.shape[0])
        print("\n*******************************************************************************\n")
        ratio_accumulated_images = early_stop_criteria_for_accumulation(number_of_accumulated_images, video.number_of_unique_images_in_global_fragments)
        if ratio_accumulated_images > THRESHOLD_EARLY_STOP_ACCUMULATION:
            logger.debug("Stopping accumulation by early stopping criteria")
            print("\n*******************************************************************************\n")
            return ratio_accumulated_images
        logger.debug('%i new images accumulated' %number_of_accumulated_images)
        #get images for training
        #(we mix images already used with new images)
        images, labels = accumulation_manager.get_images_and_labels_for_training()
        logger.info("the %f percent of the images has been accumulated" %(number_of_accumulated_images / video.number_of_unique_images_in_global_fragments * 100))
        logger.debug("images: %s" %str(images.shape))
        logger.debug("labels: %s" %str(labels.shape))
        #start training
        global_step, net, _ = train(video, blobs,
                                global_fragments,
                                net, images, labels,
                                store_accuracy_and_error = False,
                                check_for_loss_plateau = True,
                                save_summaries = True,
                                print_flag = False,
                                plot_flag = True,
                                global_step = global_step,
                                first_accumulation_flag = accumulation_manager.counter == 0,
                                knowledge_transfer_from_same_animals = knowledge_transfer_from_same_animals)

        # Set accumulation params for rest of the accumulation
        #take images from global fragments not used in training (in the remainder test global fragments)
        logger.info("Get new global fragments for training")
        candidates_next_global_fragments = [global_fragment for global_fragment in global_fragments
                                if not global_fragment.used_for_training]
        logger.info("There are %s candidate global fragments left for accumulation" %str(len(candidates_next_global_fragments)))
        if any([not global_fragment.used_for_training for global_fragment in global_fragments]):
            logger.info("Generate predictions on candidate global fragments")
            predictions,\
            softmax_probs,\
            non_shared_information,\
            indices_to_split,\
            candidate_individual_fragments_identifiers = get_predictions_of_candidates_global_fragments(net,
                                                                                                        video,
                                                                                                        candidates_next_global_fragments,
                                                                                                        accumulation_manager.individual_fragments_used)
            accumulation_manager.split_predictions_after_network_assignment(predictions,
                                                                            softmax_probs,
                                                                            non_shared_information,
                                                                            indices_to_split)
            # assign identities to the global fragments based on the predictions
            logger.info("Checking eligibility criteria and generate the new list of global fragments to accumulate")
            logger.info("Number of candidate global fragments: %i" %len(candidates_next_global_fragments))
            accumulation_manager.assign_identities_and_check_eligibility_for_training_global_fragments(candidate_individual_fragments_identifiers)
            logger.info("Number of non certain global fragments: %i" %accumulation_manager.number_of_noncertain_global_fragments)
            logger.info("Number of randomly assigned global fragments: %i" %accumulation_manager.number_of_random_assigned_global_fragments)
            logger.info("Number of non consistent global fragments: %i " %accumulation_manager.number_of_nonconsistent_global_fragments)
            logger.info("Number of non unique global fragments: %i " %accumulation_manager.number_of_nonunique_global_fragments)
            logger.info("Number of acceptable global fragments: %i " %np.sum([global_fragment.acceptable_for_training for global_fragment in global_fragments]))
            accumulation_manager.update_counter()
        else:
            logger.info("All the global fragments have been used for accumulation")
            break

        ratio_accumulated_images_over_all_unique_images_in_global_fragments = number_of_accumulated_images / video.number_of_unique_images_in_global_fragments
    return ratio_accumulated_images_over_all_unique_images_in_global_fragments
