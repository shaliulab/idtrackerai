from __future__ import absolute_import, division, print_function
import numpy as np
import random
import psutil
from assigner import assign
from trainer import train
from accumulation_manager import AccumulationManager, get_predictions_of_candidates_fragments
from constants import THRESHOLD_EARLY_STOP_ACCUMULATION
import sys
if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger('__main__.accumulator')

def early_stop_criteria_for_accumulation(number_of_accumulated_images, number_of_unique_images_in_global_fragments):
    return number_of_accumulated_images / number_of_unique_images_in_global_fragments

def perform_one_accumulation_step(accumulation_manager,
                                    video,
                                    global_step,
                                    net,
                                    identity_transfer,
                                    GUI_axes = None,
                                    net_properties = None,
                                    plot_flag = False,
                                    save_summaries = False):
    logger.info("accumulation step %s" %accumulation_manager.counter)
    video.accumulation_step = accumulation_manager.counter
    #(we do not take images from individual fragments already used)
    accumulation_manager.get_new_images_and_labels()
    images, labels = accumulation_manager.get_images_and_labels_for_training()
    logger.debug("images: %s" %str(images.shape))
    logger.debug("labels: %s" %str(labels.shape))
    global_step, net,\
    store_validation_accuracy_and_loss_data,\
    store_training_accuracy_and_loss_data = train(video,
                                                    accumulation_manager.list_of_fragments.fragments,
                                                    net, images, labels,
                                                    store_accuracy_and_error = True,
                                                    check_for_loss_plateau = True,
                                                    save_summaries = save_summaries,
                                                    print_flag = False,
                                                    plot_flag = plot_flag,
                                                    global_step = global_step,
                                                    identity_transfer = identity_transfer,
                                                    accumulation_manager = accumulation_manager)
    if net_properties is not None:
        net_properties.setter(global_step)
    # update the set of images used for training
    logger.info("Update images and labels used for training")
    accumulation_manager.update_used_images_and_labels()
    # assign identities fo the global fragments that have been used for training
    logger.info("Assigning identities to accumulated global fragments")
    accumulation_manager.assign_identities_to_fragments_used_for_training()
    # update the list of individual fragments that have been used for training
    logger.info("Update list of individual fragments used for training")
    accumulation_manager.update_list_of_individual_fragments_used()
    # compute ratio of accumulated images and stop if it is above random
    accumulation_manager.ratio_accumulated_images = accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    logger.info("The %f percent of the images has been accumulated" %(accumulation_manager.ratio_accumulated_images * 100))
    if accumulation_manager.ratio_accumulated_images > THRESHOLD_EARLY_STOP_ACCUMULATION:
        logger.debug("Stopping accumulation by early stopping criteria")
        return accumulation_manager.ratio_accumulated_images, store_validation_accuracy_and_loss_data, store_training_accuracy_and_loss_data
    # Set accumulation parameters for rest of the accumulation
    #take images from global fragments not used in training (in the remainder test global fragments)
    logger.info("Get new global fragments for training")
    if any([not global_fragment.used_for_training for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments]):
        logger.info("Generate predictions on candidate global fragments")
        predictions,\
        softmax_probs,\
        indices_to_split,\
        candidate_individual_fragments_identifiers = get_predictions_of_candidates_fragments(net,
                                                                                            video,
                                                                                            accumulation_manager.list_of_fragments.fragments)
        logger.debug('Splitting predictions by fragments...')
        accumulation_manager.split_predictions_after_network_assignment(predictions,
                                                                        softmax_probs,
                                                                        indices_to_split,
                                                                        candidate_individual_fragments_identifiers)
        # assign identities to the global fragments based on the predictions
        logger.info("Checking eligibility criteria and generate the new list of global fragments to accumulate")
        accumulation_manager.get_acceptable_global_fragments_for_training(candidate_individual_fragments_identifiers)
        #Million logs
        # logger.info("Number of candidate global fragments: %i" %len(candidates_next_global_fragments))
        logger.info("Number of non certain global fragments: %i" %accumulation_manager.number_of_noncertain_global_fragments)
        logger.info("Number of randomly assigned global fragments: %i" %accumulation_manager.number_of_random_assigned_global_fragments)
        logger.info("Number of non consistent global fragments: %i " %accumulation_manager.number_of_nonconsistent_global_fragments)
        logger.info("Number of non unique global fragments: %i " %accumulation_manager.number_of_nonunique_global_fragments)
        logger.info("Number of acceptable global fragments: %i " %accumulation_manager.number_of_acceptable_global_fragments)
        logger.info("Number of non certain fragments: %i" %accumulation_manager.number_of_noncertain_fragments)
        logger.info("Number of randomly assigned fragments: %i" %accumulation_manager.number_of_random_assigned_fragments)
        logger.info("Number of non consistent fragments: %i " %accumulation_manager.number_of_nonconsistent_fragments)
        logger.info("Number of non unique fragments: %i " %accumulation_manager.number_of_nonunique_fragments)
        logger.info("Number of acceptable fragments: %i " %accumulation_manager.number_of_acceptable_fragments)
        new_values = [len([global_fragment for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments if global_fragment.used_for_training]),
                        accumulation_manager.number_of_noncertain_global_fragments,
                        accumulation_manager.number_of_random_assigned_global_fragments,
                        accumulation_manager.number_of_nonconsistent_global_fragments,
                        accumulation_manager.number_of_nonunique_global_fragments,
                        np.sum([global_fragment.acceptable_for_training(accumulation_manager.accumulation_strategy) for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments]),
                        store_validation_accuracy_and_loss_data.accuracy,
                        store_validation_accuracy_and_loss_data.individual_accuracy,
                        store_training_accuracy_and_loss_data.accuracy,
                        store_training_accuracy_and_loss_data.individual_accuracy,
                        accumulation_manager.ratio_accumulated_images]
        video.store_accumulation_step_statistics_data(new_values)
        accumulation_manager.update_counter()
    # else:
    #     logger.info("All the global fragments have been used for accumulation")
    #     break

    accumulation_manager.ratio_accumulated_images = accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    video.store_accumulation_statistics_data(video.accumulation_trial)
    return accumulation_manager.ratio_accumulated_images, store_validation_accuracy_and_loss_data, store_training_accuracy_and_loss_data


def accumulate(accumulation_manager,
                video,
                global_step,
                net,
                identity_transfer,
                GUI_axes = None,
                net_properties = None,
                plot_flag = False):
    video.init_accumulation_statistics_attributes()
    accumulation_manager.threshold_early_stop_accumulation = THRESHOLD_EARLY_STOP_ACCUMULATION

    while accumulation_manager.continue_accumulation:
        perform_one_accumulation_step(accumulation_manager,
                        video,
                        global_step,
                        net,
                        identity_transfer,
                        GUI_axes = None,
                        net_properties = None,
                        plot_flag = False)
    return accumulation_manager.ratio_accumulated_images
