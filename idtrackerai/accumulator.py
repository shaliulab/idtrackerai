# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking unmarked individuals in large collectives
 

from __future__ import absolute_import, division, print_function
import numpy as np
import random
import psutil
from idtrackerai.assigner import assign
from idtrackerai.trainer import train
from idtrackerai.accumulation_manager import AccumulationManager, get_predictions_of_candidates_fragments
from idtrackerai.constants import  THRESHOLD_EARLY_STOP_ACCUMULATION
import sys
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger('__main__.accumulator')

def early_stop_criteria_for_accumulation(number_of_accumulated_images, number_of_unique_images_in_global_fragments):
    """A particularly succesful accumulation causes an early stop of the training
    and accumulaton process. This function returns the value, expressed as a ratio
    that is evaluated to trigger this behaviour.

    Parameters
    ----------
    number_of_accumulated_images : int
        Number of images used during the accumulation process (the labelled dataset used to train the network
        is subsampled from this set of images).
    number_of_unique_images_in_global_fragments : int
        Total number of accumulable images.

    Returns
    -------
    float
        Ratio of accumulated images over accumulable images

    """
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
                identity_transfer):
    """take care of managing  the process of accumulation
    of labelled images. Such process, in complex video, allows us to train  the
    idCNN (or whatever function approximator passed in input as `net`).

    Parameters
    ----------
    accumulation_manager : <accumulation_manager.AccumulationManager object>
        Description of parameter `accumulation_manager`.
    video : <video.Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    global_step : int
        network epoch counter
    net : <net.ConvNetwork object>
        Convolutional neural network object created according to net.params
    identity_transfer : bool
        If true the identity of the individual is also tranferred

    Returns
    -------
    float
        Ratio of accumulated images

    See Also
    --------
    early_stop_criteria_for_accumulation

    """
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
