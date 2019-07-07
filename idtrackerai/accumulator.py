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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import sys

import numpy as np

from idtrackerai.trainer import train
from idtrackerai.accumulation_manager import get_predictions_of_candidates_fragments
from confapp import conf


if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger('__main__.accumulator')


def perform_one_accumulation_step(accumulation_manager,
                                  video,
                                  global_step,
                                  net,
                                  GUI_axes=None,
                                  net_properties=None,
                                  plot_flag=False,
                                  save_summaries=False):
    logger.info("accumulation step %s" % accumulation_manager.counter)
    video.accumulation_step = accumulation_manager.counter
    accumulation_manager.get_new_images_and_labels()
    images, labels = accumulation_manager.get_images_and_labels_for_training()
    logger.debug("images: %s" % str(images.shape))
    logger.debug("labels: %s" % str(labels.shape))
    global_step, net,\
        store_validation_accuracy_and_loss_data,\
        store_training_accuracy_and_loss_data = \
        train(video,
              accumulation_manager.list_of_fragments.fragments,
              net, images, labels,
              store_accuracy_and_error=True,
              check_for_loss_plateau=True,
              save_summaries=save_summaries,
              print_flag=False,
              plot_flag=plot_flag,
              global_step=global_step,
              accumulation_manager=accumulation_manager)
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
    if accumulation_manager.ratio_accumulated_images > conf.THRESHOLD_EARLY_STOP_ACCUMULATION:
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
