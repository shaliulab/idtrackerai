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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)


import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from confapp import conf
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai.tracker.accumulation_manager import (
    get_predictions_of_candidates_fragments,
)
from idtrackerai.tracker.dataset.identification_dataloader import (
    get_training_data_loaders,
)
from idtrackerai.tracker.dataset.identification_dataset import (
    split_data_train_and_validation,
)
from idtrackerai.tracker.network.stop_training_criteria import (
    Stop_Training,
)
from idtrackerai.tracker.network.trainer import (
    TrainIdentification,
)

logger = logging.getLogger("__main__.accumulator")


def perform_one_accumulation_step(
    accumulation_manager,
    video,
    identification_model,
    learner_class,
    network_params=None,
):

    # Set accumulation counter
    logger.info("accumulation step %s" % accumulation_manager.counter)
    video.accumulation_step = accumulation_manager.counter

    # Get images for training
    accumulation_manager.get_new_images_and_labels()
    images, labels = accumulation_manager.get_images_and_labels_for_training()
    train_data, val_data = split_data_train_and_validation(
        images, labels, validation_proportion=conf.VALIDATION_PROPORTION
    )

    logger.debug("images: {} {}".format(images.shape, images.dtype))
    logger.debug("labels: %s" % str(labels.shape))

    logger.info("Training with {} images".format(len(train_data["images"])))
    logger.info("Validating with {} images".format(len(val_data["images"])))
    assert len(val_data["images"]) > 0

    # Set data loaders
    train_loader, val_loader = get_training_data_loaders(
        video, train_data, val_data
    )

    # Set criterion
    logger.info("Setting training criterion")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(train_data["weights"]))

    # Send model and criterion to GPU
    if network_params.use_gpu:
        logger.info("Sending model and criterion to GPU")
        torch.cuda.set_device(0)
        cudnn.benchmark = True  # make it train faster
        identification_model = identification_model.cuda()
        criterion = criterion.cuda()

    # Set optimizer
    logger.info("Setting optimizer")
    optimizer = torch.optim.__dict__[network_params.optimizer](
        identification_model.parameters(), **network_params.optim_args
    )

    # Set scheduler
    logger.info("Setting scheduler")
    scheduler = MultiStepLR(
        optimizer, milestones=network_params.schedule, gamma=0.1
    )

    # Set learner
    logger.info("Setting the learner")
    learner = learner_class(
        identification_model, criterion, optimizer, scheduler
    )

    # Set stopping criteria
    logger.info("Setting the stopping criteria")
    # set criteria to stop the training
    stop_training = Stop_Training(
        network_params.number_of_classes,
        check_for_loss_plateau=True,
        first_accumulation_flag=video is None or video.accumulation_step == 0,
    )

    logger.info("Training identification network")
    trainer = TrainIdentification(
        learner,
        train_loader,
        val_loader,
        network_params,
        stop_training,
        accumulation_manager=accumulation_manager,
    )
    logger.info("Identification network trained")

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
    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    )
    logger.info(
        "The %f percent of the images has been accumulated"
        % (accumulation_manager.ratio_accumulated_images * 100)
    )
    if (
        accumulation_manager.ratio_accumulated_images
        > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
    ):
        logger.debug("Stopping accumulation by early stopping criteria")
        return accumulation_manager.ratio_accumulated_images

    # Set accumulation parameters for rest of the accumulation
    # take images from global fragments not used in training (in the remainder test global fragments)
    logger.info("Get new global fragments for training")
    if any(
        [
            not global_fragment.used_for_training
            for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
        ]
    ):
        logger.info("Generate predictions on candidate global fragments")
        (
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_individual_fragments_identifiers,
        ) = get_predictions_of_candidates_fragments(
            identification_model,
            video,
            network_params,
            accumulation_manager.list_of_fragments.fragments,
        )
        logger.debug("Splitting predictions by fragments...")
        accumulation_manager.split_predictions_after_network_assignment(
            predictions,
            softmax_probs,
            indices_to_split,
            candidate_individual_fragments_identifiers,
        )
        # assign identities to the global fragments based on the predictions
        logger.info(
            "Checking eligibility criteria and generate the new list of global fragments to accumulate"
        )
        accumulation_manager.get_acceptable_global_fragments_for_training(
            candidate_individual_fragments_identifiers
        )
        # Million logs

        logger.info(
            "Number of non certain global fragments: %i"
            % accumulation_manager.number_of_noncertain_global_fragments
        )
        logger.info(
            "Number of randomly assigned global fragments: %i"
            % accumulation_manager.number_of_random_assigned_global_fragments
        )
        logger.info(
            "Number of non consistent global fragments: %i "
            % accumulation_manager.number_of_nonconsistent_global_fragments
        )
        logger.info(
            "Number of non unique global fragments: %i "
            % accumulation_manager.number_of_nonunique_global_fragments
        )
        logger.info(
            "Number of acceptable global fragments: %i "
            % accumulation_manager.number_of_acceptable_global_fragments
        )
        logger.info(
            "Number of non certain fragments: %i"
            % accumulation_manager.number_of_noncertain_fragments
        )
        logger.info(
            "Number of randomly assigned fragments: %i"
            % accumulation_manager.number_of_random_assigned_fragments
        )
        logger.info(
            "Number of non consistent fragments: %i "
            % accumulation_manager.number_of_nonconsistent_fragments
        )
        logger.info(
            "Number of non unique fragments: %i "
            % accumulation_manager.number_of_nonunique_fragments
        )
        logger.info(
            "Number of acceptable fragments: %i "
            % accumulation_manager.number_of_acceptable_fragments
        )

        new_values = [
            len(
                [
                    global_fragment
                    for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
                    if global_fragment.used_for_training
                ]
            ),
            accumulation_manager.number_of_noncertain_global_fragments,
            accumulation_manager.number_of_random_assigned_global_fragments,
            accumulation_manager.number_of_nonconsistent_global_fragments,
            accumulation_manager.number_of_nonunique_global_fragments,
            np.sum(
                [
                    global_fragment.acceptable_for_training(
                        accumulation_manager.accumulation_strategy
                    )
                    for global_fragment in accumulation_manager.list_of_global_fragments.global_fragments
                ]
            ),
            accumulation_manager.ratio_accumulated_images,
        ]
        video.store_accumulation_step_statistics_data(new_values)
        accumulation_manager.update_counter()

    accumulation_manager.ratio_accumulated_images = (
        accumulation_manager.list_of_fragments.compute_ratio_of_images_used_for_training()
    )
    video.store_accumulation_statistics_data(video.accumulation_trial)
    return accumulation_manager.ratio_accumulated_images
