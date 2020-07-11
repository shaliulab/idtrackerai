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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from confapp import conf

from idtrackerai.network.data_sets.crossings_dataset import (
    get_train_validation_and_toassign_blobs,
)
from idtrackerai.network.data_loaders.crossings_dataloader import (
    get_training_data_loaders,
)
from idtrackerai.network.crossings_detector_model.network_params_crossings import (
    NetworkParams_crossings,
)
from idtrackerai.network.crossings_detector_model.trainer_crossing_detector import (
    TrainDeepCrossing,
)
from idtrackerai.network.crossings_detector_model.predictor_crossing_detector import (
    GetPredictionCrossigns,
)
from idtrackerai.network.crossings_detector_model.stop_training_criteria_crossings import (
    Stop_Training,
)


from idtrackerai.network.learners.learners import Learner_Classification

import logging

logger = logging.getLogger("__main__.crossing_detector")


def detect_crossings(
    list_of_blobs,
    video,
    model_area,
    use_network=True,
    return_store_objects=False,
    plot_flag=False,
):
    """Classify all blobs in the video as bing crossings or individuals.

    Parameters
    ----------
    list_of_blobs : <ListOfBlobs object>
        Collection of the Blob objects extracted from the video
    video :  <Video object>
        Object containing all the parameters of the video.
    model_area : function
        Model of the area of a single individual
    use_network : bool
        If True the Deep Crossing Detector is used to distinguish between
        individuals and crossings images. Otherwise only the model area is applied
    return_store_objects : bool
        If True the instantiations of the class :class:`.Store_Accuracy_and_Loss`
        are returned by the function
    plot_flag : bool
        If True a figure representing the values of the loss function, accuracy
        and accuracy per class for both the training and validation set.

    Returns
    -------

    trainer or list_of_blobs : TrainDeepCrossing or ListOfBlobs()
    """

    if video.number_of_animals > 1:
        logger.info(
            "Discriminating blobs representing individuals from blobs associated to crossings"
        )
        list_of_blobs.apply_model_area_to_video(
            video,
            model_area,
            video.identification_image_size[0],
            video.number_of_animals,
        )

        if use_network:
            video.create_crossings_detector_folder()
            logger.info("Get list of blobs for training, validation and test")
            (
                train_blobs,
                val_blobs,
                toassign_blobs,
            ) = get_train_validation_and_toassign_blobs(list_of_blobs)

            if (
                len(train_blobs["crossings"])
                > conf.MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR
            ):
                video._there_are_crossings = True
                logger.info(
                    "There are enough crossings to train the crossing detector"
                )
                train_loader, val_loader = get_training_data_loaders(
                    video, train_blobs, val_blobs
                )
                logger.info("Setting crossing detector network parameters")
                network_params = NetworkParams_crossings(
                    number_of_classes=2,
                    architecture="DCD",
                    save_folder=video.crossings_detector_folder,
                    saveid="",
                    model_name="crossing_detector",
                    image_size=video.identification_image_size,
                    loss="CE",
                    print_freq=-1,
                    use_gpu=True,
                    optimizer="Adam",
                    schedule=[30, 60],
                    optim_args={"lr": conf.LEARNING_RATE_DCD},
                    apply_mask=False,
                    dataset="supervised",
                    skip_eval=False,
                    epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_DCD,
                    plot_flag=False,
                    return_store_objects=False,
                )
                logger.info("Setting training criterion")
                criterion = nn.CrossEntropyLoss(
                    weight=torch.tensor(train_blobs["weights"])
                )
                logger.info("Setting learner class")
                learner_class = Learner_Classification
                logger.info("Creating model")
                crossing_detector_model = learner_class.create_model(
                    network_params
                )

                if network_params.use_gpu:
                    logger.info("Sending model and criterion to GPU")
                    torch.cuda.set_device(0)
                    cudnn.benchmark = True  # make it train faster
                    crossing_detector_model = crossing_detector_model.cuda()
                    criterion = criterion.cuda()

                logger.info("Setting optimizer")
                optimizer = torch.optim.__dict__[network_params.optimizer](
                    crossing_detector_model.parameters(),
                    **network_params.optim_args
                )
                logger.info("Setting scheduler")
                scheduler = MultiStepLR(
                    optimizer, milestones=network_params.schedule, gamma=0.1
                )
                logger.info("Setting the learner")
                learner = learner_class(
                    crossing_detector_model, criterion, optimizer, scheduler
                )
                logger.info("Setting the stopping criteria")
                # set criteria to stop the training
                stop_training = Stop_Training(
                    check_for_loss_plateau=True,
                    num_epochs=network_params.epochs,
                )
                logger.info("Training crossing detector")
                trainer = TrainDeepCrossing(
                    learner,
                    train_loader,
                    val_loader,
                    network_params,
                    stop_training,
                )
                logger.info("Crossing detector training finished")

                if not trainer.model_diverged:
                    del train_loader
                    del val_loader

                    logger.info(
                        "=> Load model weights: {}".format(
                            trainer.best_model_path
                        )
                    )
                    model_state = torch.load(
                        trainer.best_model_path,
                        map_location=lambda storage, loc: storage,
                    )  # Load to CPU as the default!
                    crossing_detector_model.load_state_dict(
                        model_state, strict=True
                    )
                    logger.info("=> Load Done")

                    logger.info("Classify individuals and crossings")
                    crossings_predictor = GetPredictionCrossigns(
                        video,
                        crossing_detector_model,
                        toassign_blobs,
                        network_params,
                    )
                    predictions = crossings_predictor.get_all_predictions()

                    print(
                        len([p for p in predictions if p == 0]), "individuals"
                    )
                    print(len([p for p in predictions if p == 1]), "crossings")
                    for blob, prediction in zip(toassign_blobs, predictions):
                        if prediction == 1:
                            blob._is_a_crossing = True
                            blob._is_an_individual = False
                        else:
                            blob._is_a_crossing = False
                            blob._is_an_individual = True
                    logger.debug("Freeing memory. Test crossings set deleted")

                    list_of_blobs.update_identification_image_dataset_with_crossings(
                        video
                    )

                    if return_store_objects:
                        return trainer
            else:
                logger.debug(
                    "There are not enough crossings to train the crossing detector"
                )
                video._there_are_crossings = False
                return list_of_blobs

    elif video.number_of_animals == 1:
        video._there_are_crossings = False
        return list_of_blobs
