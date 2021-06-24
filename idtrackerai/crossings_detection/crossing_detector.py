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

from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from confapp import conf
from torch.optim.lr_scheduler import MultiStepLR

from idtrackerai.crossings_detection.network.network_params_crossings import (
    NetworkParams_crossings,
)
from idtrackerai.crossings_detection.network.predictor_crossing_detector import (
    GetPredictionCrossigns,
)
from idtrackerai.crossings_detection.network.stop_training_criteria_crossings import (
    Stop_Training,
)
from idtrackerai.crossings_detection.network.trainer_crossing_detector import (
    TrainDeepCrossing,
)
from idtrackerai.crossings_detection.dataset.crossings_dataloader import (
    get_training_data_loaders,
)
from idtrackerai.crossings_detection.dataset.crossings_dataset import (
    get_train_validation_and_eval_blobs,
)
from idtrackerai.network.learners.learners import Learner_Classification
from idtrackerai.network.utils.utils import weights_xavier_init

logger = logging.getLogger("__main__.crossing_detector")


def _apply_area_and_unicity_heuristics(
    list_of_blobs,
    number_of_animals,
    model_area,
):
    """Applies `model_area` to every blob extracted from video

    Parameters
    ----------
    list_of_blobs : ListOfBlobs
    model_area : <ModelArea object>
        See :class:`~model_area.ModelArea`
    number_of_animals : int
        number of animals to be tracked
    """
    for blobs_in_frame in tqdm(
        list_of_blobs.blobs_in_video, desc="Applying model area"
    ):
        number_of_blobs = len(blobs_in_frame)
        unicity_cond = number_of_blobs == number_of_animals
        for blob in blobs_in_frame:
            blob.is_an_individual = unicity_cond or model_area(blob.area)


def detect_crossings(
    list_of_blobs,
    video,
    model_area,
):
    """Classify all blobs in the video as being crossings or individuals.

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

    Returns
    -------

    trainer or list_of_blobs : TrainDeepCrossing or ListOfBlobs()
    """

    logger.info("Classifying blobs as individuals or crossings")
    _apply_area_and_unicity_heuristics(
        list_of_blobs,
        video.user_defined_parameters["number_of_animals"],
        model_area,
    )

    logger.info("Get list of blobs for training, validation and eval")
    (
        train_blobs,
        val_blobs,
        eval_blobs,
    ) = get_train_validation_and_eval_blobs(list_of_blobs)

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
        )
        logger.info("Setting training criterion")
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(train_blobs["weights"])
        )
        logger.info("Setting learner class")
        learner_class = Learner_Classification
        logger.info("Creating model")
        crossing_detector_model = learner_class.create_model(network_params)
        logger.info("Initialize networks params with Xavier initialization")
        crossing_detector_model.apply(weights_xavier_init)

        if network_params.use_gpu:
            logger.info("Sending model and criterion to GPU")
            torch.cuda.set_device(0)
            cudnn.benchmark = True  # make it train faster
            crossing_detector_model = crossing_detector_model.cuda()
            criterion = criterion.cuda()

        logger.info("Setting optimizer")
        optimizer = torch.optim.__dict__[network_params.optimizer](
            crossing_detector_model.parameters(), **network_params.optim_args
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
                "=> Load model weights: {}".format(trainer.best_model_path)
            )
            model_state = torch.load(trainer.best_model_path)
            crossing_detector_model.load_state_dict(model_state, strict=True)
            logger.info("=> Load Done")

            logger.info("Classify individuals and crossings")
            crossings_predictor = GetPredictionCrossigns(
                video,
                crossing_detector_model,
                eval_blobs,
                network_params,
            )
            predictions = crossings_predictor.get_all_predictions()

            print(len([p for p in predictions if p == 0]), "individuals")
            print(len([p for p in predictions if p == 1]), "crossings")
            for blob, prediction in zip(eval_blobs, predictions):
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
    else:
        logger.debug(
            "There are not enough crossings to train the crossing detector"
        )
        video._there_are_crossings = False
