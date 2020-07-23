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
import sys

import numpy as np
from confapp import conf

logger = logging.getLogger("__main__.stop_training_criteria")


class Stop_Training(object):
    """Stops the training of the network according to the conditions specified
    in :meth:`__call__`

    Attributes
    ----------
    num_epochs : int
        Number of epochs before starting the training
    number_of_animals : int
        Number of animals in the video
    epochs_before_checking_stopping_conditions : int
        Number of epochs before starting to check the stopping conditions
    overfitting_counter : int
        Counts the number of consecutive overfitting epochs during training
    check_for_loss_plateau : bool
        Flag to check for a plateu in the loss of the validation
    first_accumulation_flag : bool
        Flag to indicate that it is the first step of the accumulation

    """

    conf_variables = [
        "MAXIMUM_NUMBER_OF_EPOCHS_IDCNN",
        "MAX_FLOAT",
        "OVERFITTING_COUNTER_THRESHOLD_IDCNN",
        "OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM",
        "LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN",
        "LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN",
    ]

    def __init__(
        self,
        number_of_animals,
        epochs_before_checking_stopping_conditions=10,
        check_for_loss_plateau=True,
        first_accumulation_flag=False,
        conf_dict=None,
    ):

        self.conf_dict = self.get_conf_dict(conf_dict)
        self.num_epochs = self.conf_dict[
            "MAXIMUM_NUMBER_OF_EPOCHS_IDCNN"
        ]  # maximal num of epochs
        self.number_of_animals = number_of_animals
        self.epochs_before_checking_stopping_conditions = (
            epochs_before_checking_stopping_conditions
        )
        # number of epochs in which the network is overfitting
        # before stopping the training
        self.overfitting_counter = 0
        # bool: if true the training is stopped if the loss
        # is not decreasing enough
        self.check_for_loss_plateau = check_for_loss_plateau
        self.first_accumulation_flag = first_accumulation_flag
        self.epochs_completed = -1

    def __call__(self, loss_training, loss_validation, accuracy_validation):
        """Returns True when one of the conditions to stop the training is
        satisfied, otherwise it returns False

        Parameters
        ----------
        loss_accuracy_training : list
            List with the values of the loss in the training set for the
            previous epochs
        loss_accuracy_validation : list
            List with the values of the loss in the validation set for the
            previous epochs
        epochs_completed : int
            Number of epochs completed before checking the conditions

        """
        # check that the model did not diverged (nan loss).
        self.epochs_completed += 1

        if self.epochs_completed > 0 and (
            np.isnan(loss_training.values[-1])
            or np.isnan(loss_validation.values[-1])
        ):
            logger.error(
                "The model diverged. Oops. Check the hyperparameters and "
                "the architecture of the network."
            )
            return True
        # check if it did not reached the epochs limit
        if self.epochs_completed > self.num_epochs - 1:
            logger.warning(
                "The number of epochs completed is larger than the number "
                "of epochs set for training, we stop the training"
            )
            return True

        # check that the model is not overfitting or if it reached
        # a stable saddle (minimum)
        if (
            self.epochs_completed
            > self.epochs_before_checking_stopping_conditions
        ):
            current_loss = loss_validation.values[-1]
            previous_loss = np.nanmean(
                loss_validation.values[
                    -self.epochs_before_checking_stopping_conditions : -1
                ]
            )

            # The validation loss in the first 10 epochs could have exploded
            # but being decreasing.
            if np.isnan(previous_loss):
                previous_loss = self.conf_dict["MAX_FLOAT"]
            losses_difference = previous_loss - current_loss

            # check overfitting
            if losses_difference < 0.0:
                self.overfitting_counter += 1
                if (
                    self.overfitting_counter
                    >= self.conf_dict["OVERFITTING_COUNTER_THRESHOLD_IDCNN"]
                    and not self.first_accumulation_flag
                ):
                    logger.info("Overfitting\n")
                    return True
                elif (
                    self.first_accumulation_flag
                    and self.overfitting_counter
                    > self.conf_dict[
                        "OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM"
                    ]
                ):
                    # print('Overfitting counter, ', self.overfitting_counter)
                    logger.info("Overfitting first accumulation\n")
                    return True
            else:
                self.overfitting_counter = 0

            # check if the error is not decreasing much
            if self.check_for_loss_plateau:
                if self.first_accumulation_flag and np.abs(
                    losses_difference
                ) < self.conf_dict[
                    "LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN"
                ] * 10 ** (
                    int(np.log10(current_loss)) - 1
                ):
                    logger.info(
                        "The losses difference is very small, "
                        "we stop the training\n"
                    )
                    return True
                elif np.abs(losses_difference) < self.conf_dict[
                    "LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN"
                ] * 10 ** (int(np.log10(current_loss)) - 1):
                    logger.info(
                        "The losses difference is very small, we stop the training\n"
                    )
                    return True

            # if the individual accuracies in validation are 1.
            # for all the animals
            if accuracy_validation.values[-1] == 1.0:
                logger.info(
                    "The individual accuracies in validation is 1. for "
                    "all the individuals, we stop the training\n"
                )
                return True

            # if the validation loss is 0.
            if previous_loss == 0.0 or current_loss == 0.0:
                logger.info("The validation loss is 0., we stop the training")
                return True

        return False

    def get_conf_dict(self, conf_dict):
        if conf_dict is not None:
            # Check that the conf_dict has the variables needed
            for conf_variable in self.conf_variables:
                if conf_variable not in conf_dict.keys():
                    raise Exception(
                        f"The conf_variable {conf_variable} is not in "
                        f"the conf_dict"
                    )
            return conf_dict
        else:
            return {
                conf_var: getattr(conf, conf_var)
                for conf_var in self.conf_variables
            }
