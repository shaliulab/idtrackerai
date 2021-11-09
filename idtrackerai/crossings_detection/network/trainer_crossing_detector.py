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
import os

import matplotlib.pyplot as plt
import numpy as np

from idtrackerai.network.evaluate import evaluate
from idtrackerai.network.train import train
from idtrackerai.network.utils.metric import Metric

logger = logging.getLogger("__main__.train_crossing_detector")


class TrainDeepCrossing(object):
    def __init__(
        self, learner, train_loader, val_loader, network_params, stop_training
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learner = learner
        self.network_params = network_params
        self.stop_training = stop_training
        self.train_model()

    def train_model(self):

        logger.info("\nTraining Deep Crossing Detector")
        # store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.network_params.save_folder,
        #                                                                 name='training')
        # store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.network_params.save_folder,
        #                                                                   name='validation')

        if self.network_params.plot_flag:
            # Initialize pre-trainer plot
            plt.ion()
            fig, ax_arr = plt.subplots(3)
            title = "Crossing-detector"
            fig.canvas.set_window_title(title)
            fig.subplots_adjust(
                left=None,
                bottom=None,
                right=None,
                top=None,
                wspace=None,
                hspace=0.5,
            )

        # Initialize metric storage
        train_losses = Metric()
        if self.network_params.loss in ["CEMCL", "CEMCL_weighted"]:
            train_losses_CE = Metric()
            train_losses_MCL = Metric()
            val_losses_CE = Metric()
            val_losses_MCL = Metric()
        train_accs = Metric()
        val_losses = Metric()
        val_accs = Metric()

        best_train_acc = -1
        best_val_acc = -1
        logger.debug("entering the epochs loop...")
        while not self.stop_training(train_losses, val_losses, val_accs):
            epoch = self.stop_training.epochs_completed
            losses, train_acc = train(
                epoch, self.train_loader, self.learner, self.network_params
            )

            train_losses.update(losses[0].avg)
            if self.network_params.loss in ["CEMCL", "CEMCL_weighted"]:
                train_losses_CE.update(losses[1].avg)
                train_losses_MCL.update(losses[2].avg)
            train_accs.update(train_acc)

            if self.val_loader is not None and (
                (not self.network_params.skip_eval)
                or (epoch == self.network_params.epochs - 1)
            ):
                losses, val_acc = evaluate(
                    self.val_loader,
                    None,
                    "Validation",
                    self.network_params,
                    self.learner,
                )
                val_losses.update(losses[0].avg)
                if self.network_params.loss in ["CEMCL", "CEMCL_weighted"]:
                    val_losses_CE.update(losses[1].avg)
                    val_losses_MCL.update(losses[2].avg)
                val_accs.update(val_acc)
            # Save checkpoint at each LR steps and the end of optimization

            self.best_model_path = self.learner.snapshot(
                os.path.join(
                    self.network_params.save_folder,
                    "%s_%s_%s"
                    % (
                        self.network_params.dataset,
                        self.network_params.model_name,
                        self.network_params.saveid,
                    ),
                ),
                val_acc,
            )

            if best_val_acc <= val_acc:
                best_train_acc = train_acc
                best_val_acc = val_acc

        if np.isnan(train_losses.values[-1]) or np.isnan(
            val_losses.values[-1]
        ):
            logger.warn(
                "The model diverged. Falling back to individual-crossing discrimination by average area model."
            )
            self.model_diverged = True
        else:
            self.model_diverged = False

            # global_step += trainer.epochs_completed
            # logger.debug('validation losses: %s' % str(store_validation_accuracy_and_loss_data.loss))
            # # plot if asked
            # if self.plot_flag:
            #     store_training_accuracy_and_loss_data.plot(ax_arr, color='r')
            #     store_validation_accuracy_and_loss_data.plot(ax_arr, color='b')
            #     fig.savefig(os.path.join(self.net.params.save_folder, title + '.pdf'))
            # # store training and validation losses and accuracies
            # if store_accuracy_and_error:
            #     store_training_accuracy_and_loss_data.save()
            #     store_validation_accuracy_and_loss_data.save()
            # # Save network model
            # self.net.save(global_step=global_step)
            # if self.return_store_objects:
            #     self.store_training_accuracy_and_loss_data = store_training_accuracy_and_loss_data
            #     self.store_validation_accuracy_and_loss_data = store_validation_accuracy_and_loss_data
