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

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from idtrackerai.network.crossings_detector_model.stop_training_criteria_crossings import Stop_Training
from idtrackerai.network.crossings_detector_model.store_accuracy_and_loss_crossings import Store_Accuracy_and_Loss
from idtrackerai.network.crossings_detector_model.epoch_runner_crossings import EpochRunner

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.train_crossing_detector")


class TrainDeepCrossing(object):
    def __init__(self, net, training_dataset, validation_dataset,
                 num_epochs=50, plot_flag=True, return_store_objects=False):
        """Build the dataset and trains the model
        The dataset is built according to
        Taken from tensorflow/contrib/learn/datasets/mnist.
        """
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.num_epochs = num_epochs
        self.plot_flag = plot_flag
        self.net = net
        self.return_store_objects = return_store_objects
        self.train_model()

    def train_model(self, global_step=0,
                    check_for_loss_plateau=None,
                    print_flag=False,
                    store_accuracy_and_error=False):
        logger.info("\nTraining Deep Crossing Detector")
        store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name='training')
        store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name='validation')

        if self.plot_flag:
            # Initialize pre-trainer plot
            plt.ion()
            fig, ax_arr = plt.subplots(3)
            title = 'Crossing-detector'
            fig.canvas.set_window_title(title)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

        self.net.compute_loss_weights(self.training_dataset.labels)
        trainer = EpochRunner(self.training_dataset,
                              starting_epoch=global_step,
                              print_flag=print_flag)
        validator = EpochRunner(self.validation_dataset,
                                starting_epoch=global_step,
                                print_flag=print_flag)
        # set criteria to stop the training
        stop_training = Stop_Training(check_for_loss_plateau=check_for_loss_plateau)
        logger.debug("entering the epochs loop...")
        while not stop_training(store_training_accuracy_and_loss_data,
                                store_validation_accuracy_and_loss_data,
                                trainer._epochs_completed):
            # --- Training
            trainer.run_epoch('Training',
                              store_training_accuracy_and_loss_data,
                              self.net.train)
            # --- Validation
            validator.run_epoch('Validation',
                                store_validation_accuracy_and_loss_data,
                                self.net.validate)
            # Update counter
            trainer._epochs_completed += 1
            validator._epochs_completed += 1

        if (np.isnan(store_training_accuracy_and_loss_data.loss[-1]) or np.isnan(store_validation_accuracy_and_loss_data.loss[-1])):
            logger.warn("The model diverged. Falling back to individual-crossing discrimination by average area model.")
            self.model_diverged = True
        else:
            self.model_diverged = False
            global_step += trainer.epochs_completed
            logger.debug('validation losses: %s' % str(store_validation_accuracy_and_loss_data.loss))
            # plot if asked
            if self.plot_flag:
                store_training_accuracy_and_loss_data.plot(ax_arr, color='r')
                store_validation_accuracy_and_loss_data.plot(ax_arr, color='b')
                fig.savefig(os.path.join(self.net.params.save_folder, title + '.pdf'))
            # store training and validation losses and accuracies
            if store_accuracy_and_error:
                store_training_accuracy_and_loss_data.save()
                store_validation_accuracy_and_loss_data.save()
            # Save network model
            self.net.save(global_step=global_step)
            if self.return_store_objects:
                self.store_training_accuracy_and_loss_data = store_training_accuracy_and_loss_data
                self.store_validation_accuracy_and_loss_data = store_validation_accuracy_and_loss_data
