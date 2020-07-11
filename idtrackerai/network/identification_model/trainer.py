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

# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# from confapp import conf
#
# from idtrackerai.network.identification_model.get_data import split_data_train_and_validation
# from idtrackerai.network.identification_model.epoch_runner import EpochRunner
# from idtrackerai.network.identification_model.stop_training_criteria import Stop_Training
# from idtrackerai.network.identification_model.store_accuracy_and_loss import Store_Accuracy_and_Loss


import logging

logger = logging.getLogger("__main__.trainer")


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

import matplotlib.pyplot as plt
import numpy as np

from idtrackerai.network.train import train
from idtrackerai.network.evaluate import evaluate

from idtrackerai.network.utils.metric import Metric


import logging

logger = logging.getLogger("__main__.train_identification_model")


class TrainIdentification(object):
    def __init__(
        self,
        learner,
        train_loader,
        val_loader,
        network_params,
        stop_training,
        accumulation_manager=None,
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learner = learner
        self.network_params = network_params
        self.stop_training = stop_training
        self.accumulation_manager = accumulation_manager
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
            title = "Identification_network"
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
                self.network_params.save_model_path, val_acc
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
            # update used_for_training flag to True for fragments used
            logger.info("Step completed.")
            if self.accumulation_manager is not None:
                logger.info("Updating global fragments used for training")
                self.accumulation_manager.update_fragments_used_for_training()


# def train(video,
#           fragments,
#           net,
#           images,
#           labels,
#           store_accuracy_and_error,
#           check_for_loss_plateau,
#           save_summaries,
#           print_flag,
#           plot_flag,
#           global_step=0,
#           accumulation_manager=None,
#           batch_size=conf.BATCH_SIZE_IDCNN):
#     """Short summary.
#
#     Parameters
#     ----------
#     video : <Video object>
#         an instance of the class :class:`~video.Video`
#     fragments : list
#         list of instances of the class :class:`~fragment.Fragment`
#     net : <ConvNetwork object>
#         an instance of the class :class:`~id_CNN.ConvNetwork`
#     images : ndarray
#         array of shape [number_of_images, height, width]
#     labels : type
#         array of shape [number_of_images, number_of_animals]
#     store_accuracy_and_error : bool
#         if True the values of the loss function, accuracy and individual
#         accuracy will be stored
#     check_for_loss_plateau : bool
#         if True the stopping criteria (see :mod:`~stop_training_criteria`) will
#         automatically stop the training in case the loss functin computed for
#         the validation set of images reaches a plateau
#     sasave_summaries : bool
#         if True tensorflow summaries will be generated and stored to allow
#         tensorboard visualisation of both loss and activity histograms
#     print_flag : bool
#         if True additional information are printed in the terminal
#     plot_flag : bool
#         if True training and validation loss, accuracy and individual accuracy
#         are plot in a graph at the end of the training session
#     global_epoch : int
#         global counter of the training epoch in pretraining
#     accumulation_manager : <AccumulationManager object>
#         an instance of the class
#         :class:`~accumulation_manager.AccumulationManager`
#     batch_size : int
#         size of the batch of images used for training
#
#     Returns
#     -------
#     int
#         global epoch counter updated after the training session
#     <ConvNetwork object>
#         network with updated parameters after training
#     float
#         ration of images used for pretraining over the total number of
#         available images
#     <Store_Accuracy_and_Loss object>
#         updated with the values collected on the training set of labelled
#         images
#     <Store_Accuracy_and_Loss object>
#         updated with the values collected on the validation set of labelled
#         images
#     """
#     # Save accuracy and error during training and validation
#     # The loss and accuracy of the validation are saved to allow the automatic stopping of the training
#     logger.info("Training...")
#     store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'training', scope = 'training')
#     store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'validation', scope = 'training')
#     if plot_flag:
#         plt.ion()
#         fig, ax_arr = plt.subplots(4)
#         title = 'Accumulation-' + str(video.accumulation_trial) + '-' + str(video.accumulation_step)
#         fig.canvas.set_window_title(title)
#         fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
#     # Instantiate data set
#     training_dataset, validation_dataset = split_data_train_and_validation(net.params.number_of_animals, images, labels)
#     # Convert labels to one hot vectors
#     training_dataset.convert_labels_to_one_hot()
#     validation_dataset.convert_labels_to_one_hot()
#     # Reinitialize softmax and fully connected
#     if video is None or video.accumulation_step == 0:
#         net.reinitialize_softmax_and_fully_connected()
#     # Train network
#     # compute weights to be fed to the loss function (weighted cross entropy)
#     net.compute_loss_weights(training_dataset.labels)
#     trainer = EpochRunner(training_dataset,
#                         starting_epoch = global_step,
#                         print_flag = print_flag,
#                         batch_size = batch_size)
#     validator = EpochRunner(validation_dataset,
#                         starting_epoch = global_step,
#                         print_flag = print_flag,
#                         batch_size = batch_size)
#     #set criteria to stop the training
#     stop_training = Stop_Training(net.params.number_of_animals,
#                                 check_for_loss_plateau = check_for_loss_plateau,
#                                 first_accumulation_flag = video is None or video.accumulation_step == 0)
#
#     global_step0 = global_step
#
#     while not stop_training(store_training_accuracy_and_loss_data,
#                             store_validation_accuracy_and_loss_data,
#                             trainer._epochs_completed):
#         # --- Training
#         feed_dict_train = trainer.run_epoch('Training', store_training_accuracy_and_loss_data, net.train)
#         # --- Validation
#         feed_dict_val = validator.run_epoch('Validation', store_validation_accuracy_and_loss_data, net.validate)
#         # update global step
#         net.session.run(net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed))
#         # write summaries if asked
#         if save_summaries:
#             net.write_summaries(trainer.starting_epoch + trainer._epochs_completed, feed_dict_train, feed_dict_val)
#         # Update counter
#         trainer._epochs_completed += 1
#         validator._epochs_completed += 1
#
#     if (np.isnan(store_training_accuracy_and_loss_data.loss[-1]) or np.isnan(store_validation_accuracy_and_loss_data.loss[-1])):
#         raise ValueError("The model diverged")
#     else:
#         global_step += trainer.epochs_completed
#         #logger.debug('loss values in validation: %s' %str(store_validation_accuracy_and_loss_data.loss[global_step0:]))
#         # update used_for_training flag to True for fragments used
#         logger.info("Accumulation step completed. Updating global fragments used for training")
#         if accumulation_manager is not None:
#             accumulation_manager.update_fragments_used_for_training()
#         # plot if asked
#         if plot_flag:
#             store_training_accuracy_and_loss_data.plot_global_fragments(ax_arr, video, fragments, black = False)
#             store_training_accuracy_and_loss_data.plot(ax_arr, color='r')
#             store_validation_accuracy_and_loss_data.plot(ax_arr, color='b')
#             fig.savefig(os.path.join(net.params.save_folder, title + '.pdf'))
#         # store training and validation losses and accuracies
#         if store_accuracy_and_error:
#             store_training_accuracy_and_loss_data.save(trainer._epochs_completed)
#             store_validation_accuracy_and_loss_data.save(trainer._epochs_completed)
#         net.save()
#         return global_step, net, store_validation_accuracy_and_loss_data, store_training_accuracy_and_loss_data
