from __future__ import absolute_import, division, print_function

import itertools
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import psutil
import logging
sys.path.append('./network/crossings_detector_model')
from cnn_architectures import cnn_model_crossing_detector
from network_params_crossings import NetworkParams_crossings
from crossings_detector_model import ConvNetwork_crossings
from stop_training_criteria_crossings import Stop_Training
from store_accuracy_and_loss_crossings import Store_Accuracy_and_Loss
from epoch_runner_crossings import EpochRunner

logger = logging.getLogger("__main__.train_crossing_detector")

class TrainDeepCrossing(object):
    def __init__(self, net, training_dataset, validation_dataset, num_epochs = 50, plot_flag = True):
        """Build the dataset and trains the model
        The dataset is built according to
        Taken from tensorflow/contrib/learn/datasets/mnist.
        """
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.num_epochs = num_epochs
        self.plot_flag = plot_flag
        self.net = net
        self.train_model()

    def train_model(self, global_step = 0,
                            check_for_loss_plateau = None,
                            print_flag = True,
                            store_accuracy_and_error = False):
        print("\nTraining...")
        store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name = 'training')
        store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name = 'validation')

        if self.plot_flag:
            # Initialize pre-trainer plot
            plt.ion()
            fig, ax_arr = plt.subplots(3)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

        self.net.compute_loss_weights(self.training_dataset.labels)
        trainer = EpochRunner(self.training_dataset,
                            starting_epoch = global_step,
                            print_flag = print_flag)
        validator = EpochRunner(self.validation_dataset,
                            starting_epoch = global_step,
                            print_flag = print_flag)
        #set criteria to stop the training
        stop_training = Stop_Training(check_for_loss_plateau = check_for_loss_plateau)
        print("entering the epochs loop...")
        while not stop_training(store_training_accuracy_and_loss_data,
                                store_validation_accuracy_and_loss_data,
                                trainer._epochs_completed):
            # --- Training
            feed_dict_train = trainer.run_epoch('Training', store_training_accuracy_and_loss_data, self.net.train)
            # --- Validation
            feed_dict_val = validator.run_epoch('Validation', store_validation_accuracy_and_loss_data, self.net.validate)
            # Update counter
            trainer._epochs_completed += 1
            validator._epochs_completed += 1

        if (np.isnan(store_training_accuracy_and_loss_data.loss[-1]) or np.isnan(store_validation_accuracy_and_loss_data.loss[-1])):
            logger.warn("The model diverged. Falling back to individual-crossing discrimination by average area model.")
            self.model_diverged = True
        else:
            self.model_diverged = False
            global_step += trainer.epochs_completed
            print('\nvalidation losses: ', store_validation_accuracy_and_loss_data.loss)
            # plot if asked
            if self.plot_flag:
                store_training_accuracy_and_loss_data.plot(ax_arr, color = 'r')
                store_validation_accuracy_and_loss_data.plot(ax_arr, color ='b')
            # store training and validation losses and accuracies
            if store_accuracy_and_error:
                store_training_accuracy_and_loss_data.save()
                store_validation_accuracy_and_loss_data.save()
            # Save network model
            self.net.save(global_step = global_step)
            if self.plot_flag:
                fig.savefig(os.path.join(self.net.params.save_folder,'crossing_detector.pdf'))
