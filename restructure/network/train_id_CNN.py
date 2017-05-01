from __future__ import absolute_import, division, print_function

import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

BATCH_SIZE = 50

class TrainIdCNN(object):
    def __init__(self, network, data_set,
                num_epochs = 10000, starting_epoch = 0,
                save_summaries = True,
                check_for_loss_plateau = True,
                store_accuracy_and_error = False,
                print_flag = True):
        """Runs training a model given the network and the data set
        """
        # Counters
        self._epoches_completed = 0
        self.starting_epoch= starting_epoch
        self._index_in_epoch_train = 0
        self._index_in_epoch_val = 0
        self.overfitting_counter = 0
        # Number of epochs
        self.num_epochs = num_epochs
        self._epochs_before_checking_stopping_conditions = 10
        self.check_for_loss_plateau = check_for_loss_plateau
        self.save_summaries = save_summaries
        self.store_accuracy_and_error = store_accuracy_and_error
        self.print_flag = print_flag
        # Data set
        self.data_set = data_set
        # Network
        self.net = network
        # Save accuracy and error during training and validation
        # The loss and accuracy of the validation are saved to allow the automatic stopping of the training
        self.training_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name = 'training')
        self.validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name = 'validation')

    def next_batch_train(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch_train
        self._index_in_epoch_train += batch_size
        end = self._index_in_epoch_train
        return (self.data_set._train_images[start:end], self.data_set._train_labels[start:end])

    def next_batch_validation(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch_val
        self._index_in_epoch_val += batch_size
        end = self._index_in_epoch_val
        return (self.data_set._validation_images[start:end], self.data_set._validation_labels[start:end])

    def train_epoch(self):
        train_loss_epoch = []
        train_accuracy_epoch = []
        train_individual_accuracy_epoch = []
        while self._index_in_epoch_train < self.data_set._num_train_images:
            loss_acc_batch, feed_dict = self.net.train(self.next_batch_train(BATCH_SIZE))
            train_loss_epoch.append(loss_acc_batch[0])
            train_accuracy_epoch.append(loss_acc_batch[1])
            train_individual_accuracy_epoch.append(loss_acc_batch[2])
        train_loss_epoch = np.mean(np.vstack(train_loss_epoch))
        train_accuracy_epoch = np.mean(np.vstack(train_accuracy_epoch))
        train_individual_accuracy_epoch = np.nanmean(np.vstack(train_individual_accuracy_epoch),axis=0)
        if self.print_flag:
            print('\nTraining (epoch %i)' %(self.starting_epoch + self._epoches_completed), ': loss, ', train_loss_epoch, ', accuracy, ', train_accuracy_epoch, ', individual accuray, ')
            print(train_individual_accuracy_epoch)
        self._index_in_epoch_train = 0
        self.training_accuracy_and_loss_data.append_data(train_loss_epoch, train_accuracy_epoch, train_individual_accuracy_epoch)
        return feed_dict

    def validation_epoch(self):
        val_loss_epoch = []
        val_accuracy_epoch = []
        val_individual_accuracy_epoch = []
        while self._index_in_epoch_val < self.data_set._num_validation_images:
            loss_acc_batch, feed_dict = self.net.validate(self.next_batch_validation(BATCH_SIZE))
            val_loss_epoch.append(loss_acc_batch[0])
            val_accuracy_epoch.append(loss_acc_batch[1])
            val_individual_accuracy_epoch.append(loss_acc_batch[2])
        val_loss_epoch = np.nanmean(np.vstack(val_loss_epoch))
        val_accuracy_epoch = np.nanmean(np.vstack(val_accuracy_epoch))
        val_individual_accuracy_epoch = np.nanmean(np.vstack(val_individual_accuracy_epoch),axis=0)
        if self.print_flag:
            print('Validation (epoch %i)' %(self.starting_epoch + self._epoches_completed), ': loss, ', val_loss_epoch, ', accuracy, ', val_accuracy_epoch, ', individual accuray, ')
            print(val_individual_accuracy_epoch)
        self._index_in_epoch_val = 0
        self.validation_accuracy_and_loss_data.append_data(val_loss_epoch, val_accuracy_epoch, val_individual_accuracy_epoch)
        return feed_dict

    def train_model(self):
        #compute weights to be fed to the loss function (weighted cross entropy)
        self.net.compute_loss_weights(self.data_set._train_labels)
        self.stop_training = Stop_Training(self.num_epochs,
                                self.net.params.number_of_animals,
                                check_for_loss_plateau = self.check_for_loss_plateau)

        while not self.stop_training(self.training_accuracy_and_loss_data,
                                    self.validation_accuracy_and_loss_data,
                                    self._epoches_completed):
            # --- Training
            feed_dict_train = self.train_epoch()
            ### NOTE here we can shuffle the training data if we think it is necessary.
            # --- Validation
            feed_dict_val = self.validation_epoch()
            # update global step
            self.net.session.run(self.net.global_step.assign(self.starting_epoch + self._epoches_completed)) # set and update(eval) global_step with index, i
            # take times (for library)

            # plot if asked

            # write summaries if asked
            if self.save_summaries:
                self.net.write_summaries(self.starting_epoch + self._epoches_completed,feed_dict_train, feed_dict_val)
            # Update counter
            self._epoches_completed += 1

        if self.store_accuracy_and_error:
            self.training_accuracy_and_loss_data.save()
            self.validation_accuracy_and_loss_data.save()
