from __future__ import absolute_import, division, print_function

import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 50
MIN_FLOAT = sys.float_info[3]
MAX_FLOAT = sys.float_info[0]
LEARNING_PERCENTAGE_DIFFERENCE = .01
OVERFITTING_COUNTER_THRESHOLD = 5


class TrainIdCNN(object):
    def __init__(self, network, data_set, num_epochs = 1000, starting_epoch = 0, save_summaries = True, check_learning_percentage = True):
        """Runs training a model given the network and the data set
        """
        # Counters
        self._epoches_completed = 0
        self.starting_epoch= starting_epoch
        self._index_in_epoch_train = 0
        self._index_in_epoch_val = 0
        self.overfitting_counter = 0
        # Number of epochs
        self.num_epochs = 10000
        self._epochs_before_checking_stopping_conditions = 10
        self.check_learning_percentage = check_learning_percentage
        self.save_summaries = save_summaries
        # Data set
        self.data_set = data_set
        # Network
        self.net = network
        # Time series to save
        self.restore()
        # We start the training
        self.train_model()

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

    def stop_training(self):
        if self._epoches_completed > self.num_epochs:
            print('The number of epochs completed is larger than the number of epochs set for training, we stop the training\n')
            return True
        if self._epoches_completed > self._epochs_before_checking_stopping_conditions:
            current_loss = self._validation_loss[-1]
            previous_loss = self._validation_loss[-self._epochs_before_checking_stopping_conditions]
            if np.isnan(previous_loss):
                previous_loss = MAX_FLOAT #The loss in the first 10 epochs could have exploted but being decreasing.
            losses_difference = (previous_loss-current_loss)

            if np.isnan(current_loss): # If the network has exploted we stop
                print('The current loss in nan, we stop the training\n')
                return True

            if losses_difference < 0.: # If the network is overfitting after 5 epochs we stop
                self.overfitting_counter += 1
                if self.overfitting_counter >= OVERFITTING_COUNTER_THRESHOLD:
                    print('The network is overfitting, we stop the training\n')
                    return True
            else:
                self.overfitting_counter = 0

            # if the differences of the losses is very small
            if np.abs(losses_difference) < LEARNING_PERCENTAGE_DIFFERENCE*10**(int(np.log10(current_loss))-1) and self.check_learning_percentage:
                print('Losses differece, ', losses_difference)
                print('Epsilon, ', LEARNING_PERCENTAGE_DIFFERENCE*10**(int(np.log10(current_loss))-1))
                print('The losses difference is very small, we stop the training\n')
                return True

            # if the individual accuracies in validation are 1. for all the animals
            if list(self._validation_individual_accuracy[-1]) == list(np.ones(self.data_set.number_of_animals)):
                print('The individual accuracies in validation is 1. for all the individuals, we stop the training\n')
                return True

            # if the validation loss is 0.
            if previous_loss == 0. or current_loss == 0.:
                print('The validation loss is 0., we stop the training')
                return True

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
        print('\nTraining (epoch %i)' %(self.starting_epoch + self._epoches_completed), ': loss, ', train_loss_epoch, ', accuracy, ', train_accuracy_epoch, ', individual accuray, ')
        print(train_individual_accuracy_epoch)
        self._index_in_epoch_train = 0
        self._train_loss.append(train_loss_epoch)
        self._train_accuracy.append(train_accuracy_epoch)
        self._train_individual_accuracy.append(train_individual_accuracy_epoch)
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
        print('Validation (epoch %i)' %(self.starting_epoch + self._epoches_completed), ': loss, ', val_loss_epoch, ', accuracy, ', val_accuracy_epoch, ', individual accuray, ')
        print(val_individual_accuracy_epoch)
        self._index_in_epoch_val = 0
        self._validation_loss.append(val_loss_epoch)
        self._validation_accuracy.append(val_accuracy_epoch)
        self._validation_individual_accuracy.append(val_individual_accuracy_epoch)
        return feed_dict

    def save(self):
        """method to save the variables of the training (loss, acc, individual acc, times )
        """
        loss_acc_dict = {
                        "train_loss": self._train_loss,
                        "train_accuracy": self._train_accuracy,
                        "train_individual_accuracy": self._train_individual_accuracy ,
                        "validation_loss": self._validation_loss,
                        "validation_accuracy": self._validation_accuracy,
                        "validation_individual_accuracy": self._validation_individual_accuracy,
                        }
        np.save(os.path.join(self.net.network_params.save_folder,'loss_acc_dict.npy'), loss_acc_dict)
        print('loss_acc_dict.npy saved in %s' %self.net.network_params.save_folder)
        pass

    def restore(self):
        """method to restore variables of the training (loss, acc, individua acc, times)
        """
        if self.net.is_restoring:
            loss_acc_dict_path = os.path.join(self.net.network_params.restore_folder,'loss_acc_dict.npy')
        elif self.net.is_knowledge_transfer:
            # during pre-training of the network we want to perform knowledge transfer for each global fragment
            # and we want to load the model and save from the same folder: knowledge_transfer_folder and save_folder are the same.
            loss_acc_dict_path = os.path.join(self.net.network_params.save_folder,'loss_acc_dict.npy')
        if os.path.isfile(loss_acc_dict_path):
            print('Restoring loss_acc_dict from %s' %loss_acc_dict_path)
            loss_acc_dict = np.load(loss_acc_dict_path).item()
            self._train_loss = loss_acc_dict["train_loss"]
            self._train_accuracy = loss_acc_dict["train_accuracy"]
            self._train_individual_accuracy = loss_acc_dict["train_individual_accuracy"]
            self._validation_loss = loss_acc_dict["validation_loss"]
            self._validation_accuracy = loss_acc_dict["validation_accuracy"]
            self._validation_individual_accuracy = loss_acc_dict["validation_individual_accuracy"]
        else:
            self._train_loss = []
            self._train_accuracy = []
            self._train_individual_accuracy = []
            self._validation_loss = []
            self._validation_accuracy = []
            self._validation_individual_accuracy = []
        pass

    def plot_training(self):
        """
        """
        pass

    def train_model(self):
        self.overfitting_counter = 0

        self.net.compute_loss_weights(self.data_set._train_labels)

        while not self.stop_training():

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
