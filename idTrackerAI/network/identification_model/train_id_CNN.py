from __future__ import absolute_import, division, print_function
import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../')
from constants import BATCH_SIZE_IDCNN
if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.train_id_CNN")

class TrainIdCNN(object):
    def __init__(self, network, data_set,
                num_epochs = 10000,
                starting_epoch = 0,
                check_for_loss_plateau = True,
                print_flag = True):
        """Runs training a model given the network and the data set
        """
        # Counters
        self._epochs_completed = 0
        self.starting_epoch= starting_epoch
        self._index_in_epoch_train = 0
        self._index_in_epoch_val = 0
        self.overfitting_counter = 0
        # Number of epochs
        self.num_epochs = num_epochs
        self._epochs_before_checking_stopping_conditions = 10
        self.check_for_loss_plateau = check_for_loss_plateau
        self.print_flag = print_flag
        # Data set
        self.data_set = data_set
        # Network
        self.net = network

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return (self.data_set.images[start:end], self.data_set.labels[start:end])

    def run_epoch(self, name, store_loss_and_accuracy, batch_operation):
        loss_epoch = []
        accuracy_epoch = []
        individual_accuracy_epoch = []
        self._index_in_epoch = 0
        while self._index_in_epoch < self.data_set._num_images:
            loss_acc_batch, feed_dict = batch_operation(self.next_batch(BATCH_SIZE))
            loss_epoch.append(loss_acc_batch[0])
            accuracy_epoch.append(loss_acc_batch[1])
            individual_accuracy_epoch.append(loss_acc_batch[2])
        loss_epoch = np.mean(np.vstack(loss_epoch))
        accuracy_epoch = np.mean(np.vstack(accuracy_epoch))
        individual_accuracy_epoch = np.nanmean(np.vstack(individual_accuracy_epoch),axis=0)
        if self.print_flag:
            logger.info(name)
            logger.info('(epoch %i)' %(self.starting_epoch + self._epochs_completed)
            logger.info('loss: %s' %str(loss_epoch))
            logger.info('accuracy: %s' %str(straccuracy_epoch)
            logger.info('individual accuray: %s' %str(individual_accuracy_epoch))

        self._index_in_epoch_train = 0
        store_loss_and_accuracy.append_data(loss_epoch, accuracy_epoch, individual_accuracy_epoch)
        return feed_dict
