from __future__ import absolute_import, division, print_function
import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import logging
sys.path.append('../../')

logger = logging.getLogger("__main__.epoch_runner")

class EpochRunner(object):
    def __init__(self, data_set,
                starting_epoch = 0,
                print_flag = True,
                batch_size = None):
        """Runs training a model given the network and the data set
        """
        # Counters
        self._epochs_completed = 0
        self.starting_epoch= starting_epoch
        self.print_flag = print_flag
        # Data set
        self.data_set = data_set
        self.batch_size = batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

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
            loss_acc_batch, feed_dict = batch_operation(self.next_batch(self.batch_size))
            loss_epoch.append(loss_acc_batch[0])
            accuracy_epoch.append(loss_acc_batch[1])
            individual_accuracy_epoch.append(loss_acc_batch[2])
        loss_epoch = np.mean(np.vstack(loss_epoch))
        accuracy_epoch = np.mean(np.vstack(accuracy_epoch))
        individual_accuracy_epoch = np.nanmean(np.vstack(individual_accuracy_epoch),axis=0)
        if self.print_flag:
            # logger.info('%s (epoch %i). Loss: %f, accuracy %f, individual accuracy: %s' %(name, self.starting_epoch + self._epochs_completed, loss_epoch, accuracy_epoch , individual_accuracy_epoch))
            print('%s (epoch %i). Loss: %f, accuracy %f, individual accuracy: %s' %(name, self.starting_epoch + self._epochs_completed, loss_epoch, accuracy_epoch , individual_accuracy_epoch))

        # self._index_in_epoch_train = 0
        store_loss_and_accuracy.append_data(loss_epoch, accuracy_epoch, individual_accuracy_epoch)
        return feed_dict
