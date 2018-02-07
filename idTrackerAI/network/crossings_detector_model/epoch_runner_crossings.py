from __future__ import absolute_import, division, print_function
import numpy as np
import sys
sys.path.append('../../')
from constants import BATCH_SIZE_DCD

if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.epoch_runner_crossings")

class EpochRunner(object):
    def __init__(self, data_set,
                starting_epoch = 0,
                print_flag = True):
        """Runs training a model given the network and the data set
        """
        # Counters
        self._epochs_completed = 0
        self.starting_epoch= starting_epoch
        # Number of epochs
        self.print_flag = print_flag
        # Data set
        self.data_set = data_set
        self._num_images = len(data_set.images)

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
        while self._index_in_epoch < self._num_images:
            loss_acc_batch, feed_dict = batch_operation(self.next_batch(BATCH_SIZE_DCD))
            loss_epoch.append(loss_acc_batch[0])
            accuracy_epoch.append(loss_acc_batch[1])
            individual_accuracy_epoch.append(loss_acc_batch[2])
        loss_epoch = np.mean(np.vstack(loss_epoch))
        accuracy_epoch = np.mean(np.vstack(accuracy_epoch))
        individual_accuracy_epoch = np.nanmean(np.vstack(individual_accuracy_epoch),axis=0)
        if self.print_flag:
            logger.debug(name)
            logger.debug('epoch: %i' %(self.starting_epoch + self._epochs_completed))
            logger.debug('loss: %s' %str(loss_epoch))
            logger.debug('accuracy: %s' %str(accuracy_epoch))
            logger.debug('individual accuray: %s' %str(individual_accuracy_epoch))
        store_loss_and_accuracy.append_data(loss_epoch, accuracy_epoch, individual_accuracy_epoch)
        return feed_dict
