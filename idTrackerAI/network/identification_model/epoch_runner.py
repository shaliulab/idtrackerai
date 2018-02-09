from __future__ import absolute_import, division, print_function
import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../')
if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.epoch_runner")

class EpochRunner(object):
    """ Runs an epoch divided in batches for a given operation and a given
    data set (:class:`~get_data.DataSet`).

    Attributes
    ----------

    epochs_completed : int
        Number of epochs completed in the current step of the training
    starting_epoch : int
        Epoch at which the training step started
    print_flag : bool
        If `True` prints the values of the loss, the accucacy, and the individual
        accuracy at the end of the epoch
    data_set : <DataSet object>
        Object containing the images and labels to be passed through the network
        in the current epoch (see :class:`~get_data.DataSet`)
    batch_size : int
        Number of images to be passed through the network in each bach.

    """
    def __init__(self, data_set,
                starting_epoch = 0,
                print_flag = True,
                batch_size = None):
        self._epochs_completed = 0
        self.starting_epoch = starting_epoch
        self.print_flag = print_flag
        self.data_set = data_set
        self.batch_size = batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Returns the images and labels for the next batch to be computed. Images
        and labels are extracted from a :class:`get_data.DataSet` object

        Parameters
        ----------
        batch_size : int
            Number of examples to be passed through the network in this batch

        Returns
        -------
        images : ndarray
            Array of shape [batch_size, height, width, channels] containing
            the images to be used in this batch
        labels : ndarray
            Array of shape [batch_size, number_of_classes] containing the
            labels corresponding to the images to be used in this batch

        See Also
        --------
        :class:`get_data.DataSet`

        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return (self.data_set.images[start:end], self.data_set.labels[start:end])

    def run_epoch(self, name, store_loss_and_accuracy, batch_operation):
        """Performs a given `batch_operation` for an entire epoch and stores the
        values of the loss and the accurcaies in a :class:`~store_accuracy_and_loss.Store_Accuracy_and_Loss`
        object for visualization

        Parameters
        ----------
        name : string
            A string to be printed in the epoch information. Typically 'Training'
            or 'Validation'.
        store_loss_and_accuracy : <Store_Accuracy_and_Loss object>
            Object collecting the values of the loss, accurcay and individual accuracies
            (see :class:`store_accuracy_and_loss.Store_Accuracy_and_Loss`)

        Returns
        -------
        feed_dict : dict
            Dictionary with the parameters and variables needed to run the
            `batch_operation`. It is used to save the Tensorflow summaries
            if needed

        See Also
        --------
        :class:`get_data.DataSet`
        :class:`store_accuracy_and_loss.Store_Accuracy_and_Loss`

        """
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
            logger.info('%s (epoch %i). Loss: %f, accuracy %f, individual accuracy: %s' %(name, self.starting_epoch + self._epochs_completed, loss_epoch, accuracy_epoch , individual_accuracy_epoch))

        # self._index_in_epoch_train = 0
        store_loss_and_accuracy.append_data(loss_epoch, accuracy_epoch, individual_accuracy_epoch)
        return feed_dict
