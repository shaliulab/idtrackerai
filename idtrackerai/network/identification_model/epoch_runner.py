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

import sys

import numpy as np

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
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
                print_flag = False,
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
        batch_operation : func
            Function to be run in the epoch

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
        :meth:`next_batch`

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
