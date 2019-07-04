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

from confapp import conf

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.epoch_runner_crossings")

class EpochRunner(object):
    def __init__(self, data_set,
                starting_epoch = 0,
                print_flag = False):
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
            loss_acc_batch, feed_dict = batch_operation(self.next_batch(conf.BATCH_SIZE_DCD))
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
