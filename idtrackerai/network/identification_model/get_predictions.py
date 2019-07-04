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
    logger = logging.getLogger("__main__.get_predictions")

class GetPrediction(object):
    """Manages the inference of the identities of a set of images

    Attributes
    ----------

    data_set : <DataSet object>
        Object containing the images whose labels has to be predicted
        (see :class:`~get_data.DataSet`)
    softmax_probs : ndarray
        Array of shape [number of images, number of animals] with the softmax vectors
        for every image in the `data_set`
    predictions : ndarray
        Array of shape [number of images, 1] with the predictions computed as the argmax
        of the softmax vector of every image (in dense format).
    _fc_vectors : ndarray
        Array of shape [number of images, number of neurons in second to last fully connected layer]
    batch_size : int
        Size of the batch to send the images through the network to get the predictions

    """
    def __init__(self, data_set):
        # Data set
        self.data_set = data_set
        self._softmax_probs = []
        self._predictions = []
        self._predictions_KNN = []
        self._fc_vectors = []
        self.batch_size = conf.BATCH_SIZE_PREDICTIONS_IDCNN

    @property
    def softmax_probs(self):
        return self._softmax_probs

    @property
    def predictions(self):
        return self._predictions

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set.

        Parameters
        ----------
        batch_size : int
            Number of images to get for the next batch

        Returns
        -------
        images : ndarray
            Array of shape [batch_size, height, width, channels] with the images
            of the batch
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.data_set.images[start:end]

    def get_predictions_softmax(self, batch_operation):
        """Runs a `batch_operation` (typically :meth:`~id_CNN.predict`)

        Parameters
        ----------
        batch_operation : func
            Function to be run in the epoch

        """
        self._index_in_epoch = 0
        while self._index_in_epoch < self.data_set._num_images:
            softmax_probs_batch, predictions_batch = batch_operation(self.next_batch(self.batch_size))
            self._softmax_probs.append(softmax_probs_batch)
            self._predictions.append(predictions_batch)
        self._softmax_probs = np.concatenate(self._softmax_probs, axis = 0)
        self._predictions = np.concatenate(self._predictions, axis = 0)
