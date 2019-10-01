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


import numpy as np

from idtrackerai.list_of_fragments import load_identification_images
from idtrackerai.network.identification_model.get_data import duplicate_PCA_images

import logging
logger = logging.getLogger("__main__.get_crossing_data_set")

class CrossingDataset(object):
    def __init__(self, blobs_list, video, scope):
        self.identification_images_file_path = video.identification_images_file_path
        self.blobs = blobs_list
        self.scope = scope
        self.images = None
        self.labels = None


    def get_data(self):

        if isinstance(self.blobs, dict):
            logger.info("Generating crossing {} set.".format(self.scope))
            crossings_images =  self.get_images_indices(image_type='crossings')
            crossing_labels = np.ones(len(crossings_images))

            logger.info("Generating single individual {} set".format(self.scope))
            individual_images = self.get_images_indices(image_type='individuals')
            individual_labels = np.zeros(len(individual_images))

            logger.info("Preparing images and labels")
            images_indices = crossings_images + individual_images
            print(images_indices)
            self.images = load_identification_images(self.identification_images_file_path, images_indices)
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)

            self.labels = np.concatenate([crossing_labels, individual_labels], axis = 0)
            self.labels = self.dense_to_one_hot(np.expand_dims(self.labels, axis = 1))

            if self.scope == "training":
                self.images, self.labels = duplicate_PCA_images(self.images, self.labels)

            np.random.seed(0)
            permutation = np.random.permutation(len(self.labels)).astype(np.int)
            print(permutation)
            print(self.images.shape)
            self.images = self.images[permutation]
            self.labels = self.labels[permutation]

        elif isinstance(self.blobs, list):
            images_indices = self.get_images_indices()
            self.images = load_identification_images(self.identification_images_file_path, images_indices)
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)

    def get_images_indices(self, image_type=None):
        images = []

        if image_type is not None:
            blobs = self.blobs[image_type]
        else:
            blobs = self.blobs

        for blob in blobs:
            images.append(blob.identification_image_index)

        return images

    @staticmethod
    def dense_to_one_hot(labels, n_classes=2):
        """Convert class labels from scalars to one-hot vectors."""
        labels = np.array(labels)
        n_labels = labels.shape[0]
        index_offset = np.arange(n_labels) * n_classes
        labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.int16)
        indices = (index_offset + labels.ravel()).astype('int')
        labels_one_hot.flat[indices] = 1
        return labels_one_hot