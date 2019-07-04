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
    logger = logging.getLogger("__main__.get_data")

np.random.seed(0)


class DataSet(object):
    """Contains the `images` and `labels` to be used for training a particular
    model

    Attributes
    ----------

    images : ndarray
        Array of shape [num_images, height, width, channels] containing
        the images of the dataset
    num_images : int
        Number of images in the dataset
    labels : ndarray
        Array of shape [num_images, number_of_classes] containing the
        labels corresponding to the images in the dataset
    number_of_animals : int
        Number of classes in the dataset

    """
    def __init__(self, number_of_animals = None, images = None, labels = None):
        self.images = images
        self._num_images = len(self.images)
        self.labels = labels
        self.number_of_animals = number_of_animals
        #check the number of images and labels are the same. If it true set the num_images
        self.consistency_check()

    def consistency_check(self):
        """Checks that the length of :attr:`images` and :attr:`labels` is the same
        """
        if self.labels is not None:
            assert len(self.images) == len(self.labels)

    def convert_labels_to_one_hot(self):
        """Converts labels from dense format to one hot format
        See Also
        --------
        :func:`~get_data.shuffle_images_and_labels`
        """
        self.labels = dense_to_one_hot(self.labels, n_classes=self.number_of_animals)


def duplicate_PCA_images(training_images, training_labels):
    """Creates a copy of every image in `training_images` by rotating 180 degrees

    Parameters
    ----------
    training_images : ndarray
        Array of shape [number of images, height, width, channels] containing
        the images to be rotated
    training_labels : ndarray
        Array of shape [number of images, 1] containing the labels corresponding
        to the `training_images`

    Returns
    -------
    training_images : ndarray
        Array of shape [2*number of images, height, width, channels] containing
        the original images and the images rotated
    training_labels : ndarray
        Array of shape [2*number of images, 1] containing the labels corresponding
        to the original images and the images rotated
    """
    augmented_images = [np.rot90(image, 2) for image in training_images]
    training_images = np.concatenate([training_images, augmented_images], axis = 0)
    training_labels = np.concatenate([training_labels, training_labels], axis = 0)
    return training_images, training_labels

def split_data_train_and_validation(number_of_animals, images, labels, validation_proportion = conf.VALIDATION_PROPORTION):
    """Splits a set of `images` and `labels` into training and validation sets

    Parameters
    ----------
    number_of_animals : int
        Number of classes in the set of images
    images : list
        List of images (arrays of shape [height, width])
    labels : list
        List of integers from 0 to `number_of_animals` - 1
    validation_proportion : float
        The proportion of images that will be used to create the validation set.


    Returns
    -------
    training_dataset : <DataSet object>
        Object containing the images and labels for training
    validation_dataset : <DataSet object>
        Object containing the images and labels for validation

    See Also
    --------
    :class:`get_data.DataSet`
    :func:`get_data.duplicate_PCA_images`
    :func:`get_data.shuffle_images_and_labels`
    """
    # Init variables
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []
    images = np.expand_dims(np.asarray(images), axis = 3)
    labels = np.expand_dims(np.asarray(labels), axis = 1)
    images, labels = shuffle_images_and_labels(images, labels)
    for i in np.unique(labels):
        # Get images of this individual
        this_indiv_images = images[np.where(labels == i)[0]]
        this_indiv_labels = labels[np.where(labels == i)[0]]
        # Compute number of images for training and validation
        num_images = len(this_indiv_labels)
        num_images_validation = np.ceil(validation_proportion*num_images).astype(int)
        num_images_training = num_images - num_images_validation
        # Get train, validation and test, images and labels
        train_images.append(this_indiv_images[:num_images_training])
        train_labels.append(this_indiv_labels[:num_images_training])
        validation_images.append(this_indiv_images[num_images_training:])
        validation_labels.append(this_indiv_labels[num_images_training:])

    train_images = np.vstack(train_images)
    train_labels = np.vstack(train_labels)
    train_images, train_labels = duplicate_PCA_images(train_images, train_labels)
    train_images, train_labels = shuffle_images_and_labels(train_images, train_labels)
    validation_images = np.vstack(validation_images)
    validation_labels = np.vstack(validation_labels)
    return DataSet(number_of_animals, train_images, train_labels), DataSet(number_of_animals, validation_images, validation_labels)

def shuffle_images_and_labels(images, labels):
    """Shuffles images and labels with a random
    permutation, according to the number of examples"""
    np.random.seed(0)
    perm = np.random.permutation(len(labels))
    images = images[perm]
    labels = labels[perm]
    return images, labels

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
