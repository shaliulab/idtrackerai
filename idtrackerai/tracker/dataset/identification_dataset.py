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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)


import logging

import numpy as np
from confapp import conf
from torchvision.datasets.folder import VisionDataset

logger = logging.getLogger("__main__.crossings_data_set")


class IdentificationDataset(VisionDataset):
    def __init__(self, data_dict, scope, transform=None):
        super(IdentificationDataset, self).__init__(
            data_dict, transform=transform
        )
        self.scope = scope
        self.images = data_dict["images"]
        if self.scope in ["training", "validation", "test"]:
            self.labels = data_dict["labels"]
        else:
            self.labels = np.zeros((self.images.shape[0]))
        self.get_data()

    def get_data(self):

        if self.images.ndim <= 3:
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)

        if self.scope == "training":
            self.images, self.labels = duplicate_PCA_images(
                self.images, self.labels
            )
            self.images, self.labels = shuffle_images_and_labels(
                self.images, self.labels
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def split_data_train_and_validation(
    images, labels, validation_proportion=conf.VALIDATION_PROPORTION
):
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

    images, labels = shuffle_images_and_labels(images, labels)
    for i in np.unique(labels):
        # Get images of this individual
        this_indiv_images = images[np.where(labels == i)[0]]
        this_indiv_labels = labels[np.where(labels == i)[0]]
        # Compute number of images for training and validation
        num_images = len(this_indiv_labels)
        num_images_validation = np.ceil(
            validation_proportion * num_images
        ).astype(int)
        num_images_training = num_images - num_images_validation
        # Get train, validation and test, images and labels
        train_images.append(this_indiv_images[:num_images_training])
        train_labels.append(this_indiv_labels[:num_images_training])
        validation_images.append(this_indiv_images[num_images_training:])
        validation_labels.append(this_indiv_labels[num_images_training:])

    train_images = np.vstack(train_images)
    train_labels = np.concatenate(train_labels, axis=0)

    validation_images = np.vstack(validation_images)
    validation_labels = np.concatenate(validation_labels, axis=0)

    training_weights = (
        1.0
        - np.unique(train_labels, return_counts=True)[1] / len(train_labels)
    ).astype("float32")

    train_dict = {
        "images": train_images,
        "labels": train_labels,
        "weights": training_weights,
    }
    val_dict = {"images": validation_images, "labels": validation_labels}
    return train_dict, val_dict


def shuffle_images_and_labels(images, labels):
    """Shuffles images and labels with a random
    permutation, according to the number of examples"""
    np.random.seed(0)
    perm = np.random.permutation(len(labels)).astype(int)
    images = images[perm]
    labels = labels[perm]
    return images, labels


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
    training_images = np.concatenate(
        [training_images, augmented_images], axis=0
    )
    training_labels = np.concatenate(
        [training_labels, training_labels], axis=0
    )
    return training_images, training_labels
