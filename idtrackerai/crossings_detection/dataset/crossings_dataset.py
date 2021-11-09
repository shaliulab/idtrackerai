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

from idtrackerai.list_of_fragments import load_identification_images
from idtrackerai.tracker.dataset.identification_dataset import (
    duplicate_PCA_images,
)

logger = logging.getLogger("__main__.crossings_data_set")


class CrossingDataset(VisionDataset):
    def __init__(self, blobs_list, video, scope, transform=None):
        super(CrossingDataset, self).__init__(blobs_list, transform=transform)
        self.identification_images_file_paths = (
            video.identification_images_file_paths
        )
        self.blobs = blobs_list
        self.scope = scope
        self.images = None
        self.labels = None
        self.get_data()

    def get_data(self):

        if isinstance(self.blobs, dict):
            logger.info("Generating crossing {} set.".format(self.scope))
            crossings_images = self.get_images_indices(image_type="crossings")
            crossing_labels = np.ones(len(crossings_images)).astype(int)

            logger.info(
                "Generating single individual {} set".format(self.scope)
            )
            individual_images = self.get_images_indices(
                image_type="individuals"
            )
            individual_labels = np.zeros(len(individual_images)).astype(int)

            logger.info("Preparing images and labels")
            images_indices = crossings_images + individual_images
            self.images = load_identification_images(
                self.identification_images_file_paths, images_indices
            )
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)

            self.labels = np.concatenate(
                [crossing_labels, individual_labels], axis=0
            )

            if self.scope == "training":
                self.images, self.labels = duplicate_PCA_images(
                    self.images, self.labels
                )

            np.random.seed(0)
            permutation = np.random.permutation(len(self.labels)).astype(
                np.int
            )
            self.images = self.images[permutation]
            self.labels = self.labels[permutation]

        elif isinstance(self.blobs, list):
            images_indices = self.get_images_indices()
            self.images = load_identification_images(
                self.identification_images_file_paths, images_indices
            )
            self.images = np.expand_dims(np.asarray(self.images), axis=-1)
            self.labels = np.zeros((self.images.shape[0]))

    def get_images_indices(self, image_type=None):
        images = []

        if image_type is not None:
            blobs = self.blobs[image_type]
        else:
            blobs = self.blobs

        for blob in blobs:
            images.append((blob.identification_image_index, blob.episode))

        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_train_validation_and_eval_blobs(list_of_blobs, ratio_validation=0.1):
    """Given a list of blobs return 2 dictionaries (training_blobs, validation_blobs), and a list (toassign_blobs).

    :param list_of_blobs:
    :param ratio_validation:
    :return: training_blobs, validation_blobs, toassign_blobs
    """

    training_blobs = {"individuals": [], "crossings": []}
    validation_blobs = {}
    toassign_blobs = []
    for blobs_in_frame in list_of_blobs.blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_sure_individual() or blob.in_a_global_fragment_core(
                blobs_in_frame
            ):
                training_blobs["individuals"].append(blob)
            elif blob.is_a_sure_crossing():
                training_blobs["crossings"].append(blob)
            elif (
                blob.is_an_individual
                and not blob.in_a_global_fragment_core(blobs_in_frame)
                and not blob.is_a_sure_individual()
            ) or (blob.is_a_crossing and not blob.is_a_sure_crossing()):
                toassign_blobs.append(blob)

    n_blobs_crossings = len(training_blobs["crossings"])
    n_blobs_individuals = len(training_blobs["individuals"])
    logger.debug(
        "number of individual blobs (before cut): {}".format(
            n_blobs_individuals
        )
    )
    logger.debug("number of crossing blobs: {}".format(n_blobs_crossings))

    # Shuffle and make crossings and individuals even
    np.random.shuffle(training_blobs["individuals"])
    np.random.shuffle(training_blobs["crossings"])
    if n_blobs_crossings > conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR:
        training_blobs["crossings"] = training_blobs["crossings"][
            : conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR
        ]
        n_blobs_crossings = conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR
    if n_blobs_individuals > conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR:
        training_blobs["individuals"] = training_blobs["individuals"][
            : conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR
        ]
        n_blobs_individuals = conf.MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR
    n_individual_blobs_validation = int(n_blobs_individuals * ratio_validation)
    n_crossing_blobs_validation = int(n_blobs_crossings * ratio_validation)

    # split training and validation
    validation_blobs["individuals"] = training_blobs["individuals"][
        :n_individual_blobs_validation
    ]
    validation_blobs["crossings"] = training_blobs["crossings"][
        :n_crossing_blobs_validation
    ]
    training_blobs["individuals"] = training_blobs["individuals"][
        n_individual_blobs_validation:
    ]
    training_blobs["crossings"] = training_blobs["crossings"][
        n_crossing_blobs_validation:
    ]

    ratio_crossings = n_blobs_crossings / (
        n_blobs_crossings + n_blobs_individuals
    )
    training_blobs["weights"] = [ratio_crossings, 1 - ratio_crossings]

    logger.info(
        "{} individual blobs and {} crossing blobs for training".format(
            len(training_blobs["individuals"]),
            len(training_blobs["crossings"]),
        )
    )
    logger.info(
        "{} individual blobs and {} crossing blobs for validation".format(
            len(validation_blobs["individuals"]),
            len(validation_blobs["crossings"]),
        )
    )
    logger.info("{} blobs to test".format(len(toassign_blobs)))

    return training_blobs, validation_blobs, toassign_blobs
