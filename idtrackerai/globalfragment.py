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
import sys

import numpy as np
from confapp import conf

from idtrackerai.list_of_fragments import load_identification_images

logger = logging.getLogger("__main__.globalfragment")


class GlobalFragment(object):
    """Representes a collection of :class:`fragment.Fragment` N different
    animals. Where N is the number of animals in the video as defined by the
    user.

        Parameters
    ----------
    blobs_in_video : list
        List of lists of instances of :class:`blob.Blob`.
    fragments : list
        List of lists of instances of the class :class:`fragment.Fragment`
    first_frame_of_the_core : int
        First frame of the core of the global fragment. See also
        :func:`list_of_global_fragments.detect_global_fragments_core_first_frame`.
        This also acts as a unique identifier of the global fragment.
    number_of_animals : int
        Number of animals to be tracked as defined by the user.
    """

    def __init__(
        self,
        blobs_in_video,
        fragments,
        first_frame_of_the_core,
        number_of_animals,
    ):
        self.first_frame_of_the_core = first_frame_of_the_core
        self.number_of_animals = number_of_animals
        self.individual_fragments_identifiers = [
            blob.fragment_identifier
            for blob in blobs_in_video[first_frame_of_the_core]
        ]

        # Copies some attributes of the fragments as attributes that are lists
        # For example, a list of the distances travelled in each fragment,
        # or a list of the number of images of each fragment
        self._get_list_of_attributes_from_individual_fragments(
            fragments, ["distance_travelled", "number_of_images"]
        )
        self._set_minimum_distance_travelled()

        # TODO: this should be part of the accumulation module
        self._set_candidate_for_accumulation()

        # Initializes some attributes that will be used in other processes
        # during the cascade of training and identification profocols
        self._init_attributes()

        # TODO: add property and a warning if they are not linked.
        self.individual_fragments = None

    # TODO: This should be part of the accumulation module
    @property
    def candidate_for_accumulation(self):
        """Boolean indicating whether the global fragment is a candidate
        for accomulation in the cascade of training and identification
         protocols.
        """
        return self._candidate_for_accumulation

    @property
    def used_for_training(self):
        """Booleand indicating if all the fragments in the global fragment
        have been used for training the identification network"""
        return all(
            [
                fragment.used_for_training
                for fragment in self.individual_fragments
            ]
        )

    @property
    def is_unique(self):
        """Boolean indicating that the global fragment has unique
        identities, i.e. it does not have duplications."""
        self.check_uniqueness(scope="global")
        return self._is_unique

    @property
    def is_partially_unique(self):
        """Boolean indicating that a subset of the fragments in the global
        fragment have unique identities"""
        self.check_uniqueness(scope="partial")
        return self._is_partially_unique

    def _init_attributes(self):
        """Initializes some attributes required for the cascade of
        training and identification protocols"""
        self._ids_assigned = np.nan * np.ones(self.number_of_animals)
        self._temporary_ids = np.arange(self.number_of_animals)
        self._score = None
        self._is_unique = False
        self._is_certain = False
        self._uniqueness_score = None
        self._repeated_ids = []
        self._missing_ids = []
        self.predictions = []
        self.softmax_probs_median = []

    def reset(self, roll_back_to):
        """Resets attributes to the fragmentation step in the algorithm,
        allowing for example to start a new accumulation.

        Parameters
        ----------
        roll_back_to : str
            "fragmentation"
        """
        if roll_back_to == "fragmentation":
            self._init_attributes()

    def set_individual_fragments(self, fragments):
        """Gets the list of instances of the class :class:`fragment.Fragment`
        that constitute the global fragment and sets an attribute with such
        list.

        Parameters
        ----------
        fragments : list
            All the fragments extracted from the video.

        """
        self.individual_fragments = [
            fragment
            for fragment in fragments
            if fragment.identifier in self.individual_fragments_identifiers
        ]

    def _get_list_of_attributes_from_individual_fragments(
        self,
        fragments,
        list_of_attributes=["distance_travelled", "number_of_images"],
    ):
        """Gets the attributes in `list_of_attributes` from the fragments that
        constitute the global fragment and sets new attributes in the class
        containing such values in a list.

        Parameters
        ----------
        fragments : list
            Lits of instances of :class:`fragment.Fragment`.
        list_of_attributes : list
            List of strings indicating the names of the attributes
            to be transferred from the individual fragments to the global
            fragment.
        """
        [
            setattr(self, attribute + "_per_individual_fragment", [])
            for attribute in list_of_attributes
        ]
        for fragment in fragments:
            if fragment.identifier in self.individual_fragments_identifiers:
                assert fragment.is_an_individual
                setattr(fragment, "_is_in_a_global_fragment", True)
                for attribute in list_of_attributes:
                    getattr(
                        self, attribute + "_per_individual_fragment"
                    ).append(getattr(fragment, attribute))

    def _set_minimum_distance_travelled(self):
        """Sets the `minimum_distance_travelled` attribute."""
        self.minimum_distance_travelled = min(
            self.distance_travelled_per_individual_fragment
        )

    def _set_candidate_for_accumulation(self):
        """Sets the attributes `_candidate_for_accumulation` which indicates
        that the global fragment to be eligible for accumulation."""
        self._candidate_for_accumulation = True
        if (
            np.min(self.number_of_images_per_individual_fragment)
            < conf.MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
        ):
            self._candidate_for_accumulation = False

    def acceptable_for_training(self, accumulation_strategy):
        """Returns True if the global fragment is acceptable for training.


        See :attr:`fragment.Fragment.acceptable_for_training` for every
        individual fragment in the global fragment.

        Parameters
        ----------
        accumulation_strategy : str
            Can be either "global" or "partial"

        Returns
        -------
        bool
            True if the global fragment is accceptable for training the
            identification neural network.
        """
        if accumulation_strategy == "global":
            return all(
                [
                    fragment.acceptable_for_training
                    for fragment in self.individual_fragments
                ]
            )
        else:
            return any(
                [
                    fragment.acceptable_for_training
                    for fragment in self.individual_fragments
                ]
            )

    def check_uniqueness(self, scope):
        """Checks that the identities assigned to the individual fragments are
        unique.

        Parameters
        ----------
        scope : str
            Either "global" or "partial".

        """
        all_identities = range(self.number_of_animals)
        if scope == "global":
            if (
                len(
                    set(all_identities)
                    - set(
                        [
                            fragment.temporary_id
                            for fragment in self.individual_fragments
                        ]
                    )
                )
                > 0
            ):
                self._is_unique = False
            else:
                self._is_unique = True
        elif scope == "partial":
            identities_acceptable_for_training = [
                fragment.temporary_id
                for fragment in self.individual_fragments
                if fragment.acceptable_for_training
            ]
            self.duplicated_identities = set(
                [
                    x
                    for x in identities_acceptable_for_training
                    if identities_acceptable_for_training.count(x) > 1
                ]
            )
            if len(self.duplicated_identities) > 0:
                self._is_partially_unique = False
            else:
                self._is_partially_unique = True

    def get_total_number_of_images(self):
        """Gets the total number of images in the global fragment"""
        if not hasattr(self, "total_number_of_images"):
            self.total_number_of_images = sum(
                [
                    fragment.number_of_images
                    for fragment in self.individual_fragments
                ]
            )
        return self.total_number_of_images

    def get_images_and_labels(
        self, identification_images_file_paths, scope="pretraining"
    ):
        """Gets the images and identities in the global fragment as a
        labelled dataset in order to train the identification neural network

        If the scope is "pretraining" the identities of each fragment
        will be arbitrary.
        If the scope is "identity_transfer" then the labels will be
        empty as they will be infered by the identification network selected
        by the user to perform the transferring of identities.

        Parameters
        ----------
        identification_images_file_paths : list
            List of paths (str) where the identification images are stored.
        scope : str, optional
            Whether the images are going to be used for training the
            identification network or for "pretraining", by default
            "pretraining".

        Returns
        -------
        Tuple
            Tuple with two Numpy arrays with the images and their labels.
        """
        images = []
        labels = []

        for temporary_id, fragment in enumerate(self.individual_fragments):
            images.extend(list(zip(fragment.images, fragment.episodes)))
            labels.extend([temporary_id] * fragment.number_of_images)
            if scope == "pretraining":
                fragment._temporary_id_for_pretraining = temporary_id

        return (
            np.asarray(
                load_identification_images(
                    identification_images_file_paths, images
                )
            ),
            np.asarray(labels),
        )

    def update_individual_fragments_attribute(self, attribute, value):
        """Updates a given `attribute` in every individual fragment in the
        global fragment by setting it at `value`

        Parameters
        ----------
        attribute : str
            Attribute to be updated in each fragment of the global fragment.
        value : any
            Value to be set to the attribute of each fragment.

        """
        [
            setattr(fragment, attribute, value)
            for fragment in self.individual_fragments
        ]
