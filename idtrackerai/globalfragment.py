# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Chalpalimuad Foundation.
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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals with correct identities in large
# animal collectives (submitted)

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from idtrackerai.constants import  MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION
import sys
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.globalfragment")

class GlobalFragment(object):
    """ A global fragment is a collection of instances of the class
    :class:`~fragment.Fragment`. Such fragments are collected from a part of the
    video in which all animals are visible.

    Attributes
    ----------

    index_beginning_of_fragment : int
        minimum frame number in which all the individual fragments (see
        :class:`~fragment.Fragment`) are all coexisting
    individual_fragments_identifiers : list
        list of the fragment identifiers associated to the individual fragments
        composing the global fragment (see :class:`~fragment.Fragment`)
    number_of_animals : int
        number of animals to be tracked
    _is_unique : bool
        True if each of the individual fragments have been assigned to a unique
        identity
    _is_certain : bool
        True if each of the individual fragments have scored a certaninty above
        the threshold :const:`~constants.CERTAINTY_THRESHOLD`
    _ids_assigned : ndarray
        shape [1, number_of_animals] each componenents correspond to the
        identity assigned by the algorithm to each of the individual fragments
    _temporary_ids : ndarray
        shape [1, number_of_animals] temporary ids assigned during the
        fingerprinting protocol to each of the fragments composing the global
        fragment
    _repeated_ids : list
        list of identities repeated during the identification of the fragments
        in the individual fragment
    _missing_ids : list
        list of identities not assigned in the global fragment (since in a
        global fragment all animals are visible, all the identities should be
        assigned)
    predictions : list
        list of :attr:`~fragment.Fragment.predictions` for every individual
        fragment
    softmax_probs_median : list
        list of :attr:`~fragment.Fragment.softmax_probs_median` for every
        individual fragment

    """
    def __init__(self, list_of_blobs, fragments, index_beginning_of_fragment, number_of_animals):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.individual_fragments_identifiers = [blob.fragment_identifier for blob in list_of_blobs[index_beginning_of_fragment]]
        self.get_list_of_attributes_from_individual_fragments(fragments)
        self.set_minimum_distance_travelled()
        self.set_candidate_for_accumulation()
        self.number_of_animals = number_of_animals
        self.reset(roll_back_to = 'fragmentation')
        self._is_unique = False
        self._is_certain = False

    def reset(self, roll_back_to):
        """Resets attributes to the fragmentation step in the algorithm,
        allowing for example to start a new accumulation

        Parameters
        ----------
        roll_back_to : str
            "fragmentation"

        """
        if roll_back_to == 'fragmentation':
            self._ids_assigned = np.nan * np.ones(self.number_of_animals)
            self._temporary_ids = np.arange(self.number_of_animals)
            self._score = None
            self._is_unique = False
            self._uniqueness_score = None
            self._repeated_ids = []
            self._missing_ids = []
            self.predictions = []
            self.softmax_probs_median = []

    def get_individual_fragments_of_global_fragment(self, fragments):
        """Get the individual fragments in the global fragments by using their
        unique identifiers

        Parameters
        ----------
        fragments : list
            all the fragments extracted from the video
            (see :class:`~fragment.Fragment`)

        """
        self.individual_fragments = [fragment for fragment in fragments
                                        if fragment.identifier in self.individual_fragments_identifiers]

    def get_list_of_attributes_from_individual_fragments(self, fragments,
                list_of_attributes = ['distance_travelled', 'number_of_images']):
        """Given a set of attributes available in the instances of the class
        :class:`~fragment.Fragment` it copies them in the global fragment.
        For instance the attribute number_of_images belonging to the individual
        fragments in the global fragment will be set as
        `global_fragment.number_of_images_per_individual_fragment`, where each
        element of the list corresponds to the number_of_images of each
        individual fragment preserving the order with which the global fragment
        has been initialised

        Parameters
        ----------
        fragments : <Fragment object>
            See :class:`~fragment.Fragment`
        list_of_attributes : list
            List of attributes to be transferred from the individual fragments
            to the global fragment

        """
        [setattr(self, attribute + '_per_individual_fragment',[]) for attribute in list_of_attributes]
        for fragment in fragments:
            if fragment.identifier in self.individual_fragments_identifiers:
                assert fragment.is_an_individual
                setattr(fragment, '_is_in_a_global_fragment', True)
                for attribute in list_of_attributes:
                    getattr(self, attribute + '_per_individual_fragment').append(getattr(fragment, attribute))

    def set_minimum_distance_travelled(self):
        """Sets the minum distance travelled attribute
        """
        self.minimum_distance_travelled = min(self.distance_travelled_per_individual_fragment)

    def set_candidate_for_accumulation(self):
        """Sets the global fragment to be eligible for accumulation
        """
        self._candidate_for_accumulation = True
        if np.min(self.number_of_images_per_individual_fragment) < MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION:
            self._candidate_for_accumulation = False

    @property
    def candidate_for_accumulation(self):
        return self._candidate_for_accumulation

    def get_total_number_of_images(self):
        return sum([fragment.number_of_images for fragment in self.individual_fragments])

    # @property
    # def uniqueness_score(self):
    #     return self._uniqueness_score
    #
    # def compute_uniqueness_score(self, P1_individual_fragments):
    #     """ Computes the distance of the assignation probabilities (P2) per
    #     individual fragment in the global fragment to the identity matrix of
    #     dimension number of animals.
    #     uniqueness_score = 0.0 means that every individual is assigned with
    #     certainty 1.0 once and only once in the global fragment
    #     """
    #     if not self._used_for_training and self.is_unique:
    #         identity = np.identity(self.number_of_animals)
    #         P1_mat = np.vstack(P1_individual_fragments) # stack P1 of each individual fragment in the global fragment into a matrix
    #         perm = np.argmax(P1_mat,axis=1) # get permutation that orders the matrix to match the identity matrix
    #         P1_mat = P1_mat[:,perm] # apply permutation
    #         self._uniqueness_score = np.linalg.norm(P1_mat - identity)
    #
    # @property
    # def score(self):
    #     return self._score
    #
    # def compute_score(self, P1_individual_fragments, max_distance_travelled):
    #     if not self._used_for_training and self.is_unique:
    #         self.compute_uniqueness_score(P1_individual_fragments)
    #         self._score = self.uniqueness_score**2 + (max_distance_travelled - self.minimum_distance_travelled)**2

    @property
    def used_for_training(self):
        return all([fragment.used_for_training for fragment in self.individual_fragments])

    def acceptable_for_training(self, accumulation_strategy):
        """Returns True if the global fragment is acceptable for training.
        See :attr:`~fragment.Fragment.acceptable_for_training` for every
        individual fragment in the global fragment

        Parameters
        ----------
        accumulation_strategy : str
            can be either "global" or "partial"

        Returns
        -------
        bool
            True if the global fragment is accceptable for training

        """
        if accumulation_strategy == 'global':
            return all([fragment.acceptable_for_training for fragment in self.individual_fragments])
        else:
            return any([fragment.acceptable_for_training for fragment in self.individual_fragments])

    @property
    def is_unique(self):
        self.check_uniqueness(scope = 'global')
        return self._is_unique

    @property
    def is_partially_unique(self):
        self.check_uniqueness(scope = 'partial')
        return self._is_partially_unique

    def check_uniqueness(self, scope):
        """Checks that the identities assigned to the individual fragments are
        unique

        Parameters
        ----------
        scope : str
            Either "global" or "partial"

        """
        all_identities = range(self.number_of_animals)
        if scope == 'global':
            if len(set(all_identities) - set([fragment.temporary_id for fragment in self.individual_fragments])) > 0:
                self._is_unique = False
            else:
                self._is_unique = True
        elif scope == 'partial':
            identities_acceptable_for_training = [fragment.temporary_id for fragment in self.individual_fragments
                                                    if fragment.acceptable_for_training]
            self.duplicated_identities = set([x for x in identities_acceptable_for_training if identities_acceptable_for_training.count(x) > 1])
            if len(self.duplicated_identities) > 0:
                self._is_partially_unique = False
            else:
                self._is_partially_unique = True

    def get_total_number_of_images(self):
        """Gets the total number of images in the global fragment
        """
        if not hasattr(self,'total_number_of_images'):
            self.total_number_of_images = sum([fragment.number_of_images for fragment in self.individual_fragments])
        return self.total_number_of_images

    def get_images_and_labels_for_pretraining(self):
        """Arrange the images and identities in the global fragment as a
        labelled dataset in order to train the idCNN
        """
        images = []
        labels = []

        for temporary_id, fragment in enumerate(self.individual_fragments):
            images.extend(fragment.images)
            labels.extend([temporary_id] * fragment.number_of_images)
            fragment._temporary_id_for_pretraining = temporary_id

        return images, labels

    def compute_start_end_frame_indices_of_individual_fragments(self, blobs_in_video):
        """

        Parameters
        ----------
        blobs_in_video : list
            list of the blob objects (see :class:`~blob.Blob`) segmented from
            the video

        """
        self.starts_ends_individual_fragments = [blob.compute_fragment_start_end()
            for blob in blobs_in_video[self.index_beginning_of_fragment]]

    def update_individual_fragments_attribute(self, attribute, value):
        """Update `attribute` in every individual fragment in the global
        fragment by setting it at `value`

        Parameters
        ----------
        attribute : str
            attribute to be updated
        value : list, int, float
            value of `attribute`

        """
        [setattr(fragment, attribute, value) for fragment in self.individual_fragments]
