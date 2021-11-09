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

from idtrackerai.utils.py_utils import delete_attributes_from_object

logger = logging.getLogger("__main__.fragment")


class Fragment(object):
    """Contains information about a collection of blobs that belong to the
    same animal or to the same crossing.

    Parameters
    ----------
    fragment_identifier : int
        It uniquely identifies the fragment.
        It is also used to link blobs to fragments, as blobs have an attribute
        called `blob.Blob.fragment_identifier`.
    start_end : tuple
        Indicates the start and end of the fragment.
        The end is exclusive, i.e. follows Python standards.
    blob_hierarchy_in_first_frame : int
        Indicates the hierarchy in the blob in the first frame of the fragment.
        The hierarchy is the order by which the function blob_extractor
        (see segmentation_utils.py) extracts information about the blobs
        of a frame.
        This attribute was used to plot the accumulation steps figures of the
        paper.
    images : list
        List of integers indicating the index of the identification image
        in the episode.
        This corresponds to the `identification_image_index` of the blob.
        Note that the images are stored in the identification_images folder
        inside of the session folder.
        Then the images are loaded using this index and the episode index.
    centroids : list
        List of tuples (x, y) with the centroid of each blob in the fragment.
        The centroids are in pixels and consider the resolution_reduction
        factor.
    episodes : list
        List of integers indicating the episode corresponding to the
        equivalent image index.
    is_an_individual : bool
        Indicates whether the fragment corresponds to a collection of blobs
        that are all labelled as being an individual.
    is_a_crossing : bool
        Indicates whether the fragment corresponds to a collection of blobs
        that are all labelled as being a crossing.
    number_of_animals : int
        Number of animals to be tracked as defined by the user.
    """

    def __init__(
        self,
        fragment_identifier,
        start_end,
        blob_hierarchy_in_first_frame,
        images,
        centroids,
        episodes,
        is_an_individual,
        is_a_crossing,
        number_of_animals,
    ):
        # Attributes set by the input
        self.identifier = fragment_identifier
        self.start_end = start_end
        self.blob_hierarchy_in_first_frame = blob_hierarchy_in_first_frame
        self.images = images
        self.centroids = np.asarray(centroids)
        self.episodes = episodes
        self.is_an_individual = is_an_individual
        self.is_a_crossing = is_a_crossing
        self.number_of_animals = number_of_animals

        # Sets the distance travelled by the blobs in the fragment
        if centroids is not None:
            self.set_distance_travelled()

        # Possible identities to be assigned to the fragment during
        # the identification process
        # TODO: probably unused. Deleted if not necessary
        self.possible_identities = range(1, self.number_of_animals + 1)

        # Attributes set in future steps of the tracking process
        # During fragmentation
        self._is_in_a_global_fragment = False
        # During the cascade of training and identification protocols
        self._used_for_training = False
        self._used_for_pretraining = False
        self._acceptable_for_training = None
        self._temporary_id = None
        self._identity = None
        self._accumulated_globally = False
        self._accumulated_partially = False
        self._accumulation_step = None
        # TODO: there are other attributes that are added later on.
        # "_frequencies",
        # "_P1_vector",
        # "_certainty",
        # "_is_certain",
        # "_P1_below_random",
        # "_non_consistent",
        # during the residual identification these other parameters are also
        # given to the fragment.
        # "_P2_vector", "_ambiguous_identities", "_certainty_P2"
        # However, there are some parts of the algorithm that use hasattr
        # and delattr. So for now we do not initialized them here, but note
        # that this is not best practice.

        # During postprocessing
        self._identity_corrected_solving_jumps = None
        self._identity_is_fixed = False
        # During the manual validation
        # TODO: not sure if this is used. Check if fragments are updated.
        self._user_generated_identities = None

    def reset(self, roll_back_to):
        """Reset attributes of the fragment to a specific part of the
        algorithm.

        Parameters
        ----------
        roll_back_to : str
            Reset all the attributes up to the process specified in input.
            'fragmentation', 'pretraining', 'accumulation', 'assignment'
        """
        #  This method was mainly used to resume the tracking from different
        # rocessing steps. Currently this function is not active, but this
        #  method might still be useful in the future.
        if roll_back_to == "fragmentation" or roll_back_to == "pretraining":
            self._used_for_training = False
            if roll_back_to == "fragmentation":
                self._used_for_pretraining = False
            self._acceptable_for_training = None
            self._temporary_id = None
            self._identity = None
            self._identity_corrected_solving_jumps = None
            self._identity_is_fixed = False
            self._accumulated_globally = False
            self._accumulated_partially = False
            self._accumulation_step = None
            attributes_to_delete = [
                "_frequencies",
                "_P1_vector",
                "_certainty",
                "_is_certain",
                "_P1_below_random",
                "_non_consistent",
            ]
            delete_attributes_from_object(self, attributes_to_delete)
        elif roll_back_to == "accumulation":
            self._identity_is_fixed = False
            attributes_to_delete = []
            if not self.used_for_training:
                self._identity = None
                self._identity_corrected_solving_jumps = None
                attributes_to_delete = ["_frequencies", "_P1_vector"]
            attributes_to_delete.extend(
                ["_P2_vector", "_ambiguous_identities", "_certainty_P2"]
            )
            delete_attributes_from_object(self, attributes_to_delete)
        elif roll_back_to == "assignment":
            self._user_generated_identity = None
            self._identity_corrected_solving_jumps = None

    @property
    def is_in_a_global_fragment(self):
        """Boolean indicating whether the fragment is part of a global
        fragment.
        """
        return self._is_in_a_global_fragment

    @property
    def used_for_training(self):
        """Boolean indicating whether the images in the fragment were used to
        train the identification network during the cascade of training and
        identification protocols. See also the accumulation_manager.py module.
        """
        return self._used_for_training

    @property
    def accumulated_globally(self):
        """Boolean indicating whether the fragment was accumulated in a
        global accumulation step of the cascade of training and identification
        protocols. See also the accumulation_manager.py module."""
        return self._accumulated_globally

    @property
    def accumulated_partially(self):
        """Boolean indicating whether the fragment was accumulated in a
        partial accumulation step of the cascade of training and identification
        protocols. See also the accumulation_manager.py module."""
        return self._accumulated_partially

    @property
    def accumulation_step(self):
        """Integer indicating the accumulation step at which the fragment was
        accumulated. See also the accumulation_manager.py module."""
        return self._accumulation_step

    @property
    def accumulable(self):
        """Boolean indicating whether the fragment can be accumulated, i.e. it
        can potentially be used for training."""
        return self._accumulable

    @property
    def used_for_pretraining(self):
        """Boolean indicating whether the images in the fragment were used to
        pretrain the identification network during the pretraining step of the
        Protocol 3. See also the accumulation_manager.py module."""
        return self._used_for_pretraining

    @property
    def acceptable_for_training(self):
        """Boolean to indicate that the fragment was identified sufficiently
        well and can in principle be used for training. See also the
        accumulation_manager.py module."""
        return self._acceptable_for_training

    @property
    def frequencies(self):
        """Numpy array indicating the number of images assigned with each of
        the possible identities. See also
        :meth:`compute_identification_statistics`."""
        return self._frequencies

    @property
    def P1_vector(self):
        """Numpy array indicating the P1 probablity of each of the possible
        identities. See also :meth:`compute_identification_statistics`"""
        return self._P1_vector

    @property
    def P2_vector(self):
        """Numpy array indicating the P2 probablity of each of the possible
        identities. See also :meth:`compute_P2_vector`"""
        return self._P2_vector

    @property
    def certainty(self):
        """Numpy array indicating the certainty of each of the possible
        identities following the P1 vector.
        See also :meth:`compute_certainty_of_individual_fragment`"""
        return self._certainty

    @property
    def certainty_P2(self):
        """Numpy array indicating the certainty of each of the possible
        identities following the P2.
        See also :meth:`compute_P2_vector`"""
        return self._certainty_P2

    @property
    def is_certain(self):
        """Booleand indicating whether the fragment is certain enough to be
        accumulated. See also the accumulation_manager.py module."""
        return self._is_certain

    @property
    def temporary_id(self):
        """Integer indicating a temporary identity assigned to the fragment
        during the cascade of training and identification protocols."""
        return self._temporary_id

    @property
    def temporary_id_for_pretraining(self):
        """Integer indicating the temporary identity used to traing the
        identification neural network during Protocol 3."""
        return self._temporary_id_for_pretraining

    @property
    def identity(self):
        """Identity assigned to the fragment during the cascade of training
        and identification protocols or during the residual identification
        (see also the assigner.py module)"""
        return self._identity

    @property
    def identity_is_fixed(self):
        """Boolean indicating whether the identity is fixed and cannot be
        modified during the postprocessing. This attribute is given during
        the residual identification (see assigner.py module)"""
        return self._identity_is_fixed

    @property
    def identity_corrected_solving_jumps(self):
        """Identity of the fragment assigned during the correction of imposible
        (unrealistic) velocity jumps in the trajectories. See also the
        correct_impossible_velocity_jumps.py module."""
        return self._identity_corrected_solving_jumps

    @property
    def identities_corrected_closing_gaps(self):
        """Identity of the fragment assigned during the interpolation of the
        gaps produced by the crossing fragments. See also the
        assign_them_all.py module."""
        return self._identities_corrected_closing_gaps

    # TODO: Change name of this property, or delete if not used.
    @property
    def user_generated_identity(self):
        """This property is give during the correction of impossible velocity
        jumps. It has nothing to do with the manual validation."""
        return self._user_generated_identity

    @property
    def final_identities(self):
        """Final identities (list) of the fragment considering all the
        corrections corrections made during posprocessing or manually
        during the validation of the trajectories by the user.

        The fragment can have multiple identities if it is a crossing fragment.
        """
        if (
            hasattr(self, "user_generated_identities")
            and self.user_generated_identities is not None
        ):
            return self.user_generated_identities
        else:
            return self.assigned_identities

    @property
    def assigned_identities(self):
        """Assigned identities (list) by the algorithm considering the
        identification process and the postprocessing steps (correction of
        impossible velocity jumps and interpolation of crossings).

        The fragment can have multiple identities if it is a crossing fragment.
        """
        if (
            hasattr(self, "identiies_corrected_closing_gaps")
            and self.identities_corrected_closing_gaps is not None
        ):
            return self.identiies_corrected_closing_gaps
        elif (
            hasattr(self, "identity_corrected_solving_jumps")
            and self.identity_corrected_solving_jumps is not None
        ):
            return [self.identity_corrected_solving_jumps]
        else:
            return [self.identity]

    @property
    def ambiguous_identities(self):
        """Identities that would be ambiguosly assigned during the residual
        identification process. See also the assigner.py module.
        """
        return self._ambiguous_identities

    # TODO: Check if this property is actually used.
    @property
    def potentially_randomly_assigned(self):
        """Identities that would be assigned at random during the cascade of
        training and identificaion protocols."""
        return self._potentially_randomly_assigned

    @property
    def non_consistent(self):
        """Boolean indicating whetherr the fragment identity is consistent with
        coexisting fragment."""
        return self._non_consistent

    @property
    def number_of_images(self):
        """Number images (or blobs) in the fragment."""
        return len(self.images)

    @property
    def has_enough_accumulated_coexisting_fragments(self):
        """Boolean indicating whether the fragment has enough coexisting and
        already accumulated fragments.

        This property is used during the partial accumulation. See also the
        accumulation_manager.py module.
        """
        return (
            sum(
                [
                    fragment.used_for_training
                    for fragment in self.coexisting_individual_fragments
                ]
            )
            >= self.number_of_coexisting_individual_fragments / 2
        )

    def get_attribute_of_coexisting_fragments(self, attribute):
        """Gets a given attribute for all the fragments coexisting with self.

        Parameters
        ----------
        attribute : str
            attribute to retrieve

        Returns
        -------
        list
            attribute specified in `attribute` for the fragments coexisting
            with self

        """
        return [
            getattr(fragment, attribute)
            for fragment in self.coexisting_individual_fragments
        ]

    def set_distance_travelled(self):
        """Computes the distance travelled by the individual in the fragment.
        It is based on the position of the centroids in consecutive images. See
        :attr:`blob.Blob.centroid`.

        """
        if self.centroids.shape[0] > 1:
            self.distance_travelled = np.sum(
                np.sqrt(np.sum(np.diff(self.centroids, axis=0) ** 2, axis=1))
            )
        else:
            self.distance_travelled = 0.0

    def frame_by_frame_velocity(self):
        """Instant speed (in each frame) of the blob in the fragment.

        Returns
        -------
        ndarray
            Frame by frame speed of the individual in the fragment

        """
        return np.sqrt(np.sum(np.diff(self.centroids, axis=0) ** 2, axis=1))

    def compute_border_velocity(self, other):
        """Velocity necessary to cover the space between two fragments.

        Note that these velocities are divided by the number of frames that
        separate self and other fragment.

        Parameters
        ----------
        other : :class:`Fragment`
            Another fragment

        Returns
        -------
        float
            Returns the speed at which an individual should travel to be
            present in both self and other fragments.

        """
        centroids = np.asarray([self.centroids[0], other.centroids[-1]])
        if not self.start_end[0] > other.start_end[1]:
            centroids = np.asarray([self.centroids[-1], other.centroids[0]])
        return np.sqrt(np.sum(np.diff(centroids, axis=0) ** 2, axis=1))[0]

    def _coexist_with(self, other):
        """Boolean indicating whether the given fragment coexists in time with
        another fragment.

        Parameters
        ----------
        other :  :class:`Fragment`
            A second fragment

        Returns
        -------
        bool
            True if self and other coexist in time in at least one frame.

        """
        (s1, e1), (s2, e2) = self.start_end, other.start_end
        return s1 < e2 and e1 > s2

    def get_coexisting_individual_fragments_indices(self, fragments):
        """Get the list of fragment objects representing and individual (i.e.
        not representing a crossing where two or more animals are touching) and
        coexisting (in frame) with self

        Parameters
        ----------
        fragments : list
            List of all the fragments in the video

        """
        self.coexisting_individual_fragments = [
            fragment
            for fragment in fragments
            if fragment.is_an_individual
            and self._coexist_with(fragment)
            and fragment is not self
        ]
        self.number_of_coexisting_individual_fragments = len(
            self.coexisting_individual_fragments
        )

    def check_consistency_with_coexistent_individual_fragments(
        self, temporary_id
    ):
        """Check that the temporary identity assigned to the fragment is
        consistent with respect to the identities already assigned to the
        fragments coexisting (in frame) with it.

        Parameters
        ----------
        temporary_id : int
            Temporary identity assigned to the fragment.

        Returns
        -------
        bool
            True if the identification of self with `temporary_id` does not
            cause any duplication of identities.

        """
        for coexisting_fragment in self.coexisting_individual_fragments:
            if coexisting_fragment.temporary_id == temporary_id:
                return False
        return True

    def compute_identification_statistics(
        self, predictions, softmax_probs, number_of_animals=None
    ):
        """Computes the statistics necessary for the identification of the
        fragment.

        Parameters
        ----------
        predictions : numpy array
            Array of shape [number_of_images_in_fragment, 1] whose components
            are the argmax(softmax_probs) per image
        softmax_probs : numpy array
            Array of shape [number_of_images_in_fragment, number_of_animals]
            whose rows are the result of applying the softmax function to the
            predictions outputted by the idCNN per image
        number_of_animals : int
            Description of parameter `number_of_animals`.

        See Also
        --------
        :meth:`compute_identification_frequencies_individual_fragment`
        :meth:`compute_P1_from_frequencies`
        :meth:`compute_median_softmax`
        :meth:`compute_certainty_of_individual_fragment`
        """
        assert self.is_an_individual
        number_of_animals = (
            self.number_of_animals
            if number_of_animals is None
            else number_of_animals
        )
        self._frequencies = (
            self.compute_identification_frequencies_individual_fragment(
                predictions, number_of_animals
            )
        )
        self._P1_vector = self.compute_P1_from_frequencies(self.frequencies)
        median_softmax = self.compute_median_softmax(
            softmax_probs, number_of_animals
        )
        self._certainty = self.compute_certainty_of_individual_fragment(
            self._P1_vector, median_softmax
        )

    def set_P1_vector_accumulated(self):
        """If the fragment has been used for training its P1_vector is
        modified to be a vector of zeros with a single component set to 1 in
        the :attr:`temporary_id` position.
        """
        assert self.used_for_training and self.is_an_individual
        self._P1_vector = np.zeros(len(self.P1_vector))
        self._P1_vector[self.temporary_id] = 1.0

    @staticmethod
    def get_possible_identities(P2_vector):
        """Returns the possible identities by the argmax of the P2 vector and
        the value of the maximum.
        """
        maxima_indices = np.where(P2_vector == np.max(P2_vector))[0]
        return maxima_indices + 1, np.max(P2_vector)

    def assign_identity(self):
        """Assigns the identity to the fragment by considering the fragments
        coexisting with it.

        If the certainty of the identification is high enough it sets
        the identity of the fragment as fixed and it won't be modified during
        the postprocessing.
        """
        assert self.is_an_individual
        if self.used_for_training and not self._identity_is_fixed:
            self._identity_is_fixed = True
        elif not self._identity_is_fixed:
            possible_identities, max_P2 = self.get_possible_identities(
                self.P2_vector
            )
            if len(possible_identities) > 1:
                self._identity = 0
                self.zero_identity_assigned_by_P2 = True
                self._ambiguous_identities = possible_identities
            else:
                if max_P2 > conf.FIXED_IDENTITY_THRESHOLD:
                    self._identity_is_fixed = True
                self._identity = possible_identities[0]
                self._P1_vector = np.zeros(len(self.P1_vector))
                self._P1_vector[self.identity - 1] = 1.0
                self.recompute_P2_of_coexisting_fragments()

    def recompute_P2_of_coexisting_fragments(self):
        """Updates the P2 of the fragments coexisting with self
        (see :attr:`coexisting_individual_fragments`) if their identity is not
        fixed (see :attr:`identity_is_fixed`)
        """
        # The P2 of fragments with fixed identity won't be recomputed
        # due to the condition in assign_identity() (second line)
        for fragment in self.coexisting_individual_fragments:
            fragment.compute_P2_vector()

    def compute_P2_vector(self):
        """Computes the P2_vector of the fragment.

        It is based on :attr:`coexisting_individual_fragments`"""
        coexisting_P1_vectors = np.asarray(
            [
                fragment.P1_vector
                for fragment in self.coexisting_individual_fragments
            ]
        )
        numerator = np.asarray(self.P1_vector) * np.prod(
            1.0 - coexisting_P1_vectors, axis=0
        )
        denominator = np.sum(numerator)
        if denominator != 0:
            self._P2_vector = numerator / denominator
            P2_vector_ordered = np.sort(self.P2_vector)
            P2_first_max = P2_vector_ordered[-1]
            P2_second_max = P2_vector_ordered[-2]
            self._certainty_P2 = (
                conf.MAX_FLOAT
                if P2_second_max == 0
                else P2_first_max / P2_second_max
            )
        else:
            self._P2_vector = np.zeros(self.number_of_animals)
            self._certainty_P2 = 0.0

    @staticmethod
    def compute_identification_frequencies_individual_fragment(
        predictions, number_of_animals
    ):
        """Counts the argmax of predictions per identity

        Parameters
        ----------
        predictions : numpy array
            Array of shape [number of images in fragment, 1] with the identity
            assigned to each image in the fragment.
            Predictions come from 1 to number of animals to be tracked.
        number_of_animals : int
            number of animals to be tracked

        Returns
        -------
        ndarray
            array of shape [1, number_of_animals], whose i-th component counts
            how many predictions have maximum components at the identity i
        """
        return np.asarray(
            [np.sum(predictions == i) for i in range(1, number_of_animals + 1)]
        )

    @staticmethod
    def compute_P1_from_frequencies(frequencies):
        """Given the frequencies of a individual fragment
        computer the P1 vector.

        P1 is the softmax of the frequencies with base 2 for each identity.
        """
        P1_of_fragment = 1.0 / np.sum(
            2.0
            ** (
                np.tile(frequencies, (len(frequencies), 1)).T
                - np.tile(frequencies, (len(frequencies), 1))
            ),
            axis=0,
        )
        return P1_of_fragment

    @staticmethod
    def compute_median_softmax(softmax_probs, number_of_animals):
        """Given the softmax of the predictions outputted by the identification
        network, it computes their median according to the argmax of the
        softmaxed predictions per image.

        Parameters
        ----------
        softmax_probs : ndarray
            array of shape [number_of_images_in_fragment, number_of_animals]
            whose rows are the result of applying the softmax function to the
            predictions outputted by the idCNN per image
        number_of_animals : int
            number of animals to be tracked as defined by the user

        Returns
        -------
        float
            Median of argmax(softmax_probs) per identity

        """
        softmax_probs = np.asarray(softmax_probs)
        # jumps are fragment composed by a single image, thus:
        if len(softmax_probs.shape) == 1:
            softmax_probs = np.expand_dims(softmax_probs, axis=1)
        max_softmax_probs = np.max(softmax_probs, axis=1)
        argmax_softmax_probs = np.argmax(softmax_probs, axis=1)
        softmax_median = np.zeros(number_of_animals)
        for i in np.unique(argmax_softmax_probs):
            softmax_median[i] = np.median(
                max_softmax_probs[argmax_softmax_probs == i]
            )
        return softmax_median

    @staticmethod
    def compute_certainty_of_individual_fragment(P1_vector, median_softmax):
        """Computes the certainty given the P1_vector of the fragment by
        using the output of :meth:`compute_median_softmax`

        Parameters
        ----------
        P1_vector : numpy array
            Array with shape [1, number_of_animals] computed from frequencies
            by :meth:`compute_identification_statistics`
        median_softmax : ndarray
            Median of argmax(softmax_probs) per image

        Returns
        -------
        float
            Fragment's certainty

        """
        argsort_p1_vector = np.argsort(P1_vector)
        sorted_p1_vector = P1_vector[argsort_p1_vector]
        sorted_softmax_probs = median_softmax[argsort_p1_vector]
        certainty = np.diff(
            np.multiply(sorted_p1_vector, sorted_softmax_probs)[-2:]
        ) / np.sum(sorted_p1_vector[-2:])
        return certainty[0]

    def get_neighbour_fragment(
        self, fragments, scope, number_of_frames_in_direction=0
    ):
        """If it exist, gets the fragment in the list of all fragment whose
        identity is the identity assigned to self and whose starting frame is
        the ending frame of self + 1, or ending frame is the starting frame of
        self - 1

        Parameters
        ----------
        fragments : list
            List of all the fragments in the video
        scope : str
            If "to_the_future" looks for the consecutive fragment wrt to self,
            if "to_the_past" looks for the fragment the precedes self
        number_of_frames_in_direction : int
            Distance (in frame) at which the previous or next fragment has to
            be

        Returns
        -------
        :class:`fragment.Fragment`
            The neighbouring fragment with respect to self in the direction
            specified by scope if it exists. Otherwise None

        """
        # TODO: Is it correct that number_of_frames_in_direction is always 0?
        if scope == "to_the_past":
            neighbour = [
                fragment
                for fragment in fragments
                if fragment.is_an_individual
                and len(fragment.assigned_identities) == 1
                and fragment.assigned_identities[0]
                == self.assigned_identities[0]
                and self.start_end[0] - fragment.start_end[1]
                == number_of_frames_in_direction
            ]
        elif scope == "to_the_future":
            neighbour = [
                fragment
                for fragment in fragments
                if fragment.is_an_individual
                and len(fragment.assigned_identities) == 1
                and fragment.assigned_identities[0]
                == self.assigned_identities[0]
                and fragment.start_end[0] - self.start_end[1]
                == number_of_frames_in_direction
            ]

        assert len(neighbour) < 2
        return neighbour[0] if len(neighbour) == 1 else None

    def set_partially_or_globally_accumulated(self, accumulation_strategy):
        """Sets :attr:`accumulated_globally` and :attr:`accumulated_partially`
        according to `accumulation_strategy`.

        Parameters
        ----------
        accumulation_strategy : str
            Can be "global" or "partial"

        """
        if accumulation_strategy == "global":
            self._accumulated_globally = True
        elif accumulation_strategy == "partial":
            self._accumulated_partially = True
