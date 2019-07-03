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

from idtrackerai.utils.py_utils import delete_attributes_from_object

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.fragment")

class Fragment(object):
    """ Collects the Blob objects (:class:`~blob.Blob`) associated to the same individual or crossing.

    Attributes
    ----------

    fragment_identifier : int
        Unique identifier of the fragment assigned with :func:`~list_of_blobs.ListOfBlobs.compute_fragment_identifier_and_blob_index` and :func:`~list_of_blobs.ListOfBlobs.compute_crossing_fragment_identifier`
    start_end : tuple
        (starting_frame,  ending_frame) of the fragment
    blob_hierarchy_in_starting_frame : int
        Hierarchy of the blob in the starting frame of the fragment (the hierarchy if compute bottom-top, left-right).
    images : list
        List of images associated to every blob represented in the fragment and ordered according to the frame they have been segmented from
    bounding_box_in_frame_coordinates : tuple
        List of bounding boxes (see :attr:`~blob.Blob.bounding_box_in_frame_coordinates`)
    centroids : list
        List of centroids (see :attr:`~blob.Blob.centroid`)
    areas : list
        List of areas (see :attr:`~blob.Blob.area`)
    is_an_individual : bool
        True if the image has been classified as representing an individual by :func:`~crossing_detector.detect_crossings`
    is_a_crossing : bool
        True if the image has been classified as representing a crossing by :func:`~crossing_detector.detect_crossings`
    number_of_animals : int
        Number of animal to be tracked
    user_generated_identity : int
        Identity generated by the user during validation (if it exists, else None)
    is_in_a_global_fragment : bool
        True if the fragment is contained in a global fragment (see :class:`~globalfragment.GlobalFragment`)
    used_for_training : bool
        True if the fragment has been used to train the idCNN in the finerprinting protocol used to track the video
    accumulated_globally : bool
        True if the fragment's images have been accumulated with global strategy as references during the fingerprinting protocol used to track the video
    accumulated_partially : bool
        True if the fragment's images have been accumulated with partial strategy as references during the fingerprinting protocol used to track the video
    accumulation_step : int
        Accumulation step in which the fragment's images have been accumulated
    accumulable : bool
        True if the fragment is eligible for accumulation i.e. it is contained in a global fragment object (see :class:`~globalfragment.GlobalFragment`)
    used_for_pretraining : bool
        True if the fragment has been used to pretrain the idCNN in the third protocol (see :func:`~pre_trainer.pre_train`)
    acceptable_for_training : bool
        True if the fragment satisfies the conditions in :meth:`~accumulation_manager.AccumulationManager.check_if_is_acceptable_for_training`
    frequencies : ndarray
        array with shape [1, number_of_animals]. The ith component is the number of times the fragment has been assigned by the idCNN (argmax of the softmax predictions) to the identity i
    P1_vector : ndarray
        array with shape [1, number_of_animals] computed from frequencies by :meth:`compute_identification_statistics`
    P2_vector : ndarray
        array with shape [1, number_of_animals] computed from the P1_vector and considering all the identified fragment overlapping (in frames) with self by :meth:`compute_identification_statistics`
    certainty : float
        certainty of the identification of the fragment computed according to meth:`compute_certainty_of_individual_fragment` (depends only on the predictions of the idCNN for the fragment's images)
    certainty_P2 : float
        certainty of the identification of the fragment computed according to meth:`compute_P2_vector`
    is_certain : bool
        True if :attr:`certainty` is greater than :const:`conf.CERTAINTY_THRESHOLD`
    temporary_id : int
        Identity assigned to the fragment during the fingerprint protocols cascade
    identity : int
        Identity assigned to the fragment
    identity_is_fixed : bool
        True if the :attr:`certainty_P2` is greater than :const:`conf.FIXED_IDENTITY_THRESHOLD`
    identity_corrected_closing_gaps : int
        Identity assigned to the fragment while solving the crossing if it exists else None
    user_generated_identity : int
        Identity assigned to the fragment by the user during validation if it exists else None
    final_identity : int
        Final identity of the fragment. It corresponds to :attr:`user_generated_identity` if it exist otherwise to the identity assigned by the algorithm
    assigned_identity : int
        Identity assigned to the fragment by the algorithm
    ambiguous_identities : list
        List of possible identities in case the assignment is ambiguous (two or more identity can be assigned to the fragment with the same certainty)
    potentially_randomly_assigned : bool
        True if :attr:`certainty` is below random wrt the number of images in the fragment
    non_consistent : bool
        True if exist a fragment that overlaps in frame with self whose identity is the same as self
    number_of_images : int
        number of images composing the fragment
    has_enough_accumulated_coexisting_fragments : bool
        the partial accumulation strategy assumes that the condition of :attr:`has_enough_accumulated_coexisting_fragments` holds
    distance_travelled : float
        distance travelled by the individual in the fragment
    coexisting_individual_fragments : list
        List of fragment objects that coexist (in frames) with self and that represent an individual
    number_of_coexisting_individual_fragments : int
        Number of individual fragments coexisting with self
    """
    def __init__(self, fragment_identifier = None, start_end = None,
                        blob_hierarchy_in_starting_frame = None, images = None,
                        bounding_box_in_frame_coordinates = None,
                        centroids = None, areas = None,
                        is_an_individual = None, is_a_crossing = None,
                        number_of_animals = None,
                        user_generated_identity = None):
        self.identifier = fragment_identifier
        self.start_end = start_end
        self.blob_hierarchy_in_starting_frame = blob_hierarchy_in_starting_frame
        self.images = images
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates
        self.centroids = np.asarray(centroids)
        if centroids is not None:
            self.set_distance_travelled()
        self.areas = np.asarray(areas)
        self.is_an_individual = is_an_individual
        self.is_a_crossing = is_a_crossing
        self.number_of_animals = number_of_animals
        self.possible_identities = range(1, self.number_of_animals + 1)
        self._is_in_a_global_fragment = False
        self._used_for_training = False
        self._used_for_pretraining = False
        self._acceptable_for_training = None
        self._temporary_id = None
        self._identity = None
        self._identity_corrected_solving_jumps = None
        self._user_generated_identity = user_generated_identity
        self._identity_is_fixed = False
        self._accumulated_globally = False
        self._accumulated_partially = False
        self._accumulation_step = None

    def reset(self, roll_back_to = None):
        """Reset attributes of self to a specific part of the algorithm

        Parameters
        ----------
        roll_back_to : str
            Reset all the attributes up to the process specified in input.
            'fragmentation', 'pretraining', 'accumulation', 'assignment'

        """
        if roll_back_to == 'fragmentation' or roll_back_to == 'pretraining':
            self._used_for_training = False
            if roll_back_to == 'fragmentation': self._used_for_pretraining = False
            self._acceptable_for_training = None
            self._temporary_id = None
            self._identity = None
            self._identity_corrected_solving_jumps = None
            self._identity_is_fixed = False
            self._accumulated_globally = False
            self._accumulated_partially = False
            self._accumulation_step = None
            attributes_to_delete = ['_frequencies',
                                    '_P1_vector', '_certainty',
                                    '_is_certain',
                                    '_P1_below_random', '_non_consistent']
            delete_attributes_from_object(self, attributes_to_delete)
        elif roll_back_to == 'accumulation':
            self._identity_is_fixed = False
            attributes_to_delete = []
            if not self.used_for_training:
                self._identity = None
                self._identity_corrected_solving_jumps = None
                attributes_to_delete = ['_frequencies', '_P1_vector']
            attributes_to_delete.extend(['_P2_vector', '_ambiguous_identities',
                                        '_certainty_P2'])
            delete_attributes_from_object(self, attributes_to_delete)
        elif roll_back_to == 'assignment':
            self._user_generated_identity = None
            self._identity_corrected_solving_jumps = None

    @property
    def is_in_a_global_fragment(self):
        return self._is_in_a_global_fragment

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def accumulated_globally(self):
        return self._accumulated_globally

    @property
    def accumulated_partially(self):
        return self._accumulated_partially

    @property
    def accumulation_step(self):
        return self._accumulation_step

    @property
    def accumulable(self):
        return self._accumulable

    @property
    def used_for_pretraining(self):
        return self._used_for_pretraining

    @property
    def acceptable_for_training(self):
        return self._acceptable_for_training

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def P1_vector(self):
        return self._P1_vector

    @property
    def P2_vector(self):
        return self._P2_vector

    @property
    def certainty(self):
        return self._certainty

    @property
    def certainty_P2(self):
        return self._certainty_P2

    @property
    def is_certain(self):
        return self._is_certain

    @property
    def temporary_id(self):
        return self._temporary_id

    @property
    def temporary_id_for_pretraining(self):
        return self._temporary_id_for_pretraining

    @property
    def identity(self):
        return self._identity

    @property
    def identity_is_fixed(self):
        return self._identity_is_fixed

    @property
    def identity_corrected_solving_jumps(self):
        return self._identity_corrected_solving_jumps

    @property
    def identity_corrected_closing_gaps(self):
        return self._identity_corrected_closing_gaps

    @property
    def user_generated_identity(self):
        return self._user_generated_identity

    @property
    def final_identity(self):
        if hasattr(self, 'user_generated_identity') and self.user_generated_identity is not None:
            return self.user_generated_identity
        else:
            return self.assigned_identity

    @property
    def assigned_identity(self):
        if hasattr(self, 'identity_corrected_closing_gaps') and self.identity_corrected_closing_gaps is not None:
            return self.identity_corrected_closing_gaps
        elif hasattr(self, 'identity_corrected_solving_jumps') and self.identity_corrected_solving_jumps is not None:
            return self.identity_corrected_solving_jumps
        else:
            return self.identity

    @property
    def ambiguous_identities(self):
        return self._ambiguous_identities

    @property
    def potentially_randomly_assigned(self):
        return self._potentially_randomly_assigned

    @property
    def non_consistent(self):
        return self._non_consistent

    @property
    def number_of_images(self):
        return len(self.images)

    @property
    def has_enough_accumulated_coexisting_fragments(self):
        return sum([fragment.used_for_training
                    for fragment in self.coexisting_individual_fragments]) >= self.number_of_coexisting_individual_fragments/2

    def get_attribute_of_coexisting_fragments(self, attribute):
        """Retrieve a spevific attribute from the collection of fragments
        coexisting (in frame) with self

        Parameters
        ----------
        attribute : str
            attribute to retrieve

        Returns
        -------
        list
            attribute specified in `attribute` of the fragments coexisting with self

        """
        return [getattr(fragment,attribute) for fragment in self.coexisting_individual_fragments]

    # def get_fixed_identities_of_coexisting_fragments(self):
    #     """Considers the fragments coexisting with self and returns their
    #     identities if they are fixed (see :attr:identity_is_fixed)
    #
    #     Returns
    #     -------
    #     type
    #         Description of returned object.
    #
    #     """
    #     return [fragment.assigned_identity for fragment in self.coexisting_individual_fragments
    #             if fragment.used_for_training
    #             or fragment.user_generated_identity is not None
    #             or (fragment.identity_corrected_solving_jumps is not None
    #             and fragment.identity_corrected_solving_jumps != 0)]
    #
    # def get_missing_identities_in_coexisting_fragments(self, fixed_identities):
    #     """Returns the identities that have not been assigned to the set of fragments coexisting with self
    #
    #     Parameters
    #     ----------
    #     fixed_identities : list
    #         List of fixed identities
    #
    #     Returns
    #     -------
    #     list
    #         List of missing identities in coexisting fragments
    #
    #     """
    #     identities = self.get_attribute_of_coexisting_fragments('assigned_identity')
    #     identities = [identity for identity in identities if identity != 0]
    #     if not self.identity in fixed_identities:
    #         return list((set(self.possible_identities) - set(identities)) | set([self.identity]))
    #     else:
    #         return list(set(self.possible_identities) - set(identities))

    def set_distance_travelled(self):
        """Computes the distance travelled by the individual in the fragment.
        It is based on the position of the centroids in consecutive images. See
        :attr:`~blob.Blob.centroid`

        """
        if self.centroids.shape[0] > 1:
            self.distance_travelled = np.sum(np.sqrt(np.sum(np.diff(self.centroids, axis = 0)**2, axis = 1)))
        else:
            self.distance_travelled = 0.

    def frame_by_frame_velocity(self):
        """Short summary.

        Returns
        -------
        ndarray
            Frame by frame speed of the individual in the fragment

        """
        return np.sqrt(np.sum(np.diff(self.centroids, axis = 0)**2, axis = 1))

    def compute_border_velocity(self, other):
        """Velocity necessary to cover the space between two fragments. Note that
        these velocities are divided by the number of frames that separate self and other
        outside of the function.

        Parameters
        ----------
        other : <Fragment object>
            A second fragment

        Returns
        -------
        float
            Returns the speed at which an individual should travel to be
            present in both self and other Fragment objects

        """
        centroids = np.asarray([self.centroids[0], other.centroids[-1]])
        if not self.start_end[0] > other.start_end[1]:
            centroids = np.asarray([self.centroids[-1],other.centroids[0]])
        return np.sqrt(np.sum(np.diff(centroids, axis = 0)**2, axis = 1))[0]

    def are_overlapping(self, other):
        """Short summary.

        Parameters
        ----------
        other :  <Fragment object>
            A second fragment

        Returns
        -------
        bool
            True if self and other coexist in at least one frame

        """
        (s1,e1), (s2,e2) = self.start_end, other.start_end
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
        self.coexisting_individual_fragments = [fragment for fragment in fragments
                                            if fragment.is_an_individual and self.are_overlapping(fragment)
                                            and fragment is not self]
        self.number_of_coexisting_individual_fragments = len(self.coexisting_individual_fragments)

    def check_consistency_with_coexistent_individual_fragments(self, temporary_id):
        """Check that the temporary identity assigned to the fragment is
        consistent with respect to the identities already assigned to the
        fragments coexisting (in frame) with it

        Parameters
        ----------
        temporary_id : int
            Temporary identity assigned to the fragment

        Returns
        -------
        bool
            True if the identification of self with `temporary_id` does not
            cause any duplication

        """
        for coexisting_fragment in self.coexisting_individual_fragments:
            if coexisting_fragment.temporary_id == temporary_id:
                return False
        return True

    def compute_identification_statistics(self, predictions, softmax_probs, number_of_animals = None):
        """Computes the statistics necessary to the identification of the
        fragment

        Parameters
        ----------
        predictions : ndarray
            array of shape [number_of_images_in_fragment, 1] whose components
            are the argmax(softmax_probs) per image
        softmax_probs : ndarray
            array of shape [number_of_images_in_fragment, number_of_animals]
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
        number_of_animals = self.number_of_animals if number_of_animals is None else number_of_animals
        self._frequencies = self.compute_identification_frequencies_individual_fragment(predictions, number_of_animals)
        self._P1_vector = self.compute_P1_from_frequencies(self.frequencies)
        median_softmax = self.compute_median_softmax(softmax_probs, number_of_animals)
        self._certainty = self.compute_certainty_of_individual_fragment(self._P1_vector,median_softmax)

    def set_P1_vector_accumulated(self):
        """If self has been used for training its P1_vector is modified to be
        a vector of zeros with a single component set to 1 in the
        :attr:`temporary_id` position
        """
        assert self.used_for_training and self.is_an_individual
        self._P1_vector = np.zeros(len(self.P1_vector))
        self._P1_vector[self.temporary_id] = 1.

    @staticmethod
    def get_possible_identities(P2_vector):
        """Check if P2 has two identical maxima. In that case returns the indices.
        Else return false
        """
        maxima_indices = np.where(P2_vector == np.max(P2_vector))[0]
        return maxima_indices + 1, np.max(P2_vector)

    def assign_identity(self):
        """Assigns the identity to self by considering the fragments coexisting
        with it. If the certainty of the identification is high enough it sets
        the identity of self to be fixed
        """
        assert self.is_an_individual
        if self.used_for_training and not self._identity_is_fixed:
            self._identity_is_fixed = True
        elif not self._identity_is_fixed:
            possible_identities, max_P2 = self.get_possible_identities(self.P2_vector)
            if len(possible_identities) > 1:
                self._identity = 0
                self.zero_identity_assigned_by_P2 = True
                self._ambiguous_identities = possible_identities
            else:
                if max_P2 > conf.FIXED_IDENTITY_THRESHOLD:
                    self._identity_is_fixed = True
                self._identity = possible_identities[0]
                self._P1_vector = np.zeros(len(self.P1_vector))
                self._P1_vector[self.identity - 1] = 1.
                self.recompute_P2_of_coexisting_fragments()

    def recompute_P2_of_coexisting_fragments(self):
        """Updates the P2 of the fragments coexisting with self
        (see :attr:`coexisting_individual_fragments`) if their identity is not
        fixed (see :attr:`identity_is_fixed`)
        """
        # The P2 of fragments with fixed identity won't be recomputed
        # due to the condition in assign_identity() (second line)
        [fragment.compute_P2_vector() for fragment in self.coexisting_individual_fragments]

    def compute_P2_vector(self):
        """Computes the P2_vector related to self. It is based on :attr:`coexisting_individual_fragments`
        """
        coexisting_P1_vectors = np.asarray([fragment.P1_vector for fragment in self.coexisting_individual_fragments])
        numerator = np.asarray(self.P1_vector) * np.prod(1. - coexisting_P1_vectors, axis = 0)
        denominator = np.sum(numerator)
        if denominator != 0:
            self._P2_vector = numerator / denominator
            P2_vector_ordered = np.sort(self.P2_vector)
            P2_first_max = P2_vector_ordered[-1]
            P2_second_max = P2_vector_ordered[-2]
            self._certainty_P2 = conf.MAX_FLOAT if P2_second_max == 0 else P2_first_max / P2_second_max
        else:
            self._P2_vector = np.zeros(self.number_of_animals)
            self._certainty_P2 = 0.

    @staticmethod
    def compute_identification_frequencies_individual_fragment(predictions, number_of_animals):
        """Counts the argmax of predictions per row

        Parameters
        ----------
        predictions : ndarray
            array of shape [n, 1]
        number_of_animals : int
            number of animals to be tracked

        Returns
        -------
        ndarray
            array of shape [1, number_of_animals], whose ith component counts how
            many predictions have maximum components at the index i

        """
        return np.asarray([np.sum(predictions == i)
                            for i in range(1, number_of_animals+1)]) # The predictions come from 1 to number_of_animals + 1

    @staticmethod
    def compute_P1_from_frequencies(frequencies):
        """Given the frequencies of a individual fragment
        computer the P1 vector. P1 is the softmax of the frequencies with base 2
        for each identity.
        """
        P1_of_fragment = 1. / np.sum(2.**(np.tile(frequencies, (len(frequencies),1)).T - np.tile(frequencies, (len(frequencies),1))), axis = 0)
        return P1_of_fragment

    @staticmethod
    def compute_median_softmax(softmax_probs, number_of_animals):
        """Given the softmax of the predictions outputted by the network, it
        computes their median according to the argmax of the softmaxed
        predictions per image

        Parameters
        ----------
        softmax_probs : ndarray
            array of shape [number_of_images_in_fragment, number_of_animals]
            whose rows are the result of applying the softmax function to the
            predictions outputted by the idCNN per image
        number_of_animals : int
            number of animals to be tracked

        Returns
        -------
        type
            Median of argmax(softmax_probs) per image

        """
        softmax_probs = np.asarray(softmax_probs)
        #jumps are fragment composed by a single image, thus:
        if len(softmax_probs.shape) == 1:
            softmax_probs = np.expand_dims(softmax_probs, axis = 1)
        max_softmax_probs = np.max(softmax_probs, axis = 1)
        argmax_softmax_probs = np.argmax(softmax_probs, axis = 1)
        softmax_median = np.zeros(number_of_animals)
        for i in np.unique(argmax_softmax_probs):
            softmax_median[i] = np.median(max_softmax_probs[argmax_softmax_probs == i])
        return softmax_median

    @staticmethod
    def compute_certainty_of_individual_fragment(P1_vector, median_softmax):
        """Computes the certainty given the P1_vector of the fragment by
        using the output of :meth:`compute_median_softmax`

        Parameters
        ----------
        P1_vector : ndarray
            array with shape [1, number_of_animals] computed from frequencies by :meth:`compute_identification_statistics`
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
        certainty = np.diff(np.multiply(sorted_p1_vector,sorted_softmax_probs)[-2:])/np.sum(sorted_p1_vector[-2:])
        return certainty[0]

    def get_neighbour_fragment(self, fragments, scope, number_of_frames_in_direction = 0):
        """If it exist, gets the fragment in the list of all fragment whose
        identity is the identity assigned to self and whose starting frame is
        the starting frame of self + 1, or ending frame is the ending frame of
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
        <Fragment object>
            The neighbouring fragment with respect to self in the direction
            specified by scope if it exists. Otherwise None

        """
        if scope == 'to_the_past':
            neighbour = [fragment for fragment in fragments
                            if fragment.assigned_identity == self.assigned_identity
                            and self.start_end[0] - fragment.start_end[1] == number_of_frames_in_direction]
        elif scope == 'to_the_future':
            neighbour = [fragment for fragment in fragments
                            if fragment.assigned_identity == self.assigned_identity
                            and fragment.start_end[0] - self.start_end[1] == number_of_frames_in_direction]

        assert len(neighbour) < 2
        return neighbour[0] if len(neighbour) == 1 else None

    def set_partially_or_globally_accumulated(self, accumulation_strategy):
        """Sets :attr:`accumulated_globally` and :attr:`accumulated_partially`
        according to `accumulation_strategy`

        Parameters
        ----------
        accumulation_strategy : str
            Can be "global" or "partial"

        """
        if accumulation_strategy == 'global':
            self._accumulated_globally = True
        elif accumulation_strategy == 'partial':
            self._accumulated_partially = True
