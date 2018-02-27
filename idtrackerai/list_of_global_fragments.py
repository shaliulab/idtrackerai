# This file is part of idtracker.ai, a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Bergomi, M.G., Romero-Ferrero, F., Heras, F.J.H.
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
import random
import numpy as np
from idtrackerai.globalfragment import GlobalFragment
from idtrackerai.assigner import assign, compute_identification_statistics_for_non_accumulated_fragments
from idtrackerai.accumulation_manager import AccumulationManager
from idtrackerai.constants import  CERTAINTY_THRESHOLD
import sys
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.list_of_global_fragments")

class ListOfGlobalFragments(object):
    """ Collects all the instances of the class
    :class:`~glboalfragment.GlobalFragment`
    generated by considering the fragments
    (see :meth:`~list_of_blobs.compute_fragment_identifier_and_blob_index`)

    Attributes
    ----------

    global_fragments : list
        list of instances of the class :class:`~globalfragment.GlobalFragment`
    number_of_global_fragments :  int
        length of `global_fragments`
    first_global_fragment_for_accumulation :  <GlobalFragment object>
        first global fragment used to start the protocols.
        See :meth:`set_first_global_fragment_for_accumulation`
    maximum_number_of_images : int
        maximum number of images contained in the global fragments listed in
        `global_fragments`
    non_accumulable_global_fragments : list
        list of global fragments that cannot be used during the accumulation
        protocols.
        See :attr:`~globalfragment.GlobalFragment.candidate_for_accumulation`

    Notes
    -----
    A minimum number of images per individual fragments is required for a
    global fragment to be acceptable for the accumulation. This number is set
    in
    :const:`~constants.MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION`

    """
    def __init__(self, global_fragments):
        self.global_fragments = global_fragments
        self.number_of_global_fragments = len(self.global_fragments)

    def reset(self, roll_back_to = None):
        """Resets all the global fragment by calling recursively the method
        :meth:`~globalfragment.GlobalFragment.reset`
        """
        [global_fragment.reset(roll_back_to) for global_fragment in self.global_fragments]

    def order_by_distance_travelled(self):
        """Sorts the global fragments by the minimum distance travelled of their
        individual fragments
        """
        self.global_fragments = sorted(self.global_fragments, key = lambda x: x.minimum_distance_travelled, reverse = True)

    @staticmethod
    def give_me_frequencies_first_fragment_accumulated(i, number_of_animals, fragment):
        """The frequencies (see :meth:`~fragments.Fragments.compute_identification_statistics`)
        are generated artificially for the first global fragments.

        Parameters
        ----------
        i : int
            identity associated to the `fragment`
        number_of_animals : int
            number of animals to track
        fragment : <Fragment object>
            an instance of the class :class:`~fragment.Fragment`

        Returns
        -------

        ndarray
            array of zeros with the `i`th component equal to
            :attr:`~fragment.Fragment.number_of_images`
        """
        frequencies = np.zeros(number_of_animals)
        frequencies[i] = fragment.number_of_images
        return frequencies

    @staticmethod
    def abort_knowledge_transfer_on_same_animals(video, net):
        identities = range(video.number_of_animals)
        net.reinitialize_softmax_and_fully_connected()
        logger.info("Identity transfer failed. We proceed by transferring only the convolutional filters.")
        return identities

    def set_first_global_fragment_for_accumulation(self, video, accumulation_trial):
        """Selects the first global fragment to be used for accumulation

        Parameters
        ----------
        video : <Video object>
            instance of the class :class:`~video.Video`
        accumulation_trial : int
            accumulation number (protocol 2 performs a single accumulation
            attempt, and if used, protocol 3 will perform 3 other attempts)

        Returns
        -------
        int
            frame number corresponding to the beginning of the selected global
            fragment

        """
        self.order_by_distance_travelled()

        try:
            self.first_global_fragment_for_accumulation = self.global_fragments[accumulation_trial]
        except:
            return None

        identities = range(video.number_of_animals)

        [(setattr(fragment, '_acceptable_for_training', True),
            setattr(fragment, '_temporary_id', identities[i]),
            setattr(fragment, '_frequencies', self.give_me_frequencies_first_fragment_accumulated(i, video.number_of_animals, fragment)),
            setattr(fragment, '_is_certain', True),
            setattr(fragment, '_certainty', 1.),
            setattr(fragment, '_P1_vector', fragment.compute_P1_from_frequencies(fragment.frequencies)))
            for i, fragment in enumerate(self.first_global_fragment_for_accumulation.individual_fragments)]

        return self.first_global_fragment_for_accumulation.index_beginning_of_fragment

    def order_by_distance_to_the_first_global_fragment_for_accumulation(self, video, accumulation_trial = None):
        """Sorts the global fragments wrt to their distance from the first
        global fragment chose for accumulation

        Parameters
        ----------
        video : <Video object>
            instance of the class :class:`~video.Video`
        accumulation_trial : int
            accumulation number (protocol 2 performs a single accumulation
            attempt, and if used, protocol 3 will perform 3 other attempts)

        """
        self.global_fragments = sorted(self.global_fragments,
                                        key = lambda x: np.abs(x.index_beginning_of_fragment - video.first_frame_first_global_fragment[accumulation_trial]),
                                        reverse = False)

    def compute_maximum_number_of_images(self):
        """Computes the maximum number of images in the global fragments
        """
        self.maximum_number_of_images = max([global_fragment.get_total_number_of_images() for global_fragment in self.global_fragments])

    def filter_candidates_global_fragments_for_accumulation(self):
        """Filters the global fragments by taking into account the minium
        number of images per individual fragments specified in
        :attr:`~globalfragment.GlobalFragment.candidate_for_accumulation`
        """
        self.non_accumulable_global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if not global_fragment.candidate_for_accumulation]
        self.global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if global_fragment.candidate_for_accumulation]
        self.number_of_global_fragments = len(self.global_fragments)

    def get_data_plot(self):
        """Gathers data to plot a global fragments statistics summary

        Returns
        -------
        int
            number of images in the shortest individual fragment
        int
            number of images in the longest individual fragment
        int
            number of images per individual fragment in global fragment
        int
            median number of images in global fragments
        float
            minimum distance travelled

        """
        number_of_images_in_shortest_individual_fragment = []
        number_of_images_in_longest_individual_fragment = []
        number_of_images_per_individual_fragment_in_global_fragment = []
        median_number_of_images = []
        minimum_distance_travelled = []
        for global_fragment in self.global_fragments:
            number_of_images_in_shortest_individual_fragment.append(min(global_fragment.number_of_images_per_individual_fragment))
            number_of_images_in_longest_individual_fragment.append(max(global_fragment.number_of_images_per_individual_fragment))
            number_of_images_per_individual_fragment_in_global_fragment.append(global_fragment.number_of_images_per_individual_fragment)
            median_number_of_images.append(np.median(global_fragment.number_of_images_per_individual_fragment))
            minimum_distance_travelled.append(min(global_fragment.distance_travelled_per_individual_fragment))

        return number_of_images_in_shortest_individual_fragment,\
                number_of_images_in_longest_individual_fragment,\
                number_of_images_per_individual_fragment_in_global_fragment,\
                median_number_of_images,\
                minimum_distance_travelled

    def delete_fragments_from_global_fragments(self):
        """Deletes the individual fragments from each of the global fragments
        """
        [setattr(global_fragment,'individual_fragments',None) for global_fragment in self.global_fragments]

    def relink_fragments_to_global_fragments(self, fragments):
        """Resets the individual fragments to their respective global fragments
        """
        [global_fragment.get_individual_fragments_of_global_fragment(fragments) for global_fragment in self.global_fragments]

    def save(self, global_fragments_path, fragments):
        """Saves an instance of the class in the path `global_fragments_path`.
        Before saving the individual fragments associated to every global
        fragment are removes by using the method
        :meth:`~delete_fragments_from_global_fragments`
        and resets them after saving by calling
        :meth:`~relink_fragments_to_global_fragments`
        """
        logger.info("saving list of global fragments at %s" %global_fragments_path)
        self.delete_fragments_from_global_fragments()
        np.save(global_fragments_path,self)
        # After saving the list of globa fragments the individual fragments are deleted and we need to relink them again
        self.relink_fragments_to_global_fragments(fragments)

    @classmethod
    def load(self, path_to_load, fragments):
        """Loads an instance of the class saved with :meth:`save` and
        associates individual fragments to each global fragment by calling
        :meth:`~relink_fragments_to_global_fragments`
        """
        logger.info("loading list of global fragments from %s" %path_to_load)
        list_of_global_fragments = np.load(path_to_load).item()
        list_of_global_fragments.relink_fragments_to_global_fragments(fragments)
        return list_of_global_fragments

def detect_beginnings(boolean_array):
    """ Detects the frame where the core of a global fragment starts.
    A core of a global fragment is the part of the global fragment where all the
    individuals are visible, i.e. the number of animals in the frame equals the
    number of animals in the video
    :boolean_array: array with True where the number of animals in the frame equals
    the number of animals in the video
    """
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def check_global_fragments(blobs_in_video, num_animals):
    """Returns an array with True if:
    * each blob has a unique blob intersecting in the past and future
    * number of blobs equals num_animals
    """
    def all_blobs_in_a_fragment(blobs_in_frame):
        """Returns all the blobs in `blobs_in_frame` that are associated to an
        individual
        """
        return all([blob.is_an_individual for blob in blobs_in_frame])

    return [all_blobs_in_a_fragment(blobs_in_frame) and len(blobs_in_frame) == num_animals for blobs_in_frame in blobs_in_video]

def create_list_of_global_fragments(blobs_in_video, fragments, num_animals):
    """Creates the list of instances of the class
    :class:`~globalfragment.GlobalFragment` used to create :class:`.ListOfGlobalFragments`

    Parameters
    ----------
    blobs_in_video : list
        list of the blob objects (see class :class:`~blob.Blob`) generated
        from the blobs segmented in the video
    fragments : list
        list of instances of the class :class:`~fragment.Fragment`
    num_animals : int
        number of animals to track

    Returns
    -------
    list
        list of instances of the class :class:`~globalfragment.GlobalFragment`

    """
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video, fragments, i, num_animals) for i in indices_beginning_of_fragment]
