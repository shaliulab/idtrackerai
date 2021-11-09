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

from idtrackerai.tracker.accumulation_manager import AccumulationManager
from idtrackerai.tracker.assigner import (
    assign,
    compute_identification_statistics_for_non_accumulated_fragments,
)
from idtrackerai.globalfragment import GlobalFragment
from idtrackerai.network.utils.utils import fc_weights_reinit

logger = logging.getLogger("__main__.list_of_global_fragments")


class ListOfGlobalFragments(object):
    """Contains a list of instances of the class
    :class:`global_fragment.GlobalFragment`.

    It contains methods to retrieve information from these global fragments
    and to update their attributes.
    These methods are manily used during the cascade of training and
    identification protocols.

    Parameters
    ----------
    global_fragments : list
        List of instances of :class:`global_fragment.GlobalFragment`.
    """

    def __init__(self, global_fragments):
        self.global_fragments = global_fragments
        self.number_of_global_fragments = len(self.global_fragments)

        # Attributes sets in other methods
        self.maximum_number_of_images = None
        self.non_accumulable_global_fragments = None

    def reset(self, roll_back_to=None):
        """Resets all the global fragment by calling recursively the method
        :meth:`globalfragment.GlobalFragment.reset`.
        """
        for global_fragment in self.global_fragments:
            global_fragment.reset(roll_back_to)

    def order_by_distance_travelled(self):
        """Sorts the global fragments by the minimum distance travelled.
        See :attr:`global_fragment.GlobalFragment.minimum_distance_travelled`
        """
        self.global_fragments = sorted(
            self.global_fragments,
            key=lambda x: x.minimum_distance_travelled,
            reverse=True,
        )

    def set_first_global_fragment_for_accumulation(
        self,
        video,  # TODO: remove video and pass only the necessary arguments
        accumulation_trial=0,
        identification_model=None,
        network_params=None,
        knowledge_transfer_info_dict=None,
    ):
        """Sets the first global fragment that will be used during the
        accumulation in the cascade of training and identification protocols.

        If the user asked to perform identity transfer, then the identities
        of the first global fragment will be tried to be assigned using the
        neural network provided by the knowledge_transfer_folrder parameter.

        Parameters
        ----------
        video : :class:`video.Video`
            Video object containing information about the video and the
            tracking process.
        accumulation_trial : int, optional
            Accumulation trials during the cascade of training and
            identification protocols, by default 0.
        identification_model : str, optional
            Path to the directory where the identification neural network
            model that should be used for identity transfer is stored,
            by default None.
        network_params : <NetworkParams object>, optional
            Object with the parameters of the network and how to train it,
             by default None.
        knowledge_transfer_info_dict : dic, optional
            Dictionary with information about the knowledge transfer,
            by default None.

        Returns
        -------
        int
            A unique identifier of the global fragment that will be used as the
            first global fragment for training.
        """
        self.order_by_distance_travelled()

        try:
            self.first_global_fragment_for_accumulation = (
                self.global_fragments[accumulation_trial]
            )
        except IndexError:
            return None

        # TODO: Everything that happens after should be outside of this method.
        # TODO: This should be a function in the module identity transfer.
        if (
            not video.user_defined_parameters["identity_transfer"]
            or identification_model is None
        ):
            identities = range(
                video.user_defined_parameters["number_of_animals"]
            )
        else:
            logger.info(
                "Transferring identities from {}".format(
                    video.user_defined_parameters["knowledge_transfer_folder"]
                )
            )
            identities = self.get_transferred_identities(
                video,
                identification_model,
                network_params,
                knowledge_transfer_info_dict,
            )

        for i, fragment in enumerate(
            self.first_global_fragment_for_accumulation.individual_fragments
        ):
            # TODO: This should be a method in the class Fragment.
            # If identity transfer is performed some of this attributes will
            # be overwritten.
            fragment._acceptable_for_training = True
            fragment._temporary_id = identities[i]
            fragment._frequencies = (
                _get_frequencies_first_fragment_accumulated(
                    i,
                    video.user_defined_parameters["number_of_animals"],
                    fragment,
                )
            )
            fragment._is_certain = True
            fragment._certainty = 1.0
            fragment._P1_vector = fragment.compute_P1_from_frequencies(
                fragment.frequencies
            )

        return (
            self.first_global_fragment_for_accumulation.first_frame_of_the_core
        )

    def order_by_distance_to_the_first_global_fragment_for_accumulation(
        self, video, accumulation_trial=None
    ):
        """Sorts the global fragments with respect to their distance from the
        first global fragment chose for accumulation.

        Parameters
        ----------
        video : :class:`video.Video`
            Instance of the class :class:`video.Video`.
        accumulation_trial : int
            accumulation number (protocol 2 performs a single accumulation
            attempt, and if used, protocol 3 will perform 3 other attempts)
        """
        # TODO: remove dependency of video object.
        self.global_fragments = sorted(
            self.global_fragments,
            key=lambda x: np.abs(
                x.first_frame_of_the_core
                - video.first_frame_first_global_fragment[accumulation_trial]
            ),
            reverse=False,
        )

    # TODO: This should be a function in a separate module.
    def get_transferred_identities(
        self,
        video,
        identification_model,
        network_params,
        knowledge_transfer_info_dict,
    ):
        (
            images,
            _,
        ) = self.first_global_fragment_for_accumulation.get_images_and_labels(
            video.identification_images_file_paths, scope="identity_transfer"
        )
        images = np.asarray(images)

        assigner = assign(identification_model, images, network_params)

        compute_identification_statistics_for_non_accumulated_fragments(
            self.first_global_fragment_for_accumulation.individual_fragments,
            assigner,
            network_params.number_of_classes,
        )

        # Check certainties of the individual fragments in the global fragment
        # for individual_fragment_identifier in global_fragment.individual_fragments_identifiers:
        [
            setattr(fragment, "_acceptable_for_training", True)
            for fragment in self.first_global_fragment_for_accumulation.individual_fragments
        ]

        for (
            fragment
        ) in self.first_global_fragment_for_accumulation.individual_fragments:
            if AccumulationManager.is_not_certain(
                fragment, conf.CERTAINTY_THRESHOLD
            ):
                logger.debug(
                    "Identity transfer failed because a fragment is not certain enough"
                )
                logger.debug(
                    "CERTAINTY_THRESHOLD %.2f, fragment certainty %.2f"
                    % (conf.CERTAINTY_THRESHOLD, fragment.certainty)
                )
                identities = _abort_knowledge_transfer_on_same_animals(
                    video, identification_model
                )
                return identities

        (
            P1_array,
            index_individual_fragments_sorted_by_P1_max_to_min,
        ) = AccumulationManager.get_P1_array_and_argsort(
            self.first_global_fragment_for_accumulation
        )

        # assign temporary identity to individual fragments by hierarchical P1
        for (
            index_individual_fragment
        ) in index_individual_fragments_sorted_by_P1_max_to_min:
            fragment = self.first_global_fragment_for_accumulation.individual_fragments[
                index_individual_fragment
            ]

            if AccumulationManager.p1_below_random(
                P1_array, index_individual_fragment, fragment
            ):
                logger.debug(
                    "Identity transfer failed because P1 is below random"
                )
                identities = _abort_knowledge_transfer_on_same_animals(
                    video, identification_model
                )
                return identities
            else:
                temporary_id = np.argmax(
                    P1_array[index_individual_fragment, :]
                )
                if not fragment.check_consistency_with_coexistent_individual_fragments(
                    temporary_id
                ):
                    logger.debug(
                        "Identity transfer failed because the identities are not consistent"
                    )
                    identities = _abort_knowledge_transfer_on_same_animals(
                        video, identification_model
                    )
                    return identities
                else:
                    P1_array = AccumulationManager.set_fragment_temporary_id(
                        fragment,
                        temporary_id,
                        P1_array,
                        index_individual_fragment,
                    )

        # Check if the global fragment is unique after assigning the identities
        if not self.first_global_fragment_for_accumulation.is_unique:
            logger.debug(
                "Identity transfer failed because the identities are not unique"
            )
            identities = _abort_knowledge_transfer_on_same_animals(
                video, identification_model
            )
            logger.info(
                "Identity transfer is not possible. Identities will be intialized"
            )
        else:
            video._first_global_fragment_knowledge_transfer_identities = [
                fragment.temporary_id
                for fragment in self.first_global_fragment_for_accumulation.individual_fragments
            ]
            if (
                video.user_defined_parameters["number_of_animals"]
                == knowledge_transfer_info_dict["number_of_classes"]
            ):
                identities = (
                    video._first_global_fragment_knowledge_transfer_identities
                )
            elif (
                video.user_defined_parameters["number_of_animals"]
                < knowledge_transfer_info_dict["number_of_classes"]
            ):
                identities = range(
                    video.user_defined_parameters["number_of_animals"]
                )
            logger.info("Identities transferred successfully")

        return identities

    def compute_maximum_number_of_images(self):
        """Computes and sets the maximum number of images in the global
        fragments"""
        self.maximum_number_of_images = max(
            [
                global_fragment.get_total_number_of_images()
                for global_fragment in self.global_fragments
            ]
        )

    def filter_candidates_global_fragments_for_accumulation(self):
        """Filters the global fragments by taking into account the minium
        number of images per individual fragments specified in
        :attr:`globalfragment.GlobalFragment.candidate_for_accumulation`
        """
        self.non_accumulable_global_fragments = [
            global_fragment
            for global_fragment in self.global_fragments
            if not global_fragment.candidate_for_accumulation
        ]
        self.global_fragments = [
            global_fragment
            for global_fragment in self.global_fragments
            if global_fragment.candidate_for_accumulation
        ]
        self.number_of_global_fragments = len(self.global_fragments)

    def _delete_fragments_from_global_fragments(self):
        for global_fragment in self.global_fragments:
            global_fragment.individual_fragments = None

    def relink_fragments_to_global_fragments(self, fragments):
        """Re-assigns the instances of :class:`fragment.Fragment` to each
        global fragment in the list of `global_fragments`.

        Parameters
        ----------
        fragments: list
            List of instances of the class :class:`fragment.Fragment`.
        """
        for global_fragment in self.global_fragments:
            global_fragment.set_individual_fragments(fragments)

    def save(self, global_fragments_path, fragments):
        """Saves an instance of the class.

        Before saving the insntances of fragments associated to every global
        fragment are removed and reseted them after saving. This
        prevents problems when pickling objects inside of objects.

        Parameters
        ----------
        global_fragments_path : str
            Path where the object will be stored
        fragments: list
            List of all the instances of the class :class:`fragment.Fragment`
            in the video.
        """
        logger.info(
            "saving list of global fragments at %s" % global_fragments_path
        )
        self._delete_fragments_from_global_fragments()
        np.save(global_fragments_path, self)
        # After saving the list of globa fragments the individual
        # fragments are deleted and we need to relink them again
        self.relink_fragments_to_global_fragments(fragments)

    @classmethod
    def load(self, path_to_load, fragments):
        """Loads an instance of the class saved with :meth:`save` and
        associates individual fragments to each global fragment by calling
        :meth:`~relink_fragments_to_global_fragments`

        Parameters
        ----------

        path_to_load : str
            Path where the object to be loaded is stored.
        fragments : list
            List of all the instances of the class :class:`fragment.Fragment`
            in the video.
        """
        logger.info("loading list of global fragments from %s" % path_to_load)
        list_of_global_fragments = np.load(
            path_to_load, allow_pickle=True
        ).item()
        list_of_global_fragments.relink_fragments_to_global_fragments(
            fragments
        )
        return list_of_global_fragments


def detect_global_fragments_core_first_frame(boolean_array):
    """Detects the frame where the core of a global fragment starts.

    A core of a global fragment is the part of the global fragment where all
    the individuals are visible, i.e. the number of animals in the frame equals
    the number of animals in the video :boolean_array: array with True
    where the number of animals in the frame equals the number of animals in
    the video.
    """
    if np.all(boolean_array):
        return [0]
    else:
        return [
            i
            for i in range(0, len(boolean_array))
            if (boolean_array[i] and not boolean_array[i - 1])
        ]


def check_global_fragments(blobs_in_video, num_animals):
    """Returns list of booleans indicating the frames where all animals are
    visible.

    The element of the array is True if.

    * each blob has a unique blob intersecting in the past and future.
    * number of blobs equals num_animals.

    Parameters
    ----------
    blobs_in_video : list
        List of lists of instances of the class :class:`blob.Blob`.

    Returns
    -------
    list
        List of booleans with lenth the number of frames in the video. An
        element is True if all the animals are visible in the frame.
    """

    def _all_blobs_in_a_fragment(blobs_in_frame):
        """Returns all the blobs in `blobs_in_frame` that are associated to an
        individual
        """
        return all([blob.is_an_individual for blob in blobs_in_frame])

    def _same_fragment_identifier(blobs_in_frame, blobs_in_frame_past):
        """Return True if the set of fragments identifiers in the current frame
        is the same as in the previous frame, otherwise returns false
        """
        condition_1 = set(
            [blob.fragment_identifier for blob in blobs_in_frame]
        ) == set([blob.fragment_identifier for blob in blobs_in_frame_past])
        condition_2 = (
            _all_blobs_in_a_fragment(blobs_in_frame_past)
            and len(blobs_in_frame_past) == num_animals
        )
        return condition_1 or not condition_2

    return [
        _all_blobs_in_a_fragment(blobs_in_frame)
        and len(blobs_in_frame) == num_animals
        and _same_fragment_identifier(blobs_in_frame, blobs_in_video[i - 1])
        for i, blobs_in_frame in enumerate(blobs_in_video)
    ]


def create_list_of_global_fragments(blobs_in_video, fragments, num_animals):
    """Creates the list of instances of the class
    :class:`~globalfragment.GlobalFragment`
     used to create :class:`.ListOfGlobalFragments`.

    Parameters
    ----------
    blobs_in_video : list
        List of lists with instances of the class class :class:`blob.Blob`).
    fragments : list
        List of instances of the class :class:`fragment.Fragment`
    num_animals : int
        Number of animals to be tracked as indicated by the user.

    Returns
    -------
    list
        list of instances of the class :class:`~globalfragment.GlobalFragment`

    """
    global_fragments_boolean_array = check_global_fragments(
        blobs_in_video, num_animals
    )
    indices_beginning_of_fragment = detect_global_fragments_core_first_frame(
        global_fragments_boolean_array
    )
    global_fragments = [
        GlobalFragment(blobs_in_video, fragments, i, num_animals)
        for i in indices_beginning_of_fragment
    ]
    logger.info("total number of global_fragments: %i" % len(global_fragments))
    logger.info(
        [
            gf.number_of_images_per_individual_fragment
            for gf in global_fragments
        ]
    )
    return global_fragments


def _get_frequencies_first_fragment_accumulated(id, num_animals, fragment):
    frequencies = np.zeros(num_animals)
    frequencies[id] = fragment.number_of_images
    return frequencies


def _abort_knowledge_transfer_on_same_animals(video, identification_model):
    identities = range(video.user_defined_parameters["number_of_animals"])
    identification_model.apply(fc_weights_reinit)
    logger.info(
        "Identity transfer failed. "
        "We proceed by transferring only the convolutional filters."
    )
    return identities
