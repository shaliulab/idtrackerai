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
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

from idtrackerai.fragment import Fragment
from idtrackerai.utils.py_utils import (
    append_values_to_lists,
    set_attributes_of_object_to_value,
)

logger = logging.getLogger("__main__.list_of_fragments")


class ListOfFragments(object):
    """Contains all the instances of the class :class:`fragment.Fragment`.

    Parameters
    ----------
    fragments : list
        List of instances of the class :class:`fragment.Fragment`.
    identification_images_file_paths : list
        List of strings with the paths to the files where the identification
        images are stored.
    """

    def __init__(self, fragments, identification_images_file_paths):
        self.fragments = fragments
        self.number_of_fragments = len(self.fragments)
        self.identification_images_file_paths = (
            identification_images_file_paths
        )

    def __len__(self):
        return len(self.fragments)

    # TODO: Check if the generated list is used at all.
    def get_fragment_identifier_to_index_list(self):
        """Creates a mapping between the attribute :attr:`fragments` and
        their identifiers build from the :class:`list_of_blobs.ListOfBlobs`

        Returns
        -------
        list
            Mapping from the collection of fragments to the list of fragment
            identifiers

        """
        fragments_identifiers = [
            fragment.identifier for fragment in self.fragments
        ]
        fragment_identifier_to_index = np.arange(len(fragments_identifiers))
        fragments_identifiers_argsort = np.argsort(fragments_identifiers)
        return fragment_identifier_to_index[fragments_identifiers_argsort]

    # TODO: if the resume feature is not active, this does not make sense|
    def reset(self, roll_back_to):
        """Resets all the fragment to a given processing step.

        Parameters
        ----------
        roll_back_to : str
            Name of the step at which the fragments should be reset.
            It can be 'fragmentation', 'pretraining', 'accumulation' or
            'assignment'

        See Also
        --------
        :meth:`fragment.Fragment.reset`
        """
        logger.warning("Reseting list_of_fragments")
        for fragment in self.fragments:
            fragment.reset(roll_back_to)
        logger.warning("Done")

    # TODO: maybe this should go to the accumulator manager
    def get_images_from_fragments_to_assign(self):
        """Take all the fragments that have not been used to train the idCNN
        and that are associated with an individual, and concatenates their
        images in order to feed them to the identification netowkr.

        Returns
        -------
        ndarray
            [number_of_images, height, width, number_of_channels]
        """
        images_lists = [
            list(zip(fragment.images, fragment.episodes))
            for fragment in self.fragments
            if not fragment.used_for_training and fragment.is_an_individual
        ]
        images = [image for images in images_lists for image in images]
        return np.asarray(
            load_identification_images(
                self.identification_images_file_paths, images
            )
        )

    # TODO: The following methods could be properties.
    # TODO: The following methods depend on the identification strategy.
    def compute_number_of_unique_images_used_for_pretraining(self):
        """Returns the number of images used for pretraining
        (without repetitions)

        Returns
        -------
        int
            Number of images used in pretraining
        """
        return sum(
            [
                fragment.number_of_images
                for fragment in self.fragments
                if fragment.used_for_pretraining
            ]
        )

    def compute_number_of_unique_images_used_for_training(self):
        """Returns the number of images used for training
        (without repetitions)

        Returns
        -------
        int
            Number of images used in training
        """
        return sum(
            [
                fragment.number_of_images
                for fragment in self.fragments
                if fragment.used_for_training
            ]
        )

    def compute_total_number_of_images_in_global_fragments(self):
        """Sets the number of images available in global fragments
        (without repetitions)"""
        self.number_of_images_in_global_fragments = sum(
            [
                fragment.number_of_images
                for fragment in self.fragments
                if fragment.identifier in self.accumulable_individual_fragments
            ]
        )
        return self.number_of_images_in_global_fragments

    def compute_ratio_of_images_used_for_pretraining(self):
        """Returns the ratio of images used for pretraining over the number of
        available images

        Returns
        -------
        float
            Ratio of images used for pretraining
        """
        return (
            self.compute_number_of_unique_images_used_for_pretraining()
            / self.number_of_images_in_global_fragments
        )

    def compute_ratio_of_images_used_for_training(self):
        """Returns the ratio of images used for training over the number of
        available images

        Returns
        -------
        float
            Ratio of images used for training
        """
        return (
            self.compute_number_of_unique_images_used_for_training()
            / self.number_of_images_in_global_fragments
        )

    def compute_P2_vectors(self):
        """Computes the P2_vector associated to every individual fragment. See
        :meth:`fragment.Fragment.compute_P2_vector`
        """
        [
            fragment.compute_P2_vector()
            for fragment in self.fragments
            if fragment.is_an_individual
        ]

    def get_number_of_unidentified_individual_fragments(self):
        """Returns the number of individual fragments that have not been
        identified during the fingerprint protocols cascade

        Returns
        -------
        int
            number of non-identified individual fragments
        """
        return len(
            [
                fragment
                for fragment in self.fragments
                if fragment.is_an_individual and not fragment.used_for_training
            ]
        )

    def get_next_fragment_to_identify(self):
        """Returns the next fragment to be identified after the cascade of
        training and identitication protocols by sorting according to the
        certainty computed with P2. See :attr:fragment.Fragment.certainty_P2`

        Returns
        -------
        :class:`fragment.Fragment`
            An instance of the class :class:`fragment.Fragment`
        """
        fragments = [
            fragment
            for fragment in self.fragments
            if fragment.is_an_individual
            and fragment.assigned_identities[0] is None
        ]
        fragments.sort(key=lambda x: x.certainty_P2, reverse=True)
        return fragments[0]

    def update_identification_images_dataset(self):
        """Updates the identification images files with the identity assigned
        to each fragment during the tracking process.
        """
        for file in self.identification_images_file_paths:
            with h5py.File(file, "a") as f:
                f.create_dataset(
                    "identities",
                    (f["identification_images"].shape[0], 1),
                    fillvalue=np.nan,
                )

        for fragment in tqdm(
            self.fragments,
            desc="Updating identities in identification images files",
        ):
            if fragment.used_for_training:
                for image, episode in zip(fragment.images, fragment.episodes):
                    with h5py.File(
                        self.identification_images_file_paths[episode], "a"
                    ) as f:
                        f["identities"][image] = fragment.identity

    def get_ordered_list_of_fragments(
        self, scope, first_frame_first_global_fragment
    ):
        """Sorts the fragments starting from the frame number
        `first_frame_first_global_fragment`. According to `scope` the sorting
        is done either "to the future" of "to the past" with respect to
        `first_frame_first_global_fragment`

        Parameters
        ----------
        scope : str
            either "to_the_past" or "to_the_future"
        first_frame_first_global_fragment : int
            frame number corresponding to the first frame in which all the
            individual fragments coexist in the first global fragment using
            in an iteration of the fingerprint protocol cascade

        Returns
        -------
        list
            list of sorted fragments

        """
        if scope == "to_the_past":
            fragments_subset = [
                fragment
                for fragment in self.fragments
                if fragment.start_end[1] <= first_frame_first_global_fragment
            ]
            fragments_subset.sort(key=lambda x: x.start_end[1], reverse=True)
        elif scope == "to_the_future":
            fragments_subset = [
                fragment
                for fragment in self.fragments
                if fragment.start_end[0] >= first_frame_first_global_fragment
            ]
            fragments_subset.sort(key=lambda x: x.start_end[0], reverse=False)
        return fragments_subset

    def save(self, fragments_path):
        """Save an instance of the object in disk,

        Parameters
        ----------
        fragments_path : str
            Path where the instance of the object will be stored.
        """
        logger.info("saving list of fragments at %s" % fragments_path)
        for fragment in self.fragments:
            fragment.coexisting_individual_fragments = None
        np.save(fragments_path, self)
        for fragment in self.fragments:
            fragment.get_coexisting_individual_fragments_indices(
                self.fragments
            )

    @staticmethod
    def load(path_to_load):
        """Loads a previously saved (see :meth:`save`) from the path
        `path_to_load`
        """
        logger.info("loading list of fragments from %s" % path_to_load)
        list_of_fragments = np.load(path_to_load, allow_pickle=True).item()
        for fragment in list_of_fragments.fragments:
            fragment.get_coexisting_individual_fragments_indices(
                list_of_fragments.fragments
            )
        return list_of_fragments

    # TODO: Consider not saving light list of fragments. Fragments now are light
    def create_light_list(self, attributes=None):
        """Creates a light version of an instance of
        :class:`list_of_fragments.ListOfFragments` by storing
        only the attributes listed in `attributes` in a list of dictionaries

        Parameters
        ----------
        attributes : list
            list of attributes to be stored

        Returns
        -------
        list
            list of dictionaries organised per fragment with keys the
            attributes listed in `attributes`
        """
        if attributes is None:
            attributes_to_discard = [
                "images",
                "coexisting_individual_fragments",
            ]
        return [
            {
                attribute: getattr(fragment, attribute)
                for attribute in fragment.__dict__.keys()
                if attribute not in attributes_to_discard
            }
            for fragment in self.fragments
        ]

    def save_light_list(self, accumulation_folder):
        """Saves a list of dictionaries created with the method
        :meth:`create_light_list` in the folder `accumulation_folder`.
        """
        np.save(
            os.path.join(accumulation_folder, "light_list_of_fragments.npy"),
            self.create_light_list(),
        )

    def load_light_list(self, accumulation_folder):
        """Loads a list of dictionaries created with the method
        :meth:`create_light_list` and saved with :meth:`save_light_list` from
        the folder `accumulation_folder`.
        """
        list_of_dictionaries = np.load(
            os.path.join(accumulation_folder, "light_list_of_fragments.npy"),
            allow_pickle=True,
        )
        self.update_fragments_dictionary(list_of_dictionaries)

    def update_fragments_dictionary(self, list_of_dictionaries):
        """Update fragment objects (see :class:`fragment.Fragment`) by
        considering a list of dictionaries.
        """
        assert len(list_of_dictionaries) == len(self.fragments)
        [
            fragment.__dict__.update(dictionary)
            for fragment, dictionary in zip(
                self.fragments, list_of_dictionaries
            )
        ]

    def get_new_images_and_labels_for_training(self):
        """Extract images and creates labels from every individual fragment
        that has not been used to train the network during the fingerprint
        protocols cascade.

        Returns
        -------
        list
            List of numpy arrays with shape [width, height, num channels]
        list
            labels
        """
        images = []
        labels = []
        for fragment in self.fragments:
            if (
                fragment.acceptable_for_training
                and not fragment.used_for_training
            ):
                assert fragment.is_an_individual
                images.extend(list(zip(fragment.images, fragment.episodes)))
                labels.extend(
                    [fragment.temporary_id] * fragment.number_of_images
                )
        if len(images) != 0:
            return images, np.asarray(labels)
        else:
            return None, None

    def get_accumulable_individual_fragments_identifiers(
        self, list_of_global_fragments
    ):
        """Gets the unique identifiers associated to individual fragments that
        can be accumulated.

        Parameters
        ----------
        list_of_global_fragments : :class:`list_of_global_fragments.ListOfGlobalFragments`
            Object collecting the global fragment objects (instances of the
            class :class:`globalfragment.GlobalFragment`) detected in the
            entire video.

        """
        self.accumulable_individual_fragments = set(
            [
                identifier
                for global_fragment in list_of_global_fragments.global_fragments
                for identifier in global_fragment.individual_fragments_identifiers
            ]
        )

    def get_not_accumulable_individual_fragments_identifiers(
        self, list_of_global_fragments
    ):
        """Gets the unique identifiers associated to individual fragments that
        cannot be accumulated.

        Parameters
        ----------
        list_of_global_fragments : :class:`list_of_global_fragments.ListOfGlobalFragments`
            Object collecting the global fragment objects (instances of the
            class :class:`globalfragment.GlobalFragment`) detected in the
            entire video.

        """
        self.not_accumulable_individual_fragments = (
            set(
                [
                    identifier
                    for global_fragment in list_of_global_fragments.non_accumulable_global_fragments
                    for identifier in global_fragment.individual_fragments_identifiers
                ]
            )
            - self.accumulable_individual_fragments
        )

    def set_fragments_as_accumulable_or_not_accumulable(self):
        """Set the attribute :attr:`fragment.Fragment.accumulable`"""
        for fragment in self.fragments:
            if fragment.identifier in self.accumulable_individual_fragments:
                setattr(fragment, "_accumulable", True)
            elif (
                fragment.identifier
                in self.not_accumulable_individual_fragments
            ):
                setattr(fragment, "_accumulable", False)
            else:
                setattr(fragment, "_accumulable", None)

    # TODO: list_of_global_fragments is not needed here
    def get_stats(self, list_of_global_fragments):
        """Collects the following counters from the fragments.

        * number_of_fragments
        * number_of_crossing_fragments
        * number_of_individual_fragments
        * number_of_individual_fragments_not_in_a_global_fragment
        * number_of_accumulable_individual_fragments
        * number_of_not_accumulable_individual_fragments
        * number_of_accumualted_individual_fragments
        * number_of_globally_accumulated_individual_fragments
        * number_of_partially_accumulated_individual_fragments
        * number_of_blobs
        * number_of_crossing_blobs
        * number_of_individual_blobs
        * number_of_individual_blobs_not_in_a_global_fragment
        * number_of_accumulable_individual_blobs
        * number_of_not_accumulable_individual_blobs
        * number_of_accumualted_individual_blobs
        * number_of_globally_accumulated_individual_blobs
        * number_of_partially_accumulated_individual_blobs

        Parameters
        ----------
        list_of_global_fragments : :class:`list_of_global_fragments.ListOfGlobalFragments`
            Object collecting the global fragment objects (instances of the
            class :class:`global_fragment.GlobalFragment`) detected in the
            entire video

        Returns
        -------
        dict
            Dictionary with the counters mentioned above

        """
        # number of fragments per class
        self.number_of_crossing_fragments = sum(
            [fragment.is_a_crossing for fragment in self.fragments]
        )
        self.number_of_individual_fragments = sum(
            [fragment.is_an_individual for fragment in self.fragments]
        )
        self.number_of_individual_fragments_not_in_a_global_fragment = sum(
            [
                not fragment.is_in_a_global_fragment
                and fragment.is_an_individual
                for fragment in self.fragments
            ]
        )
        self.number_of_accumulable_individual_fragments = len(
            self.accumulable_individual_fragments
        )
        self.number_of_not_accumulable_individual_fragments = len(
            self.not_accumulable_individual_fragments
        )
        fragments_not_accumualted = set(
            [
                fragment.identifier
                for fragment in self.fragments
                if not fragment.used_for_training
            ]
        )
        self.number_of_not_accumulated_individual_fragments = len(
            self.accumulable_individual_fragments & fragments_not_accumualted
        )
        self.number_of_globally_accumulated_individual_fragments = sum(
            [
                fragment.accumulated_globally and fragment.is_an_individual
                for fragment in self.fragments
            ]
        )
        self.number_of_partially_accumulated_individual_fragments = sum(
            [
                fragment.accumulated_partially and fragment.is_an_individual
                for fragment in self.fragments
            ]
        )
        # number of blobs per class
        self.number_of_blobs = sum(
            [fragment.number_of_images for fragment in self.fragments]
        )
        self.number_of_crossing_blobs = sum(
            [
                fragment.is_a_crossing * fragment.number_of_images
                for fragment in self.fragments
            ]
        )
        self.number_of_individual_blobs = sum(
            [
                fragment.is_an_individual * fragment.number_of_images
                for fragment in self.fragments
            ]
        )
        self.number_of_individual_blobs_not_in_a_global_fragment = sum(
            [
                (
                    not fragment.is_in_a_global_fragment
                    and fragment.is_an_individual
                )
                * fragment.number_of_images
                for fragment in self.fragments
            ]
        )
        self.number_of_accumulable_individual_blobs = sum(
            [
                fragment.accumulable * fragment.number_of_images
                for fragment in self.fragments
                if fragment.accumulable is not None
            ]
        )
        self.number_of_not_accumulable_individual_blobs = sum(
            [
                (not fragment.accumulable) * fragment.number_of_images
                for fragment in self.fragments
                if fragment.accumulable is not None
            ]
        )
        fragments_not_accumualted = (
            self.accumulable_individual_fragments
            & set(
                [
                    fragment.identifier
                    for fragment in self.fragments
                    if not fragment.used_for_training
                ]
            )
        )
        self.number_of_not_accumulated_individual_blobs = sum(
            [
                fragment.number_of_images
                for fragment in self.fragments
                if fragment.identifier in fragments_not_accumualted
            ]
        )
        self.number_of_globally_accumulated_individual_blobs = sum(
            [
                (fragment.accumulated_globally and fragment.is_an_individual)
                * fragment.number_of_images
                for fragment in self.fragments
                if fragment.accumulated_globally is not None
            ]
        )
        self.number_of_partially_accumulated_individual_blobs = sum(
            [
                (fragment.accumulated_partially and fragment.is_an_individual)
                * fragment.number_of_images
                for fragment in self.fragments
                if fragment.accumulated_partially is not None
            ]
        )

        logger.info("number_of_fragments %i" % self.number_of_fragments)
        logger.info(
            "number_of_crossing_fragments %i"
            % self.number_of_crossing_fragments
        )
        logger.info(
            "number_of_individual_fragments %i "
            % self.number_of_individual_fragments
        )
        logger.info(
            "number_of_individual_fragments_not_in_a_global_fragment %i"
            % self.number_of_individual_fragments_not_in_a_global_fragment
        )
        logger.info(
            "number_of_accumulable_individual_fragments %i"
            % self.number_of_accumulable_individual_fragments
        )
        logger.info(
            "number_of_not_accumulable_individual_fragments %i"
            % self.number_of_not_accumulable_individual_fragments
        )
        logger.info(
            "number_of_not_accumulated_individual_fragments %i"
            % self.number_of_not_accumulated_individual_fragments
        )
        logger.info(
            "number_of_globally_accumulated_individual_fragments %i"
            % self.number_of_globally_accumulated_individual_fragments
        )
        logger.info(
            "number_of_partially_accumulated_individual_fragments %i"
            % self.number_of_partially_accumulated_individual_fragments
        )
        attributes_to_return = [
            "number_of_fragments",
            "number_of_crossing_fragments",
            "number_of_individual_fragments",
            "number_of_individual_fragments_not_in_a_global_fragment",
            "number_of_accumulable_individual_fragments",
            "number_of_not_accumulable_individual_fragments",
            "number_of_accumualted_individual_fragments",
            "number_of_globally_accumulated_individual_fragments",
            "number_of_partially_accumulated_individual_fragments",
            "number_of_blobs",
            "number_of_crossing_blobs",
            "number_of_individual_blobs",
            "number_of_individual_blobs_not_in_a_global_fragment",
            "number_of_accumulable_individual_blobs",
            "number_of_not_accumulable_individual_blobs",
            "number_of_accumualted_individual_blobs",
            "number_of_globally_accumulated_individual_blobs",
            "number_of_partially_accumulated_individual_blobs",
        ]
        return {
            key: getattr(self, key)
            for key in self.__dict__
            if key in attributes_to_return
        }


def create_list_of_fragments(blobs_in_video, number_of_animals):
    """Generate a list of instances of :class:`fragment.Fragment` collecting
    all the fragments in the video.

    Parameters
    ----------
    blobs_in_video : list
        list of the blob objects (see class :class:`blob.Blob`) generated
        from the blobs segmented in the video
    number_of_animals : int
        Number of animals to track as defined by the user

    Returns
    -------
    list
        list of instances of :class:`fragment.Fragment`

    """
    attributes_to_set = ["_image_for_identification", "_next", "_previous"]
    fragments = []
    used_fragment_identifiers = set()

    for blobs_in_frame in tqdm(
        blobs_in_video, desc="creating list of fragments"
    ):
        for blob in blobs_in_frame:
            current_fragment_identifier = blob.fragment_identifier
            if current_fragment_identifier not in used_fragment_identifiers:
                images = (
                    [blob.identification_image_index]
                    if blob.is_an_individual
                    else [None]
                )
                bounding_boxes = (
                    [blob.bounding_box_in_frame_coordinates]
                    if blob.is_a_crossing
                    else []
                )
                centroids = [blob.centroid]
                areas = [blob.area]
                episodes = [blob.episode]
                start = blob.frame_number
                current = blob

                while (
                    len(current.next) > 0
                    and current.next[0].fragment_identifier
                    == current_fragment_identifier
                ):
                    current = current.next[0]
                    (images, centroids, episodes) = append_values_to_lists(
                        [
                            current.identification_image_index,
                            current.centroid,
                            current.episode,
                        ],
                        [images, centroids, episodes],
                    )

                end = current.frame_number

                fragment = Fragment(
                    current_fragment_identifier,
                    (
                        start,
                        end + 1,
                    ),  # it is not inclusive to follow Python convention
                    blob.blob_index,
                    images,
                    centroids,
                    episodes,
                    blob.is_an_individual,
                    blob.is_a_crossing,
                    number_of_animals,
                )
                used_fragment_identifiers.add(current_fragment_identifier)
                fragments.append(fragment)

            set_attributes_of_object_to_value(
                blob, attributes_to_set, value=None
            )
    logger.info("getting coexisting individual fragments indices")
    [
        fragment.get_coexisting_individual_fragments_indices(fragments)
        for fragment in fragments
    ]
    return fragments


def load_identification_images(
    identification_images_file_paths, images_indices
):
    """Loads the identification images from disk.

    Parameters
    ----------
    identification_images_file_paths : list
        List of strings with the paths to the files where the images are
        stored.
    images_indices : list
        List of tuples (image_index, episode) that indicate each of the images
        to be loaded

    Returns
    -------
    Numpy array
        Numpy array of shape [number of images, width, height]
    """
    images = []
    for (image, episode) in tqdm(
        images_indices, desc="Reading identification images from the disk"
    ):
        with h5py.File(identification_images_file_paths[episode], "r") as f:
            dataset = f["identification_images"]
            images.append(dataset[image, ...])

    images = np.asarray(images)
    # mean = np.mean(images, axis=(1, 2))[:, np.newaxis, np.newaxis]
    # std = np.std(images, axis=(1, 2))[:, np.newaxis, np.newaxis]
    # images = ((images - mean)/std).astype('float32')
    return images
