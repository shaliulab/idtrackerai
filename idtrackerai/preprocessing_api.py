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

import os
import logging
import time

import numpy as np
from confapp import conf

from idtrackerai.crossing_detector import detect_crossings
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import (
    ListOfFragments,
    create_list_of_fragments,
)
from idtrackerai.list_of_global_fragments import (
    ListOfGlobalFragments,
    create_list_of_global_fragments,
)
from idtrackerai.preprocessing.segmentation import segment

logger = logging.getLogger(__name__)


class PreprocessingAPI(object):
    def __init__(self, chosen_video=None, **kwargs):

        #: Chosen_Video: ?
        self.chosen_video = chosen_video
        #: ListOfBlobs
        self.blobs = None
        #: ListOfFragments: List of fragments ( blobs paths before crossing )
        self.list_of_fragments = None
        #: list(GlobalFragment): ?
        self.list_of_global_fragments = None
        #: ?: ?
        self.crossing_detector_trainer = None
        # #: boolean: ?
        # self.resegmentation_step_finished = True
        self.chosen_video.video.create_preprocessing_folder()

    def detect_animals(self):
        """
        Animal detection computational block.

        Detects the animals as blobs of pixels following the user-defined
        animal detection parameters

        Generates a ListOfBlobs object
        """
        self.chosen_video.video._detect_animals_time = time.time()
        self.chosen_video.video.save()
        self.chosen_video.video.create_images_folders()  # for ram optimization
        logger.info("--> segment")
        self.blobs = segment(self.chosen_video.video)
        logger.info("--> ListOfBlobs")
        self.chosen_video.list_of_blobs = ListOfBlobs(
            blobs_in_video=self.blobs
        )

    def check_segmentation_consistency(self):
        """
        Checks whether there is some frame with more blobs than number
        of animals in the video (as specified by the user)
        Then the segmentation is considered inconsistent

        :return: True if the segmentation is consistent else False.
        """
        logger.info("--> check_segmentation_consistency")
        (
            self.chosen_video.video.frames_with_more_blobs_than_animals,
            self.chosen_video.video._maximum_number_of_blobs,
        ) = self.chosen_video.list_of_blobs.check_maximal_number_of_blob(
            self.chosen_video.video.number_of_animals,
            return_maximum_number_of_blobs=True,
        )
        consistent_segmentation = (
            len(self.chosen_video.video.frames_with_more_blobs_than_animals)
            == 0
        )
        return consistent_segmentation

    def save_inconsistent_frames(self):
        logger.info("--> save_inconsistent_frames")
        outfile_path = os.path.join(
            self.chosen_video.video.session_folder, "inconsistent_frames.csv"
        )
        with open(outfile_path, "w") as outfile:
            outfile.write(
                "\n".join(
                    map(
                        str,
                        self.chosen_video.video.frames_with_more_blobs_than_animals,
                    )
                )
            )
        return outfile_path

    def save_list_of_blobs_segmented(self):
        logger.info("--> save_list_of_blobs_segmented")
        if len(self.chosen_video.list_of_blobs.blobs_in_video[-1]) == 0:
            self.chosen_video.list_of_blobs.blobs_in_video = (
                self.chosen_video.list_of_blobs.blobs_in_video[:-1]
            )
            self.chosen_video.list_of_blobs.number_of_frames = len(
                self.chosen_video.list_of_blobs.blobs_in_video
            )
            self.chosen_video.video._number_of_frames = (
                self.chosen_video.list_of_blobs.number_of_frames
            )

        self.chosen_video.video._detect_animals_time = (
            time.time() - self.chosen_video.video.detect_animals_time
        )
        self.chosen_video.video._has_animals_detected = True

    def compute_model_area(self):
        logger.info("--> compute_model_area")
        self.chosen_video.video._compute_model_area_time = time.time()

        (
            self.chosen_video.video._model_area,
            self.chosen_video.video._median_body_length,
        ) = self.chosen_video.list_of_blobs.compute_model_area_and_body_length(
            self.chosen_video.video.number_of_animals
        )
        self.chosen_video.video._compute_model_area_time = (
            time.time() - self.chosen_video.video._compute_model_area_time
        )
        self.chosen_video.video._has_model_area = True

    def set_identification_images(self):
        logger.info("--> set_identification_images")
        self.chosen_video.video._set_identification_images_time = time.time()
        self.chosen_video.video.compute_identification_image_size(
            self.chosen_video.video.median_body_length
        )
        self.chosen_video.list_of_blobs.set_images_for_identification(
            self.chosen_video.video
        )
        self.chosen_video.video._set_identification_images_time = (
            time.time()
            - self.chosen_video.video.set_identification_images_time
        )
        self.chosen_video.video._has_identification_images = True

    def connect_list_of_blobs(self):
        logger.info(
            "--> connect_list_of_blobs (crossing detector overlapping heuristic)"
        )
        self.chosen_video.video._connect_list_of_blobs_id_cd_time = time.time()
        if not self.chosen_video.list_of_blobs.blobs_are_connected:
            self.chosen_video.list_of_blobs.compute_overlapping_between_subsequent_frames()
        self.chosen_video.video._connect_list_of_blobs_id_cd_time = (
            time.time()
            - self.chosen_video.video.connect_list_of_blobs_id_cd_time
        )

    def train_and_apply_crossing_detector(self):
        logger.info("--> train_and_apply_crossing_detector")
        self.chosen_video.video._training_crossing_detector_time = time.time()

        if self.chosen_video.video.number_of_animals != 1:
            self.crossing_detector_trainer = detect_crossings(
                self.chosen_video.list_of_blobs,
                self.chosen_video.video,
                self.chosen_video.video.model_area,
                use_network=True,
                return_store_objects=True,
                plot_flag=conf.PLOT_CROSSING_DETECTOR,
            )
        else:
            self.chosen_video.list_of_blob = detect_crossings(
                self.chosen_video.list_of_blobs,
                self.chosen_video.video,
                self.chosen_video.video.model_area,
                use_network=False,
                return_store_objects=False,
                plot_flag=conf.PLOT_CROSSING_DETECTOR,
            )

        self.chosen_video.video._training_crossing_detector_time = (
            time.time()
            - self.chosen_video.video.training_crossing_detector_time
        )
        self.chosen_video.video._has_crossings_detected = True

    def generate_list_of_fragments_and_global_fragments(self):
        self.chosen_video.video._fragmentation_time = time.time()

        if self.chosen_video.video.number_of_animals != 1:
            self._generate_list_of_fragments_and_global_fragments()
        else:
            self.chosen_video.video._number_of_unique_images_in_global_fragments = (
                None
            )
            self.chosen_video.video._maximum_number_of_images_in_global_fragments = (
                None
            )
        self.chosen_video.video._has_been_fragmented = True
        self.chosen_video.video._fragmentation_time = (
            time.time() - self.chosen_video.video.fragmentation_time
        )

    def _generate_list_of_fragments_and_global_fragments(self):
        if not self.chosen_video.list_of_blobs.blobs_are_connected:
            self.chosen_video.list_of_blobs.compute_overlapping_between_subsequent_frames()

        # FRAGMENTS
        # Individual fragments and crossing fragments identifiers
        self.chosen_video.list_of_blobs.compute_fragment_identifier_and_blob_index(
            max(
                self.chosen_video.video.number_of_animals,
                self.chosen_video.video.maximum_number_of_blobs,
            )
        )
        self.chosen_video.list_of_blobs.compute_crossing_fragment_identifier()
        fragments = create_list_of_fragments(
            self.chosen_video.list_of_blobs.blobs_in_video,
            self.chosen_video.video.number_of_animals,
        )

        # List of fragments
        self.list_of_fragments = ListOfFragments(
            fragments,
            self.chosen_video.video.identification_images_file_paths,
        )
        # Populate chosen_video.list_of_fragments to save it if it chrashes
        self.chosen_video.list_of_fragments = self.list_of_fragments

        # Update video object 1
        self.chosen_video.video._fragment_identifier_to_index = (
            self.list_of_fragments.get_fragment_identifier_to_index_list()
        )

        # GLOBAL FRAGMENTS
        global_fragments = create_list_of_global_fragments(
            self.chosen_video.list_of_blobs.blobs_in_video,
            self.list_of_fragments.fragments,
            self.chosen_video.video.number_of_animals,
        )
        # Create list of global fragments
        self.list_of_global_fragments = ListOfGlobalFragments(global_fragments)
        # Populate chosen_video.list_of_global_fragments to save it
        # if it chrashes
        self.chosen_video.list_of_global_fragments = (
            self.list_of_global_fragments
        )
        self.chosen_video.video.number_of_global_fragments = (
            self.list_of_global_fragments.number_of_global_fragments
        )
        # Filter candidates global fragments for accumulation
        self.list_of_global_fragments.filter_candidates_global_fragments_for_accumulation()
        self.chosen_video.video.number_of_global_fragments_candidates_for_accumulation = (
            self.list_of_global_fragments.number_of_global_fragments
        )
        self.list_of_global_fragments.relink_fragments_to_global_fragments(
            self.list_of_fragments.fragments
        )
        self.list_of_global_fragments.compute_maximum_number_of_images()
        self.chosen_video.video._maximum_number_of_images_in_global_fragments = (
            self.list_of_global_fragments.maximum_number_of_images
        )
        self.list_of_fragments.get_accumulable_individual_fragments_identifiers(
            self.list_of_global_fragments
        )
        self.list_of_fragments.get_not_accumulable_individual_fragments_identifiers(
            self.list_of_global_fragments
        )
        self.list_of_fragments.set_fragments_as_accumulable_or_not_accumulable()
        self.chosen_video.video._number_of_unique_images_in_global_fragments = (
            self.list_of_fragments.compute_total_number_of_images_in_global_fragments()
        )
        # self.list_of_fragments.save(self.chosen_video.video.fragments_path)
        # self.chosen_video.list_of_fragments = self.list_of_fragments
