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


class PreprocessingPreviewAPI(object):
    def __init__(self, chosen_video=None, **kwargs):

        #: Chosen_Video: ?
        self.chosen_video = chosen_video
        #: int: Used to filter the image to find the blobs
        self.min_threshold = (
            chosen_video.video.min_threshold
            if chosen_video.video is not None
            else 0
        )
        #: int: Used to filter the image to find the blobs
        self.max_threshold = (
            chosen_video.video.max_threshold
            if chosen_video.video is not None
            else 135
        )
        #: int: Minimum area of a blob
        self.min_area = (
            chosen_video.video.min_area
            if chosen_video.video is not None
            else 150
        )
        #: int: Maximum area of a blob
        self.max_area = (
            chosen_video.video.max_area
            if chosen_video.video is not None
            else 60000
        )

        #: float: ?
        self.resolution_reduction = (
            chosen_video.video.resolution_reduction
            if chosen_video.video is not None
            else 1
        )
        #: int: Number of animals to track
        self.number_of_animals = (
            chosen_video.video.number_of_animals
            if chosen_video.video.number_of_animals is not None
            else 1
        )
        #: ListOfFragments: List of fragments ( blobs paths before crossing )
        self.list_of_fragments = None
        #: list(GlobalFragment): ?
        self.list_of_global_fragments = None
        #: ?: ?
        self.crossing_detector_trainer = None
        #: boolean: ?
        self.resegmentation_step_finished = True
        #: list(int): Indexes of the frames with more blobs than animals to track
        self.frames_with_more_blobs_than_animals = None

    def init_preview(self):
        ## TODO: Check how important is this method. Variables in
        # init_segment_zero do not seem to be used
        logger.debug("init_preview")
        self.init_preproc_parameters()

        ## TODO: Check this
        self.bkg = self.chosen_video.video.bkg
        self.ROI = (
            self.chosen_video.video.ROI
            if self.chosen_video.video.ROI is not None
            else np.ones(
                (
                    self.chosen_video.video.original_height,
                    self.chosen_video.video.original_width,
                ),
                dtype="uint8",
            )
            * 255
        )

        logger.debug("init_segment_zero")
        self.init_segment_zero()

    def init_segment_zero(self):
        ## TODO: This seems not to be used
        self.currentSegment = 0
        self.areas_plotted = False
        self.number_of_detected_blobs = [0]

    def init_preproc_parameters(self):

        logger.debug("init_preproc_parameters")

        if (
            self.chosen_video.old_video is not None
            and self.chosen_video.old_video._has_been_preprocessed
        ):

            self.max_threshold = self.chosen_video.old_video.max_threshold
            self.min_threshold = self.chosen_video.old_video.min_threshold
            self.min_area = self.chosen_video.old_video.min_area
            self.max_area = self.chosen_video.old_video.max_area
            self.resolution_reduction = (
                self.chosen_video.old_video.resolution_reduction
            )
            self.number_of_animals = (
                self.chosen_video.old_video.number_of_animals
            )
            self.chosen_video.video.resolution_reduction = (
                self.chosen_video.old_video.resolution_reduction
            )

        else:

            self.max_threshold = conf.MAX_THRESHOLD_DEFAULT
            self.min_threshold = conf.MIN_THRESHOLD_DEFAULT
            self.min_area = conf.MIN_AREA_DEFAULT
            self.max_area = conf.MAX_AREA_DEFAULT
            self.resolution_reduction = (
                self.chosen_video.video.resolution_reduction
            )
            if self.chosen_video.video._original_ROI is None:
                self.chosen_video.video._original_ROI = (
                    np.ones(
                        (
                            self.chosen_video.video.original_height,
                            self.chosen_video.video.original_width,
                        ),
                        dtype="uint8",
                    )
                    * 255
                )

    def compute_list_of_blobs(self, *args):
        ### TODO: This call should go to the method segment below
        self.blobs = segment(self.chosen_video.video)
        self.chosen_video.list_of_blobs = ListOfBlobs(
            blobs_in_video=self.blobs
        )
        self.chosen_video.video.create_preprocessing_folder()

    def segment(self, min_threshold, max_threshold, min_area, max_area):
        self.chosen_video.video._segmentation_time = time.time()
        logger.debug("segment")
        self.chosen_video.video._max_threshold = max_threshold
        self.chosen_video.video._min_threshold = min_threshold
        self.chosen_video.video._min_area = min_area
        self.chosen_video.video._max_area = max_area
        logger.debug("segment 1")
        self.chosen_video.video.resolution_reduction = (
            self.resolution_reduction
        )
        logger.debug("segment 2")
        self.chosen_video.video.save()

    def check_segmentation_consistency(self):
        """
        :return: True if the segmentation is consistence with the number of animals, otherwise return False
        """
        (
            self.chosen_video.video.frames_with_more_blobs_than_animals,
            self.chosen_video.video._maximum_number_of_blobs,
        ) = self.chosen_video.list_of_blobs.check_maximal_number_of_blob(
            self.chosen_video.video.number_of_animals,
            return_maximum_number_of_blobs=True,
        )
        self.frames_with_more_blobs_than_animals = (
            self.chosen_video.video.frames_with_more_blobs_than_animals
        )

        """
        #This call is used in the GUI to re-segment the image in the case the 
        tracking returned more blobs than expected
        #In the case of the API this is not necessary
        if len(self.frames_with_more_blobs_than_animals) > 0 and 
        (check_segmentation_consistency or 
        self.chosen_video.video.number_of_animals == 1):
            self.resegmentation_step_finished = True

            if resegment: self.resegmentation()
        """
        return (
            len(self.chosen_video.video.frames_with_more_blobs_than_animals)
            == 0
        )

    def save_list_of_blobs_segmented(self):
        self.chosen_video.video._has_been_segmented = True

        if len(self.chosen_video.list_of_blobs.blobs_in_video[-1]) == 0:
            self.chosen_video.list_of_blobs.blobs_in_video = self.chosen_video.list_of_blobs.blobs_in_video[
                :-1
            ]
            self.chosen_video.list_of_blobs.number_of_frames = len(
                self.chosen_video.list_of_blobs.blobs_in_video
            )
            self.chosen_video.video._number_of_frames = (
                self.chosen_video.list_of_blobs.number_of_frames
            )

        self.chosen_video.video.save()
        self.chosen_video.video._segmentation_time = (
            time.time() - self.chosen_video.video.segmentation_time
        )

    def compute_model_area_and_connect(self):
        self.chosen_video.video._crossing_detector_time = time.time()

        (
            self.chosen_video.video._model_area,
            self.chosen_video.video._median_body_length,
        ) = self.chosen_video.list_of_blobs.compute_model_area_and_body_length(
            self.chosen_video.video.number_of_animals
        )
        self.chosen_video.video.compute_identification_image_size(
            self.chosen_video.video.median_body_length
        )

        start = time.time()
        self.chosen_video.list_of_blobs.set_images_for_identification(
            self.chosen_video.video
        )
        print(
            "Setting images for identification took {}".format(
                time.time() - start
            )
        )
        if not self.chosen_video.list_of_blobs.blobs_are_connected:
            start = time.time()
            self.chosen_video.list_of_blobs.compute_overlapping_between_subsequent_frames()
            print("Connecting blobs took {}".format(time.time() - start))

    def train_and_apply_crossing_detector(self):

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
            self.chosen_video.list_of_blob.save(
                self.chosen_video.video,
                self.chosen_video.video.blobs_path_segmented,
            )

    def generate_list_of_fragments_and_global_fragments(self):
        self.chosen_video.video._fragmentation_time = time.time()

        if self.chosen_video.video.number_of_animals != 1:
            if not self.chosen_video.list_of_blobs.blobs_are_connected:
                self.chosen_video.list_of_blobs.compute_overlapping_between_subsequent_frames()
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
            self.list_of_fragments = ListOfFragments(
                fragments,
                self.chosen_video.video.identification_images_file_paths,
            )
            self.chosen_video.video._fragment_identifier_to_index = (
                self.list_of_fragments.get_fragment_identifier_to_index_list()
            )
            global_fragments = create_list_of_global_fragments(
                self.chosen_video.list_of_blobs.blobs_in_video,
                self.list_of_fragments.fragments,
                self.chosen_video.video.number_of_animals,
            )
            # Create list of global fragments
            self.list_of_global_fragments = ListOfGlobalFragments(
                global_fragments
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
            self.list_of_fragments.save(self.chosen_video.video.fragments_path)
            self.chosen_video.list_of_fragments = self.list_of_fragments
        else:
            self.chosen_video.video._number_of_unique_images_in_global_fragments = (
                None
            )
            self.chosen_video.video._maximum_number_of_images_in_global_fragments = (
                None
            )
        self.chosen_video.video._has_been_preprocessed = True
        self.chosen_video.list_of_blobs.save(
            self.chosen_video.video, self.chosen_video.video.blobs_path
        )
        if self.chosen_video.video.number_of_animals != 1:
            self.list_of_global_fragments.save(
                self.chosen_video.video.global_fragments_path,
                self.list_of_fragments.fragments,
            )
            self.chosen_video.list_of_global_fragments = (
                self.list_of_global_fragments
            )
        self.chosen_video.video._fragmentation_time = (
            time.time() - self.chosen_video.video.fragmentation_time
        )
        self.chosen_video.video.save()
