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

from abc import ABC, abstractmethod
import logging
import time

from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.crossings_detection.crossing_detector import detect_crossings
from idtrackerai.crossings_detection.model_area import (
    compute_model_area_and_body_length,
)

logger = logging.getLogger(__name__)


class CrossingsDetectionABC(ABC):
    def __init__(self, video: Video, list_of_blobs: ListOfBlobs):
        """
        Classifies all the blobs in list_of_blobs as individuals or crossings
        """
        self.video = video
        self.list_of_blobs = list_of_blobs

    def __call__(self):
        self.video._crossing_detector_time = time.time()
        self.video.create_crossings_detector_folder()
        self.classify_blobs_as_crossings_or_individuals()
        assert len(self.list_of_blobs) == self.video.number_of_frames
        self.video._crossing_detector_time = (
            time.time() - self.video.crossing_detector_time
        )

    @abstractmethod
    def classify_blobs_as_crossings_or_individuals(self):
        pass


class CrossingsDetectionAPI(CrossingsDetectionABC):
    """
    This crossings detector works under the following assumptions
        1. The number of animals in the video is known (given by the user)
        2. There are frames in the video where all animals are separated from
        each other.
        3. All animals have a similar size
        4. The frame rate of the video is higher enough so that consecutive
        segmented blobs of pixels of the same animal overlap, i.e. some of the
        pixels representing the animal A in frame i are the same in the
        frame i+1.

    NOTE: This crossing detector sets the identification images that will be
    used to identify the animals
    """

    def __init__(self, video: Video, list_of_blobs: ListOfBlobs):
        super().__init__(video, list_of_blobs)
        self.model_area = None
        self.median_body_length = None

    def __call__(self):
        super().__call__()
        # TODO: ideally video does not contain this information as a different
        # crossing detector might might use this info
        self.video._median_body_length = self.median_body_length
        self.video._model_area = self.model_area

    def classify_blobs_as_crossings_or_individuals(self):
        self._estimate_single_indiviual_size()
        self._set_identification_images()
        self._connect_list_of_blobs()
        self._train_and_apply_crossing_detector()
        self.video._has_crossings_detected = True

    def _estimate_single_indiviual_size(self):
        """
        Computes a model_area of the size of single animals using frames of the
        video where all animals are separated from each other. In these frames
        the number of segmented blobs is the same as the number of animals in
        the video. So, all blobs are individual animals.

        It also estimates them median_body_length of single individuals.

        See Also
        --------
        :class:`~idtrackerai.crossigns_detection.model_area.ModelArea`
        """
        logger.info("--> compute_model_area")
        (
            self.model_area,
            self.median_body_length,
        ) = compute_model_area_and_body_length(
            self.list_of_blobs,
            self.video.user_defined_parameters["number_of_animals"],
        )

    def _set_identification_images(self):
        """
        Creates an square image that we call "identification_image". This
        image is used both to classify the blob as crossing or individual
        and to identify the animals later on in the tracking.
        The length of the diagonal of the identification_image equals the
        medial_body_length
        """
        logger.info("--> set_identification_images")
        self.video.compute_identification_image_size(self.median_body_length)
        self.list_of_blobs.set_images_for_identification(
            self.video.episodes_start_end,
            self.video.identification_images_file_paths,
            self.video.identification_image_size,
            self.video.user_defined_parameters["number_of_animals"],
            self.video.number_of_frames,
            self.video.video_path,
            self.video.height,
            self.video.width,
        )

    def _connect_list_of_blobs(self):
        """
        Connects all consecutive blobs in the video based on the overlapping
        of the pixels
        """
        logger.info(
            "--> connect_list_of_blobs "
            "(crossing detector overlapping heuristic)"
        )
        if not self.list_of_blobs.blobs_are_connected:
            self.list_of_blobs.compute_overlapping_between_subsequent_frames()

    def _train_and_apply_crossing_detector(self):
        """
        Detects all blobs in the video as crossings or individuals
        """
        if self.video.user_defined_parameters["number_of_animals"] > 1:
            detect_crossings(
                self.list_of_blobs,
                self.video,
                self.model_area,
            )
        else:
            self.video._there_are_crossings = False
