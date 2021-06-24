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
from idtrackerai.list_of_fragments import (
    ListOfFragments,
    create_list_of_fragments,
)

logger = logging.getLogger(__name__)


class FragmentationABC(ABC):
    def __init__(self, video: Video, list_of_blobs: ListOfBlobs):
        """
        Generates a list_of_fragments given a video and a list_of_blobs
        """
        self.video = video
        self.list_of_blobs = list_of_blobs

    def __call__(self):
        self.video._fragmentation_time = time.time()
        self.list_of_fragments = self.fragment_video()
        self.video._fragmentation_time = (
            time.time() - self.video.fragmentation_time
        )
        self.video._has_been_fragmented = True
        return self.list_of_fragments

    @abstractmethod
    def fragment_video(self):
        pass


class FragmentationAPI(FragmentationABC):
    def fragment_video(self):
        if self.video.user_defined_parameters["number_of_animals"] != 1:
            return self._generate_list_of_fragments()
        else:
            # If there is only one animal there is no need to compute fragments
            # as the trajectories are obtained directly from the list_of_blobs
            return None

    def _generate_list_of_fragments(self):
        if not self.list_of_blobs.blobs_are_connected:
            # If the list of of blobs has been loaded
            self.list_of_blobs.compute_overlapping_between_subsequent_frames()

        self.list_of_blobs.compute_fragment_identifier_and_blob_index(
            max(
                self.video.user_defined_parameters["number_of_animals"],
                self.video.maximum_number_of_blobs,
            )
        )
        self.list_of_blobs.compute_crossing_fragment_identifier()
        fragments = create_list_of_fragments(
            self.list_of_blobs.blobs_in_video,
            self.video.user_defined_parameters["number_of_animals"],
        )

        # List of fragments
        list_of_fragments = ListOfFragments(
            fragments,
            self.video.identification_images_file_paths,
        )
        self._update_video_object_with_fragments(list_of_fragments)
        return list_of_fragments

    def _update_video_object_with_fragments(self, list_of_fragments):
        self.video._fragment_identifier_to_index = (
            list_of_fragments.get_fragment_identifier_to_index_list()
        )
