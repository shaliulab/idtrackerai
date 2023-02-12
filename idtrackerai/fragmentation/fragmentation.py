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
import os.path
import logging
import warnings
import time
import numpy as np
from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import (
    ListOfFragments,
    create_list_of_fragments,
)

from imgstore.stores.utils.mixins.extract import _extract_store_metadata

logger = logging.getLogger(__name__)

def show_fragment_structure(chunk, min_duration=None, list_of_fragments=None):
    """
    Get a detailed overview of the fragmentation results for a given chunk
    
    Args:
        chunk (int): Chunk number of an imgstore dataset
        min_duration (int): Minimal duration of a fragment for it to be considered significant (seconds)
        list_of_fragments (idtrackerai.list_of_fragments.ListOfFragments): Optional, list of fragments instance corresponding to the requested chunk

    Return:
        structure (list) For every fragment, the following information:
            * start_end: first and last frame number where the fragment is present. last frame follows Python convention (= actually first frame after the fragment is over)
            * length (length of the fragment (i.e. the diff of start_end)
            * whether the fragment is longer than the min_duration
            * the identity of the fragment (fragment identifier, not to be confused with blob identity)
            * annotation. One of FOLLOWED or FINISH. FOLLOWED means there should be another fragment after this for the same animal. FINISH otherwise

    Note: annotation is used in idtrackerai_app.cli.annotation.annotation to decide where does each scene start/end
    """


    assert min_duration is not None
    video_file = os.path.join(f"session_{str(chunk).zfill(6)}", "video_object.npy")
    video_object = np.load(video_file, allow_pickle=True).item()

    if list_of_fragments is None:
        fragment_file = os.path.join(f"session_{str(chunk).zfill(6)}", "preprocessing", "fragments.npy")
        assert os.path.exists(fragment_file), f"{fragment_file} not found"
        list_of_fragments = np.load(fragment_file, allow_pickle=True).item()

    store_md = os.path.realpath(video_object.video_path)
    assert os.path.exists(store_md), f"Path to metadata.yaml ({store_md}) not found"
    metadata = _extract_store_metadata(store_md)
    framerate = metadata["framerate"]

    structure = []

    for fragment in list_of_fragments.fragments:
        length = fragment.start_end[1] - fragment.start_end[0]
        followed = fragment.start_end[1] !=  video_object.episodes_start_end[-1][-1]
        if followed:
            followed_str="FOLLOWED"
        else:
            followed_str="FINISH"

        structure.append((fragment.start_end, length, length > (framerate * min_duration), fragment.identity, followed_str))
        print("FRAGMENT STRUCTURE: ", end = "")
        for field in structure[-1]:
            print(field, end=" ")
        print("")

    return structure


class FragmentationABC(ABC):
    def __init__(self, video: Video, list_of_blobs: ListOfBlobs, use_fragment_transfer_info: bool = False, threshold: int =None):
        """
        Generates a list_of_fragments given a video and a list_of_blobs
        """
        self.video = video
        self.list_of_blobs = list_of_blobs
        self._use_fragment_transfer_info=use_fragment_transfer_info
        self._threshold = threshold

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
        if self.list_of_blobs.blobs_are_connected:
            # If the list of of blobs has been loaded
            self.list_of_blobs.disconnect()
            self.list_of_blobs.reconnect_from_cache()
        else:
            self.list_of_blobs.compute_overlapping_between_subsequent_frames(
                use_fragment_transfer_info=self._use_fragment_transfer_info,
            )

            

        for blobs_in_frame in self.list_of_blobs.blobs_in_video:
            for blob in blobs_in_frame:
                blob._fragment_identifier = None
                blob._blob_index = None

        self.list_of_blobs.compute_fragment_identifier_and_blob_index(
            max(
                self.video.user_defined_parameters["number_of_animals"],
                self.video.maximum_number_of_blobs,
            )
        )
        self.list_of_blobs.compute_crossing_fragment_identifier()

        f_ids = set()
        for blobs_in_frame in self.list_of_blobs.blobs_in_video:
            for blob in blobs_in_frame:
                f_ids.add(blob.fragment_identifier)

        n_fragments = len(f_ids)
        if n_fragments > 1000:
            warnings.warn(f"{n_fragments} fragments detected. Is that right?")
            import ipdb; ipdb.set_trace()

        fragments = create_list_of_fragments(
            self.list_of_blobs.blobs_in_video,
            self.video.user_defined_parameters["number_of_animals"],
        )

        # List of fragments
        list_of_fragments = ListOfFragments(
            fragments,
            self.video.identification_images_file_paths,
        )

        chunk = getattr(self.video, "_chunk", None)
        if chunk is not None:
            show_fragment_structure(chunk=chunk, min_duration=1, list_of_fragments=list_of_fragments)
        
        self._update_video_object_with_fragments(list_of_fragments)
        return list_of_fragments

    def _update_video_object_with_fragments(self, list_of_fragments):
        self.video._fragment_identifier_to_index = (
            list_of_fragments.get_fragment_identifier_to_index_list()
        )
