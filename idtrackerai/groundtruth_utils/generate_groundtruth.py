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

import numpy as np

from idtrackerai.blob import Blob
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments

logger = logging.getLogger("__main__.generate_light_groundtruth_blob_list")

ATTRIBUTES_TO_COPY = (
    "identities",
    "centroids",
    "assigned_identities",
    "assigned_centroids",
    "used_for_training",
    "accumulation_step",
    "centroid",
    "pixels",
    "frame_number",
    "is_an_individual",
    "is_a_crossing",
    "was_a_crossing",
    "blob_index",
    "fragment_identifier",
)


class GroundTruthBlob(object):
    """Lighter blob objects.
    Attributes:
        identity (preferring the one assigned by the user, if it is not None)
        centroid
        pixels (pixels is stored to check the groundtruth in crossings)
    """

    def __init__(self, blob):
        self.attributes = ATTRIBUTES_TO_COPY
        self.set_attributes(blob)

    def set_attributes(self, blob):
        for attribute in self.attributes:
            if attribute == "identities":
                setattr(self, attribute, getattr(blob, "final_identities"))
            elif attribute == "centroids":
                setattr(self, attribute, getattr(blob, "final_centroids"))
            else:
                setattr(self, attribute, getattr(blob, attribute))

    @property
    def gt_identity(self):
        if hasattr(self, "identity"):
            return self.identity
        else:
            if len(self.identities) == 1:
                return self.identities[0]
            else:
                return -1


class GroundTruth(object):
    def __init__(self, video, blobs_in_video, start, end):
        self.video = video
        self.blobs_in_video = blobs_in_video
        self.start = start
        self.end = end

    def save(self, name=""):
        if name == "":
            path_to_save_groundtruth = os.path.join(
                os.path.split(self.video.video_path)[0], "_groundtruth.npy"
            )
        else:
            path_to_save_groundtruth = os.path.join(
                os.path.split(self.video.video_path)[0],
                "_groundtruth_" + name + ".npy",
            )
        logger.info("saving ground truth at %s" % path_to_save_groundtruth)
        np.save(path_to_save_groundtruth, self)
        logger.info("done")


def generate_groundtruth(
    video,
    blobs_in_video=None,
    start=None,
    end=None,
    wrong_crossing_counter=None,
    unidentified_individuals_counter=None,
    save_gt=True,
):
    """Generates a list of light blobs_in_video, given a video object corresponding to a
    tracked video
    """
    logger.info("Generating ground truth file")
    # make sure the video has been succesfully tracked
    blobs_in_video_groundtruth = []

    for blobs_in_frame in blobs_in_video:
        blobs_in_frame_groundtruth = []

        for blob in blobs_in_frame:
            gt_blob = GroundTruthBlob(blob)
            blobs_in_frame_groundtruth.append(gt_blob)

        blobs_in_video_groundtruth.append(blobs_in_frame_groundtruth)

    groundtruth = GroundTruth(
        video=video,
        blobs_in_video=blobs_in_video_groundtruth,
        start=start,
        end=end,
    )
    groundtruth.wrong_crossing_counter = wrong_crossing_counter
    groundtruth.unidentified_individuals_counter = (
        unidentified_individuals_counter
    )
    if save_gt:
        groundtruth.save()
    return groundtruth


if __name__ == "__main__":

    session_path = sys.argv[1]  # select path to video
    video_path = os.path.join(session_path, "video_object.npy")
    video = np.load(video_path, allow_pickle=True).item()
    start = input(
        "GroundTruth (start)"
        "Input the starting frame for the interval "
        "in which the video has been validated",
    )
    end = input(
        "GroundTruth (end)"
        "Input the ending frame for the interval "
        "in which the video has been validated",
    )
    # read blob list from video
    fragments_path = os.path.join(
        session_path, "preprocessing", "fragments.npy"
    )
    blobs_path = os.path.join(
        session_path, "preprocessing", "blobs_collection.npy"
    )
    list_of_fragments = ListOfFragments.load(fragments_path)
    list_of_blobs = ListOfBlobs.load(video, blobs_path)
    list_of_blobs.update_from_list_of_fragments(
        list_of_fragments.fragments, video.fragment_identifier_to_index
    )
    groundtruth = generate_groundtruth(
        video, list_of_blobs.blobs_in_video, int(start), int(end)
    )
