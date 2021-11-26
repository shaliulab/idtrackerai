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


class GroundTruthBlob(object):
    """Lighter blob objects.
    Attributes:
        identity (preferring the one assigned by the user, if it is not None)
        centroid
        pixels (pixels is stored to check the groundtruth in crossings)
    """

    def __init__(
        self,
        attributes_to_get=[
            "identity",
            "assigned_identity",
            "used_for_training",
            "accumulation_step",
            "centroid",
            "pixels",
            "frame_number",
            "is_an_individual",
            "is_a_crossing",
            "blob_index",
            "fragment_identifier",
        ],
    ):
        self.attributes = attributes_to_get

    def get_attribute(self, blob):
        for attribute in self.attributes:
            if attribute == "identity":
                setattr(self, attribute, getattr(blob, "final_identity"))
            else:
                setattr(self, attribute, getattr(blob, attribute))


class IndividualGroundTruth(object):
    def __init__(
        self,
        video=[],
        individual_blobs_in_video=[],
        start=None,
        end=None,
        validated_identity=None,
    ):
        self.video = video
        self.individual_blobs_in_video = individual_blobs_in_video
        self.start = start
        self.end = end
        self.validated_identity = validated_identity

    def save(self):
        gt_name = (
            "_individual_" + str(self.validated_identity) + "_groundtruth.npy"
        )
        path_to_save_groundtruth = os.path.join(
            os.path.split(self.video.video_path)[0], gt_name
        )
        logger.info("saving ground truth at %s" % path_to_save_groundtruth)
        np.save(path_to_save_groundtruth, self)
        logger.info("done")


def generate_individual_groundtruth(
    video,
    blobs_in_video=None,
    start=None,
    end=None,
    validated_identity=None,
    save_gt=True,
):
    """Generates a list of light blobs_in_video, given a video object corresponding to a
    tracked video
    """
    individual_blobs_in_video_groundtruth = []

    for blobs_in_frame in blobs_in_video:
        identities_in_frame = set(
            [blob.final_identity for blob in blobs_in_frame]
        )
        for blob in blobs_in_frame:
            if blob.final_identity == validated_identity:
                gt_blob = GroundTruthBlob()
                gt_blob.get_attribute(blob)
                individual_blobs_in_video_groundtruth.append(gt_blob)

    groundtruth = IndividualGroundTruth(
        video=video,
        individual_blobs_in_video=individual_blobs_in_video_groundtruth,
        start=start,
        end=end,
        validated_identity=validated_identity,
    )
    if save_gt:
        groundtruth.save()
    return groundtruth
