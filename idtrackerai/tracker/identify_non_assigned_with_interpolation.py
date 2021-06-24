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

from __future__ import absolute_import, division, print_function

import copy
import logging
import sys

import numpy as np

from idtrackerai.list_of_blobs import ListOfBlobs

logger = logging.getLogger("__main__.identify_non_assigned_with_interpolation")


def assign_zeros_with_interpolation_identities(
    list_of_blobs, list_of_blobs_no_gaps
):
    logger.debug("creating copy of list_of_blobs")

    for blobs_in_frame, blobs_in_frame_no_gaps in zip(
        list_of_blobs.blobs_in_video, list_of_blobs_no_gaps.blobs_in_video
    ):
        unassigned_blobs = [
            blob
            for blob in blobs_in_frame
            if blob.is_an_individual and blob.assigned_identities[0] == 0
        ]
        for unassigned_blob in unassigned_blobs:
            candidate_blobs = [
                blob
                for blob in blobs_in_frame_no_gaps
                if blob.fragment_identifier
                == unassigned_blob.fragment_identifier
            ]
            if (
                len(candidate_blobs) == 1
                and len(candidate_blobs[0].assigned_identities) == 1
            ):
                unassigned_blob._identities_corrected_closing_gaps = (
                    candidate_blobs[0].assigned_identities
                )

    return list_of_blobs
