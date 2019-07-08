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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import sys
import numpy as np
from pprint import pprint
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.blob import Blob
from idtrackerai.groundtruth_utils.generate_individual_groundtruth import IndividualGroundTruth, GroundTruthBlob
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.compute_statistics_against_groundtruth")


def compare_tracked_individual_against_groundtruth(blobs_in_individual_groundtruth,
                                                    individual_blobs, individual_groundtruth_id):
    comparison_keys = ['accuracy', 'frames_with_errors', 'mistaken_identities']
    comparison_info = {key: [] for key in comparison_keys}

    for blob_gt, blob in zip(blobs_in_individual_groundtruth, individual_blobs):
        if blob_gt.identity != blob.assigned_identity:
            comparison_info['frames_with_errors'].append(blob.frame_number)
            comparison_info['mistaken_identities'].append(blob.assigned_identity)

    comparison_info['accuracy'] = 1 -len(comparison_info['mistaken_identities']) / len(blobs_in_individual_groundtruth)
    number_of_assigned_blobs = len([blob for blob in blobs_in_individual_groundtruth if blob.assigned_identity != 0])
    number_of_mistaken_identified_blobs = len([identity for identity in comparison_info['mistaken_identities'] if identity != 0])
    comparison_info['accuracy_assigned'] = 1 - number_of_mistaken_identified_blobs / number_of_assigned_blobs
    comparison_info['id'] = individual_groundtruth_id
    return comparison_info

def check_groundtruth_consistency(blobs_in_individual_groundtruth,
                                    individual_groundtruth_id, individual_blobs,
                                    individual_id):
    non_matching_error = "The length of the collections of the ground truth individual and the selected one do not match"
    if individual_groundtruth_id != individual_id or\
        len(blobs_in_individual_groundtruth) != len(individual_blobs):
        raise ValueError(non_matching_error)

def get_individual_accuracy_wrt_groundtruth(video, blobs_in_individual_groundtruth, individual_blobs = None):
    individual_groundtruth_id = blobs_in_individual_groundtruth[0].identity
    individual_id = individual_groundtruth_id
    if individual_blobs is None:
        individual_blobs = blobs_in_individual_groundtruth
    else:
        check_groundtruth_consistency(blobs_in_individual_groundtruth,
                                    individual_groundtruth_id,
                                    individual_blobs,
                                    individual_id)
    return compare_tracked_individual_against_groundtruth(blobs_in_individual_groundtruth,
                                                individual_blobs, individual_groundtruth_id)
