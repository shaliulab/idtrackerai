# This file is part of idtracker.ai, a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Bergomi, M.G., Romero-Ferrero, F., Heras, F.J.H.
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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals with correct identities in large
# animal collectives (submitted)

from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('../../plots')
import pandas
import numpy as np
from pprint import pprint
import pandas as pd
from glob import glob
from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments
from idtrackerai.groundtruth_utils.generate_groundtruth import GroundTruthBlob, GroundTruth
from idtrackerai.groundtruth_utils.generate_individual_groundtruth import GroundTruthBlob, IndividualGroundTruth, generate_individual_groundtruth
from idtrackerai.groundtruth_utils.compute_groundtruth_statistics import get_accuracy_wrt_groundtruth
from idtrackerai.groundtruth_utils.compute_individual_groundtruth_statistics import get_individual_accuracy_wrt_groundtruth
from idtrackerai.postprocessing.identify_non_assigned_with_interpolation import assign_zeros_with_interpolation_identities
from global_fragments_statistics import compute_and_plot_fragments_statistics

def correct_blob_in_list_of_blobs(gt_blob,list_of_blobs):
    blobs_in_frame = list_of_blobs.blobs_in_video[gt_blob.frame_number]
    blob_to_correct = [blob for blob in blobs_in_frame if blob.blob_index == gt_blob.blob_index]
    assert len(blob_to_correct) == 1
    blob_to_correct = blob_to_correct[0]
    if blob_to_correct.assigned_identity != gt_blob.identity:
        print("correcting identity")
        blob_to_correct._user_generated_identity = gt_blob.identity

def correct_blobs_in_frame_in_list_of_blobs(gt_blobs_in_frame, list_of_blobs):
    blobs_in_frame = list_of_blobs.blobs_in_video[frame_number]
    for gt_blob, blob in zip(blobs_in_frame, gt_blobs_in_frame):
        if blob.assigned_identity != gt_blob.identity:
            print("correcting identity")
            blob._user_generated_identity = gt_blob.identity


if __name__ == '__main__':
    from idtrackerai.utils.GUI_utils import selectDir
    ''' select blobs_in_video list tracked to compare against ground truth '''
    session_path = selectDir('./') #select path to video
    video_object_path = os.path.join(session_path,'video_object.npy')
    print("loading video object")
    video = np.load(video_object_path).item(0)
    print("loading list_of_blobs")
    list_of_blobs = ListOfBlobs.load(video, os.path.join(session_path, 'preprocessing', 'blobs_collection.npy'))
    print("loading list_of_blobs_interpolated")
    list_of_blobs_interpolated = ListOfBlobs.load(video, os.path.join(session_path, 'preprocessing', 'blobs_collection_interpolated.npy'))

    ### Correct list_of_blobs from individual groundtruth
    print("correcting from individual groundtruths")
    individual_groundtruth_paths = glob(os.path.join(video.video_folder,'_individual*.npy'))
    for individual_groundtruth_path in individual_groundtruth_paths:
        individual_groundtruth = np.load(individual_groundtruth_path).item()

        for gt_blob in individual_groundtruth.individual_blobs_in_video:
            correct_blob_in_list_of_blobs(gt_blob, list_of_blobs)
            correct_blob_in_list_of_blobs(gt_blob, list_of_blobs_interpolated)

    ### Correct list_of_blobs from global groundtruth
    print("correcting from global groundtruths")
    ground_truth_path = os.path.join(video.video_folder,'_groundtruth.npy')
    groundtruth = np.load(ground_truth_path).item()
    for frame_number in range(groundtruth.start, groundtruth.end + 1):
        gt_blobs_in_frame = groundtruth.blobs_in_video[frame_number]
        correct_blobs_in_frame_in_list_of_blobs(gt_blobs_in_frame, list_of_blobs)
        correct_blobs_in_frame_in_list_of_blobs(gt_blobs_in_frame, list_of_blobs_interpolated)

    list_of_blobs.save(video, os.path.join(session_path, 'preprocessing', 'blobs_collection.npy'), number_of_chunks = video.number_of_frames)
    list_of_blobs_interpolated.save(video, os.path.join(session_path, 'preprocessing', 'blobs_collection_interpolated.npy'), number_of_chunks = video.number_of_frames)
