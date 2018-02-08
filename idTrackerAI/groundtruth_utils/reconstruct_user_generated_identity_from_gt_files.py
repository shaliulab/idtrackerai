from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./groundtruth_utils')
sys.path.append('./postprocessing')
sys.path.append('./plots')
sys.path.append('./utils')
sys.path.append('./network/identification_model')

import pandas
import numpy as np
from pprint import pprint
import pandas as pd
from glob import glob

from video import Video
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments
from generate_groundtruth import GroundTruthBlob, GroundTruth
from generate_individual_groundtruth import GroundTruthBlob, IndividualGroundTruth, generate_individual_groundtruth
from compute_groundtruth_statistics import get_accuracy_wrt_groundtruth
from compute_individual_groundtruth_statistics import get_individual_accuracy_wrt_groundtruth
from identify_non_assigned_with_interpolation import assign_zeros_with_interpolation_identities
from global_fragments_statistics import compute_and_plot_fragments_statistics
from GUI_utils import selectDir

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
