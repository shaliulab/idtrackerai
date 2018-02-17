from __future__ import absolute_import, division, print_function
import sys
import copy
import numpy as np
from idtrackerai.list_of_blobs import ListOfBlobs
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.identify_non_assigned_with_interpolation")

def assign_zeros_with_interpolation_identities(list_of_blobs, list_of_blobs_no_gaps):
    logger.debug("creating copy of list_of_blobs")

    for blobs_in_frame, blobs_in_frame_no_gaps in zip(list_of_blobs.blobs_in_video, list_of_blobs_no_gaps.blobs_in_video):
        unassigned_blobs = [blob for blob in blobs_in_frame if blob.assigned_identity == 0]
        for unassigned_blob in unassigned_blobs:
            candidate_blobs = [blob for blob in blobs_in_frame_no_gaps
                                    if blob.fragment_identifier == unassigned_blob.fragment_identifier]
            if len(candidate_blobs) == 1 \
                and isinstance(candidate_blobs[0].assigned_identity, list) \
                and len(candidate_blobs[0].assigned_identity) == 1:
                unassigned_blob._identity_corrected_closing_gaps = candidate_blobs[0].assigned_identity[0]


    return list_of_blobs

if __name__ == '__main__':
    import os
    from idtrackerai.utils.GUI_utils import selectDir
    from idtrackerai.video import Video
    from idtrackerai.groundtruth_utils.generate_groundtruth import GroundTruthBlob, GroundTruth
    from idtrackerai.groundtruth_utils.compute_groundtruth_statistics import get_accuracy_wrt_groundtruth

    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    logger.debug("loading list_of_blobs")
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    logger.debug("loading list_of_blobs_no_gaps")
    list_of_blobs_no_gaps = ListOfBlobs.load(video, video.blobs_no_gaps_path)
    logger.debug("creating list_of_blobs_interpolated")
    video._blobs_path_interpolated = os.path.join(video.preprocessing_folder, 'blobs_collection_interpolated.npy')
    list_of_blobs_interpolated = assign_zeros_with_interpolation_identities(list_of_blobs, list_of_blobs_no_gaps)
    list_of_blobs_interpolated.save(video, video._blobs_path_interpolated, number_of_chunks = video.number_of_frames)
    logger.debug("loading ground truth file")
    groundtruth = np.load(os.path.join(video.video_folder, '_groundtruth.npy')).item()
    blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
    blobs_in_video_interpolated = list_of_blobs_interpolated.blobs_in_video[groundtruth.start:groundtruth.end]
    logger.debug("computing groundtruth")
    accuracies, results = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth, blobs_in_video_interpolated)
    logger.debug("saving video")
    video.gt_accuracy_interpolated = accuracies
    video.gt_results_interpolated = results
    video.save()
