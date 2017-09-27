from __future__ import absolute_import, division, print_function
import os
import sys
from glob import glob
sys.path.append('./')

if __name__ == '__main__':

    ''' select blobs list tracked to compare against ground truth '''
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)

    ''' select ground truth file '''
    groundtruth_path = os.path.join(video._video_folder,'_groundtruth.npy')
    groundtruth = np.load(groundtruth_path).item()
    groundtruth.list_of_blobs = groundtruth.list_of_blobs[groundtruth.start:groundtruth.end]
    blobs = blobs[groundtruth.start:groundtruth.end]

    accuracy, individual_accuracy, accuracy_assigned, individual_accuracy_assigned = get_statistics_against_groundtruth(groundtruth, blobs)
