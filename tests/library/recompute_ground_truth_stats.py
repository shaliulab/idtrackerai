from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('./')
sys.path.append('./preprocessing')
sys.path.append('./utils')
sys.path.append('./groundtruth_utils')
import numpy as np
import logging
from pprint import pprint
from tqdm import tqdm
import pandas as pd

from idtrackerai.utils.GUI_utils import selectDir
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.blob import Blob
from idtrackerai.groundtruth_utils.generate_groundtruth import GroundTruth, GroundTruthBlob, generate_groundtruth
from idtrackerai.groundtruth_utils.compute_groundtruth_statistics import get_accuracy_wrt_groundtruth

from library_utils import Dataset, BlobsListConfig, subsample_dataset_by_individuals, generate_list_of_blobs, LibraryJobConfig, check_if_repetition_has_been_computed

if __name__ == '__main__':


    '''
    argv[1]: 1 = cluster, 0 = no cluster
    argv[2]: test_number

    e.g.
    run_library_tests.py 1 1 P None 0 .5 .1 DEF afs 1_2 (running in the cluster, job1, pretraining, libraries DEF, all individuals in library D and first half obf E second half of F, repetitions[1 2])
    '''
    print('\n\n ********************************************* \n\n')
    print("cluster:", sys.argv[1])
    print("test_number:", sys.argv[2])

    tests_data_frame = pd.read_pickle('./library/tests_data_frame.pkl')
    test_dictionary = tests_data_frame.loc[int(sys.argv[2])].to_dict()
    pprint(test_dictionary)

    job_config = LibraryJobConfig(cluster = sys.argv[1], test_dictionary = test_dictionary)
    job_config.condition_path = os.path.join('./library','library_test_' + test_dictionary['test_name'])

    for repetition in job_config.repetitions:

        for group_size in job_config.group_sizes:

            for frames_in_video in job_config.frames_in_video:

                for scale_parameter in job_config.scale_parameter:

                    for shape_parameter in job_config.shape_parameter:

                        repetition_path = os.path.join(job_config.condition_path,'group_size_' + str(group_size),
                                                                'num_frames_' + str(frames_in_video),
                                                                'scale_parameter_' + str(scale_parameter),
                                                                'shape_parameter_' + str(shape_parameter),
                                                                'repetition_' + str(repetition))
                        print("\n********** group size %i - frames_in_video %i - scale_parameter %s -  shape_parameter %s - repetition %i ********"
                                %(group_size, frames_in_video, str(scale_parameter), str(shape_parameter), repetition))

                        session_path = os.path.join(repetition_path, 'session')
                        video_object_path = os.path.join(session_path,'video_object.npy')
                        print("loading video object...")
                        try:
                            video = np.load(video_object_path).item(0)
                            print("loading list_of_blobs...")
                            list_of_blobs = np.load(video.blobs_path).item()
                            print("loading list_of_fragments...")
                            list_of_fragments_dicts = np.load(os.path.join(video.accumulation_folder, 'light_list_of_fragments.npy'))
                            for blobs_in_frame in tqdm(list_of_blobs.blobs_in_video, desc = 'updating list of blobs from list of fragments'):
                                for blob in blobs_in_frame:
                                    fragment_dict = list_of_fragments_dicts[video.fragment_identifier_to_index[blob.fragment_identifier]]
                                    blob._P2_vector = fragment_dict['_P2_vector']

                            print("loading groundtruth")
                            groundtruth_path = os.path.join(video.video_folder,'_groundtruth.npy')
                            groundtruth = np.load(groundtruth_path).item()
                            blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
                            blobs_in_video = list_of_blobs.blobs_in_video[groundtruth.start:groundtruth.end]

                            print("computting groundtrugh")
                            accuracies, _ = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth,
                                                                            blobs_in_video,
                                                                            video.first_frame_first_global_fragment[video.accumulation_trial])

                            if accuracies is not None:
                                print("saving accuracies in video")
                                video.gt_start_end = (groundtruth.start,groundtruth.end)
                                video.gt_accuracy = accuracies
                            video.save()

                        except Exception as e:
                            print(e)
                            print("No video object")
