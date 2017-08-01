from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from blob import ListOfBlobs

class GroundTruthBlob(object):
    """Lighter blob objects.
    Attributes:
        identity (preferring the one assigned by the user, if it is not None)
        centroid
        pixels (pixels is stored to check the groundtruth in crossings)
    """
    def __init__(self, attributes_to_get = ['identity', 'centroid', 'pixels']):
        self.attributes = attributes_to_get

    def get_attribute(self, blob):
        for attribute in self.attributes:
            if attribute == 'identity':
                self.identity = blob.identity if blob.user_generated_identity == None else blob.user_generated_identity
            else:
                setattr(self, attribute, getattr(blob, attribute))

class GroundTruth(object):
    def __init__(self, video_object = [], list_of_blobs = [], count_unoccluded_individual_assignments = {}, count_crossing_individual_assignments = {}):
        self.video_object = video_object
        self.list_of_blobs = list_of_blobs
        self.unoccluded_individual_assignments = count_unoccluded_individual_assignments
        self.crossing_individual_assignments = count_crossing_individual_assignments

    def save(self):
        path_to_save_groundtruth = os.path.join(os.path.split(self.video_object.video_path)[0], '_groundtruth.npy')
        print("saving groundtruth at ", path_to_save_groundtruth)
        np.save(path_to_save_groundtruth, self)

def generate_groundtruth_files(video_object):
    """Generates a list of light blobs, given a video object corresponding to a
    tracked video
    """
    #make sure the video has been succesfully tracked
    assert video_object._has_been_assigned == True
    #read blob list from video
    blobs_list = ListOfBlobs.load(video_object.blobs_path)
    blobs = blobs_list.blobs_in_video
    #init groundtruth blobs list
    groundtruth_blobs_list = []
    #count number of assignment per individual in groundtruth
    count_number_assignment_per_individual_no_crossing = {i: 0 for i in range(1, video_object.number_of_animals + 1)}
    #XXX here we include, but it has to be removed
    count_number_assignment_per_individual_during_crossing = {i: 0 for i in range(video_object.number_of_animals + 1)}

    for blobs_in_frame in blobs:

        for blob in blobs_in_frame:
            gt_blob = GroundTruthBlob()
            gt_blob.get_attribute(blob)
            groundtruth_blobs_list.append(gt_blob)
            if not blob.is_a_crossing:
                if blob.user_generated_identity is not None and blob.user_generated_identity != blob.identity:
                    count_number_assignment_per_individual_no_crossing[blob.user_generated_identity] += 1
                else:
                    count_number_assignment_per_individual_no_crossing[blob.identity] += 1
            else:
                if blob.user_generated_identity is not None and blob.user_generated_identity != blob.identity:
                    count_number_assignment_per_individual_during_crossing[blob.user_generated_identity] += 1
                else:
                    count_number_assignment_per_individual_during_crossing[blob.identity] += 1

    groundtruth = GroundTruth(video_object = video_object,
                            list_of_blobs = groundtruth_blobs_list,
                            count_unoccluded_individual_assignments = count_number_assignment_per_individual_no_crossing,
                            count_crossing_individual_assignments = count_number_assignment_per_individual_during_crossing)
    groundtruth.save()




if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/video_object.npy').item()
    generate_groundtruth_files(video)
