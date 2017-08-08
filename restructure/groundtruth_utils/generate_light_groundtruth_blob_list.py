from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from blob import ListOfBlobs, Blob
from GUI_utils import selectDir, getInput

class GroundTruthBlob(object):
    """Lighter blob objects.
    Attributes:
        identity (preferring the one assigned by the user, if it is not None)
        centroid
        pixels (pixels is stored to check the groundtruth in crossings)
    """
    def __init__(self, attributes_to_get = ['identity', 'centroid', 'pixels', 'frame_number']):
        self.attributes = attributes_to_get

    def get_attribute(self, blob):
        for attribute in self.attributes:
            if attribute == 'identity':
                self.identity = blob.identity if blob.user_generated_identity == None else blob.user_generated_identity
            else:
                setattr(self, attribute, getattr(blob, attribute))

class GroundTruth(object):
    def __init__(self, video_object = [], list_of_blobs = [], count_number_assignment_per_individual_assigned = {}, count_number_assignment_per_individual_all = {}, start = None, end = None):
        self.video_object = video_object
        self.list_of_blobs = list_of_blobs
        self.count_number_assignment_per_individual_assigned = count_number_assignment_per_individual_assigned
        self.count_number_assignment_per_individual_all = count_number_assignment_per_individual_all
        self.start = start
        self.end = end

    def save(self):
        path_to_save_groundtruth = os.path.join(os.path.split(self.video_object.video_path)[0], '_groundtruth.npy')
        print("saving groundtruth at ", path_to_save_groundtruth)
        np.save(path_to_save_groundtruth, self)

def generate_groundtruth_files(video_object, start = None, end = None):
    """Generates a list of light blobs, given a video object corresponding to a
    tracked video
    """
    #make sure the video has been succesfully tracked
    assert video_object._has_been_assigned == True
    #read blob list from video
    blobs_list = ListOfBlobs.load(video_object.blobs_path)
    blobs = blobs_list.blobs_in_video
    count_number_assignment_per_individual_assigned = {i: 0 for i in range(1,video_object.number_of_animals+1)}
    count_number_assignment_per_individual_all = {i: 0 for i in range(1,video_object.number_of_animals+1)}
    #init groundtruth blobs list
    groundtruth_blobs_list = []
    for blobs_in_frame in blobs:
        groundtruth_blobs_in_frame = []
        for blob in blobs_in_frame:
            gt_blob = GroundTruthBlob()
            gt_blob.get_attribute(blob)
            groundtruth_blobs_in_frame.append(gt_blob)
            if (blob.is_a_fish_in_a_fragment or\
                    blob.is_a_jump or\
                    blob.is_a_jumping_fragment or\
                    hasattr(blob,'is_an_extreme_of_individual_fragment')) and\
                    blob.user_generated_identity != -1: # we are not considering crossing or failures of the model area
                if blob.user_generated_identity is not None and blob.user_generated_identity != blob.identity:
                    count_number_assignment_per_individual_all[blob.user_generated_identity] += 1
                    if blob.identity != 0:
                        count_number_assignment_per_individual_assigned[blob.user_generated_identity] += 1
                elif blob.identity != 0:
                    count_number_assignment_per_individual_all[blob.identity] += 1
                    count_number_assignment_per_individual_assigned[blob.identity] += 1

        groundtruth_blobs_list.append(groundtruth_blobs_in_frame)
    groundtruth = GroundTruth(video_object = video_object,
                            list_of_blobs = groundtruth_blobs_list,
                            count_number_assignment_per_individual_assigned = count_number_assignment_per_individual_assigned,
                            count_number_assignment_per_individual_all = count_number_assignment_per_individual_all,
                            start = start,
                            end = end)
    groundtruth.save()



if __name__ == "__main__":

    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item()
    start = getInput('GroundTruth (start)', 'Input the starting frame for the interval in which the video has been validated')
    end = getInput('GroundTruth (end)', 'Input the ending frame for the interval in which the video has been validated')
    generate_groundtruth_files(video, int(start), int(end))
