from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from blob import Blob
from GUI_utils import selectDir, getInput

if sys.argv[0] == 'idtrackeraiApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.generate_light_groundtruth_blob_list")

class GroundTruthBlob(object):
    """Lighter blob objects.
    Attributes:
        identity (preferring the one assigned by the user, if it is not None)
        centroid
        pixels (pixels is stored to check the groundtruth in crossings)
    """
    def __init__(self, attributes_to_get = ['identity', 'assigned_identity',
                                            'used_for_training', 'accumulation_step',
                                            'centroid', 'pixels',
                                            'frame_number',
                                            'is_an_individual', 'is_a_crossing',
                                            'was_a_crossing',
                                            'blob_index', 'fragment_identifier']):
        self.attributes = attributes_to_get

    def get_attribute(self, blob):
        for attribute in self.attributes:
            if attribute == 'identity':
                setattr(self, attribute, getattr(blob, 'final_identity'))
            else:
                setattr(self, attribute, getattr(blob, attribute))

class GroundTruth(object):
    def __init__(self, video = [], blobs_in_video = [], start = None, end = None):
        self.video = video
        self.blobs_in_video = blobs_in_video
        self.start = start
        self.end = end

    def save(self, name = ''):
        if name == '':
            path_to_save_groundtruth = os.path.join(os.path.split(self.video.video_path)[0], '_groundtruth.npy')
        else:
            path_to_save_groundtruth = os.path.join(os.path.split(self.video.video_path)[0], '_groundtruth_' + name + '.npy')
        logger.info("saving ground truth at %s" %path_to_save_groundtruth)
        np.save(path_to_save_groundtruth, self)
        logger.info("done")

def generate_groundtruth(video, blobs_in_video = None, start = None, end = None,
                        wrong_crossing_counter = None, unidentified_individuals_counter = None,
                        save_gt = True):
    """Generates a list of light blobs_in_video, given a video object corresponding to a
    tracked video
    """
    #make sure the video has been succesfully tracked
    assert video.has_been_assigned == True
    blobs_in_video_groundtruth = []

    for blobs_in_frame in blobs_in_video:
        blobs_in_frame_groundtruth = []

        for blob in blobs_in_frame:
            gt_blob = GroundTruthBlob()
            gt_blob.get_attribute(blob)
            blobs_in_frame_groundtruth.append(gt_blob)

        blobs_in_video_groundtruth.append(blobs_in_frame_groundtruth)

    groundtruth = GroundTruth(video = video,
                            blobs_in_video = blobs_in_video_groundtruth,
                            start = start,
                            end = end)
    groundtruth.wrong_crossing_counter = wrong_crossing_counter
    groundtruth.unidentified_individuals_counter = unidentified_individuals_counter
    if save_gt:
        groundtruth.save()
    return groundtruth

if __name__ == "__main__":

    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item()
    start = getInput('GroundTruth (start)', 'Input the starting frame for the interval in which the video has been validated')
    end = getInput('GroundTruth (end)', 'Input the ending frame for the interval in which the video has been validated')
    #read blob list from video
    # fragments_path = os.path.join(session_path, 'preprocessing', 'fragments.npy')
    # blobs_path = os.path.join(session_path, 'preprocessing', 'blobs_collection.npy')
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments, video.fragment_identifier_to_index)
    groundtruth = generate_groundtruth(video, list_of_blobs.blobs_in_video, int(start), int(end))
