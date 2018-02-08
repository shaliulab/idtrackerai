from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
import logging

from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from blob import Blob
from GUI_utils import selectDir, getInput

if sys.argv[0] == 'idtrackerdeepApp.py':
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
                                            'blob_index', 'fragment_identifier']):
        self.attributes = attributes_to_get

    def get_attribute(self, blob):
        for attribute in self.attributes:
            if attribute == 'identity':
                setattr(self, attribute, getattr(blob, 'final_identity'))
            else:
                setattr(self, attribute, getattr(blob, attribute))

class IndividualGroundTruth(object):
    def __init__(self, video = [], individual_blobs_in_video = [], start = None, end = None, validated_identity = None):
        self.video = video
        self.individual_blobs_in_video = individual_blobs_in_video
        self.start = start
        self.end = end
        self.validated_identity = validated_identity

    def save(self):
        gt_name = '_individual_' + str(self.validated_identity) + '_groundtruth.npy'
        path_to_save_groundtruth = os.path.join(os.path.split(self.video.video_path)[0], gt_name)
        logger.info("saving ground truth at %s" %path_to_save_groundtruth)
        np.save(path_to_save_groundtruth, self)
        logger.info("done")

def generate_individual_groundtruth(video, blobs_in_video = None,
                                    start = None, end = None,
                                    validated_identity = None, save_gt = True):
    """Generates a list of light blobs_in_video, given a video object corresponding to a
    tracked video
    """
    assert video.has_been_assigned == True
    individual_blobs_in_video_groundtruth = []

    for blobs_in_frame in blobs_in_video:
        identities_in_frame = set([blob.final_identity for blob in blobs_in_frame])
        for blob in blobs_in_frame:
            if blob.final_identity == validated_identity:
                gt_blob = GroundTruthBlob()
                gt_blob.get_attribute(blob)
                individual_blobs_in_video_groundtruth.append(gt_blob)

    groundtruth = IndividualGroundTruth(video = video,
                            individual_blobs_in_video = individual_blobs_in_video_groundtruth,
                            start = start,
                            end = end,
                            validated_identity = validated_identity)
    if save_gt:
        groundtruth.save()
    return groundtruth
