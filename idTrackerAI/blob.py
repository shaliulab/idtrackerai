from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')

import cv2
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from sklearn.decomposition import PCA

from get_portraits import get_body
from fishcontour import FishContour


logger = logging.getLogger("__main__.blob")

class Blob(object):
    def __init__(self, centroid, contour, area, bounding_box_in_frame_coordinates, bounding_box_image = None, estimated_body_length = None, portrait = None, pixels = None, number_of_animals = None, frame_number = None):
        self.frame_number = frame_number
        self.number_of_animals = number_of_animals
        self.centroid = np.array(centroid) # numpy array (int64): coordinates of the centroid of the blob in pixels
        self.contour = contour # openCV contour [[[x1,y1]],[[x2,y2]],...,[[xn,yn]]]
        self.area = area # int: number of pixels in the blob
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates #tuple of tuples: ((x1,y1),(x2,y2)) (top-left corner, bottom-right corner) in pixels
        self.bounding_box_image = bounding_box_image # numpy array (uint8): image of the fish cropped from the video according to the bounding_box_in_frame_coordinates
        self.estimated_body_length = estimated_body_length
        self._portrait = portrait # numpy array (float32)
        self.pixels = pixels # list of int's: linearized pixels of the blob
        self._is_an_individual = False
        self._is_a_crossing = False
        self.reset_before_fragmentation('fragmentation')


    def reset_before_fragmentation(self, recovering_from):
        if recovering_from == 'fragmentation':
            self.next = [] # next blob object overlapping in pixels with current blob object
            self.previous = [] # previous blob object overlapping in pixels with the current blob object
            self._fragment_identifier = None # identity in individual fragment after fragmentation
            self._blob_index = None # index of the blob to plot the individual fragments

    @property
    def fragment_identifier(self):
        return self._fragment_identifier

    @property
    def is_an_individual(self):
        return self._is_an_individual

    @property
    def is_a_jump(self):
        is_a_jump = False
        if self.is_an_individual and len(self.next) == 0 and len(self.previous) == 0: # 1 frame jumps
            is_a_jump = True
        return is_a_jump

    @property
    def is_a_jumping_fragment(self):
        # this is a fragment of 2 frames that it is not considered a individual fragment but it is also not a single frame jump
        is_a_jumping_fragment = False
        if self.is_an_individual and len(self.next) == 0 and len(self.previous) == 1 and len(self.previous[0].previous) == 0 and len(self.previous[0].next) == 1: # 2 frames jumps
            is_a_jumping_fragment = True
        elif self.is_an_individual and len(self.next) == 1 and len(self.previous) == 0 and len(self.next[0].next) == 0 and len(self.next[0].previous) == 1: # 2 frames jumps
            is_a_jumping_fragment = True
        return is_a_jumping_fragment

    @property
    def is_a_ghost_crossing(self):
        return (self.is_an_individual and (len(self.next) != 1 or len(self.previous) != 1))

    @property
    def is_a_crossing(self):
        return self._is_a_crossing

    @property
    def has_ambiguous_identity(self):
        return self.is_an_individual_in_a_fragment and self.identity is list

    def overlaps_with(self, other):
        """Checks if pixels are disjoint
        """
        overlaps = False
        intersection = np.intersect1d(self.pixels, other.pixels)
        if len(intersection) > 0:
            overlaps = True
        return overlaps

    def now_points_to(self, other):
        self.next.append(other)
        other.previous.append(self)

    @property
    def used_for_training(self):
        return self._used_for_training

    @property
    def is_in_a_fragment(self):
        return len(self.previous) == len(self.next) == 1

    @property
    def is_an_individual_in_a_fragment(self):
        return self.is_an_individual and self.is_in_a_fragment

    @property
    def blob_index(self):
        return self._blob_index

    @blob_index.setter
    def blob_index(self, new_blob_index):
        if self.is_an_individual_in_a_fragment:
            self._blob_index = new_blob_index

    @property
    def portrait(self):
        return self._portrait

    @property
    def nose_coordinates(self):
        return self._nose_coordinates

    @property
    def head_coordinates(self):
        return self._head_coordinates

    @property
    def extreme1_coordinate(self):
        return self._extreme1_coordinates

    @property
    def extreme2_coordinates(self):
        return self._extreme2_coordinates

    @property
    def assigned_during_accumulation(self):
        return self._assigned_during_accumulation

    def in_a_global_fragment_core(self, blobs_in_frame):
        '''a blob in a frame is a fish in the core of a global fragment if in
        that frame there are as many blobs as number of animals to track
        '''
        return len(blobs_in_frame) == self.number_of_animals

    @property
    def identity(self):
        return self._identity

    @property
    def user_generated_identity(self):
        return self._user_generated_identity

    @property
    def identity_corrected_solving_duplication(self):
        return self._identity_corrected_solving_duplication

    @property
    def final_identity(self):
        if hasattr(self, 'user_generated_identity') and self.user_generated_identity is not None:
            return self.user_generated_identity
        elif hasattr(self, 'identity_corrected_solving_duplication') and self.identity_corrected_solving_duplication is not None:
            return self.identity_corrected_solving_duplication
        else:
            return self.identity

    @property
    def is_identified(self):
        return self._identity is not None

    def compute_overlapping_with_previous_blob(self):
        number_of_previous_blobs = len(self.previous)
        if number_of_previous_blobs == 1:
            self.non_shared_information_with_previous = 1. - len(np.intersect1d(self.pixels, self.previous[0].pixels)) / np.mean([len(self.pixels), len(self.previous[0].pixels)])
            if self.non_shared_information_with_previous is np.nan:
                logger.debug("intersection both blobs %s" %str(len(np.intersect1d(self.pixels, self.previous[0].pixels))))
                logger.debug("mean pixels both blobs %s" %str(np.mean([len(self.pixels), len(self.previous[0].pixels)])))
                raise ValueError("non_shared_information_with_previous is nan")

    def apply_model_area(self, video, model_area, portraitSize, number_of_blobs):
        if model_area(self.area) or number_of_blobs == video.number_of_animals: #Checks if area is compatible with the model area we built
            if video.resolution_reduction == 1:
                height = video._height
                width = video._width
            else:
                height  = int(video._height * video.resolution_reduction)
                width  = int(video._width * video.resolution_reduction)

            portrait, \
            self._extreme1_coordinates, \
            self._extreme2_coordinates = get_body(height, width,
                                                self.bounding_box_image,
                                                self.pixels,
                                                self.bounding_box_in_frame_coordinates,
                                                portraitSize)
            self._portrait = ((portrait - np.mean(portrait))/np.std(portrait)).astype('float32')
            self._is_an_individual = True
        else:
            self._is_a_crossing = True

    def get_nose_and_head_coordinates(self):
        if self.is_an_individual:
            # Calculating nose coordinates in the full frame reference
            contour_cnt = FishContour.fromcv2contour(self.contour)
            noseFull, _, head_centroid_full = contour_cnt.find_nose_and_orientation()
            self._nose_coordinates = tuple(noseFull.astype('float32'))
            self._head_coordinates = tuple(head_centroid_full.astype('float32'))
        else:
            self._nose_coordinates = None
            self._head_coordinates = None
