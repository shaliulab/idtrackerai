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

from fishcontour import FishContour

logger = logging.getLogger("__main__.blob")

class Blob(object):
    def __init__(self, centroid, contour, area, bounding_box_in_frame_coordinates, bounding_box_image = None, estimated_body_length = None, pixels = None, number_of_animals = None, frame_number = None):
        self.frame_number = frame_number
        self.number_of_animals = number_of_animals
        self.centroid = np.array(centroid) # numpy array (int64): coordinates of the centroid of the blob in pixels
        self.contour = contour # openCV contour [[[x1,y1]],[[x2,y2]],...,[[xn,yn]]]
        self.area = area # int: number of pixels in the blob
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates #tuple of tuples: ((x1,y1),(x2,y2)) (top-left corner, bottom-right corner) in pixels
        self.bounding_box_image = bounding_box_image # numpy array (uint8): image of the fish cropped from the video according to the bounding_box_in_frame_coordinates
        self.estimated_body_length = estimated_body_length
        self._image_for_identification = None # numpy array (float32)
        self.pixels = pixels # list of int's: linearized pixels of the blob
        self._is_an_individual = False
        self._is_a_crossing = False
        self.reset_before_fragmentation('fragmentation')
        self._used_for_training = None
        self._accumulation_step = None

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

    def check_for_multiple_next_or_previous(self, direction = None):
        current = getattr(self, direction)[0]

        while len(getattr(current, direction)) == 1:

            current = getattr(current, direction)[0]
            if len(getattr(self, direction)) > 1:
                return True
                break

        return False

    def is_a_sure_individual(self):
        if self.is_an_individual and len(self.previous) > 0 and len(self.next) > 0:
            has_multiple_previous = self.check_for_multiple_next_or_previous('previous')
            has_multiple_next = self.check_for_multiple_next_or_previous('next')
            if not has_multiple_previous and not has_multiple_next:
                return True
        else:
            return False

    def is_a_sure_crossing(self):
        if len(self.previous) > 1 or len(self.next) > 1:
            return True
        elif len(self.previous) == 1 and len(self.next) == 1:
            has_multiple_previous = self.check_for_multiple_next_or_previous('previous')
            has_multiple_next = self.check_for_multiple_next_or_previous('next')
            if has_multiple_previous and has_multiple_next:
                return True
        else:
            return False

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
    def accumulation_step(self):
        return self._accumulation_step

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
    def image_for_identification(self):
        return self._image_for_identification

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

    def apply_model_area(self, video, model_area, identification_image_size, number_of_blobs):
        if model_area(self.area) or number_of_blobs == video.number_of_animals: #Checks if area is compatible with the model area we built

            image_for_identification, \
            self._extreme1_coordinates, \
            self._extreme2_coordinates = self.get_image_for_identification(video)
            self._image_for_identification = ((image_for_identification - np.mean(image_for_identification))/np.std(image_for_identification)).astype('float32')
            self._is_an_individual = True
        else:
            self._is_a_crossing = True

    def get_image_for_identification(self, video):
        if video.resolution_reduction == 1:
            height = video._height
            width = video._width
        else:
            height  = int(video._height * video.resolution_reduction)
            width  = int(video._width * video.resolution_reduction)

        return self._get_image_for_identification(height, width,
                                                self.bounding_box_image, self.pixels,
                                                self.bounding_box_in_frame_coordinates,
                                                video.identification_image_size[0])

    @staticmethod
    def _get_image_for_identification(height, width, bounding_box_image, pixels, bounding_box_in_frame_coordinates, identification_image_size):
        bounding_box_image = remove_background_pixels(height, width, bounding_box_image, pixels, bounding_box_in_frame_coordinates)
        pca = PCA()
        pxs = np.unravel_index(pixels,(height,width))
        pxs1 = np.asarray(zip(pxs[0],pxs[1]))
        pca.fit(pxs1)
        rot_ang = 180 - np.arctan(pca.components_[0][1]/pca.components_[0][0])*180/np.pi - 45 # we substract 45 so that the fish is aligned in the diagonal. This say we have smaller frames
        center = (pca.mean_[1], pca.mean_[0])
        # print("PCA center before: ", center)
        center = full2miniframe(center, bounding_box_in_frame_coordinates)
        center = np.array([int(center[0]), int(center[1])])
        # print("PCA center and angle: ", center, rot_ang)

        #rotate
        diag = np.sqrt(np.sum(np.asarray(bounding_box_image.shape)**2)).astype(int)
        diag = (diag, diag)
        M = cv2.getRotationMatrix2D(tuple(center), rot_ang, 1)
        minif_rot = cv2.warpAffine(bounding_box_image, M, diag, borderMode=cv2.BORDER_CONSTANT, flags = cv2.INTER_CUBIC)


        crop_distance = int(identification_image_size/2)
        x_range = xrange(center[0] - crop_distance, center[0] + crop_distance)
        y_range = xrange(center[1] - crop_distance, center[1] + crop_distance)
        image_for_identification = minif_rot.take(y_range, mode = 'wrap', axis=0).take(x_range, mode = 'wrap', axis=1)
        height, width = image_for_identification.shape

        rot_ang_rad = rot_ang * np.pi / 180
        h_or_t_1 = np.array([np.cos(rot_ang_rad), np.sin(rot_ang_rad)]) * rot_ang_rad
        h_or_t_2 = - h_or_t_1
        # print(h_or_t_1,h_or_t_2,image_for_identification.shape)
        return image_for_identification, tuple(h_or_t_1.astype('int')), tuple(h_or_t_2.astype('int'))

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

def remove_background_pixels(height, width, bounding_box_image, pixels, bounding_box_in_frame_coordinates):
    pxs = np.array(np.unravel_index(pixels,(height, width))).T
    pxs = np.array([pxs[:, 0] - bounding_box_in_frame_coordinates[0][1], pxs[:, 1] - bounding_box_in_frame_coordinates[0][0]])
    temp_image = np.zeros_like(bounding_box_image).astype('uint8')
    temp_image[pxs[0,:], pxs[1,:]] = 255
    temp_image = cv2.dilate(temp_image, np.ones((3,3)).astype('uint8'), iterations = 1)
    rows, columns = np.where(temp_image == 255)
    dilated_pixels = np.array([rows, columns])

    temp_image[dilated_pixels[0,:], dilated_pixels[1,:]] = bounding_box_image[dilated_pixels[0,:], dilated_pixels[1,:]]
    return temp_image

def full2miniframe(point, boundingBox):
    """
    Push a point in the fullframe to miniframe coordinate system.
    Here it is use for centroids
    """
    return tuple(np.asarray(point) - np.asarray([boundingBox[0][0],boundingBox[0][1]]))
