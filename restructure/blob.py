from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')
from get_portraits import getPortrait
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

STD_TOLERANCE = 4 # tolerance to select a blob as being a single fish according to the area model

class Blob(object):
    def __init__(self, centroid, contour, area, bounding_box_in_frame_coordinates, bounding_box_image = None, portrait = None, pixels = None):
        self.centroid = np.array(centroid) # numpy array (int64): coordinates of the centroid of the blob in pixels
        self.contour = contour # openCV contour [[[x1,y1]],[[x2,y2]],...,[[xn,yn]]]
        self.area = area # int: number of pixels in the blob
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates #tuple of tuples: ((x1,y1),(x2,y2)) (top-left corner, bottom-right corner) in pixels
        self.bounding_box_image = bounding_box_image # numpy array (uint8): image of the fish cropped from the video according to the bounding_box_in_frame_coordinates
        self.portrait = portrait # (numpy array (uint8),tuple(int,int),tuple(int,int)): (36x36 image of the animal,nose coordinates, head coordinates)
        self.pixels = pixels # list of int's: linearized pixels of the blob
        self.next = [] # next blob object overlapping in pixels with current blob object
        self.previous = [] # previous blob object overlapping in pixels with the current blob object
        self._fragment_identifier = None # identity in individual fragment after fragmentation
        self._identity = None # identity assigned by the algorithm

    @property
    def is_a_fish(self):
        return self.portrait is not None

    def overlaps_with(self, other):
        """Checks if contours are disjoint
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
    def is_in_a_fragment(self):
        return len(self.previous) == len(self.next) == 1

    @property
    def is_a_fish_in_a_fragment(self):
        return self.is_a_fish and self.is_in_a_fragment

    @property
    def fragment_identifier(self):
        return self._fragment_identifier

    @fragment_identifier.setter
    def fragment_identifier(self, new_fragment_identifier):
        if self.is_a_fish_in_a_fragment:
            self._fragment_identifier = new_fragment_identifier

    @property
    def identity(self):
        return self._identity

    @identity.setter
    def identity(self, new_identifier):
        assert self.is_a_fish
        self._identity = new_identifier

    @property
    def is_identified(self):
        return self._identity is not None

    def distance_travelled_in_fragment(self):
        distance = 0
        if self.is_in_a_fragment:
            current = self
            while current.next[0].is_in_a_fragment:
                distance += np.linalg.norm(current.centroid - current.next[0].centroid)
                current = current.next[0]
            current = self
            while current.previous[0].is_in_a_fragment:
                distance += np.linalg.norm(current.centroid - current.previous[0].centroid)
                current = current.previous[0]
        return distance

    def portraits_in_fragment(self):
        portraits = []
        if self.is_in_a_fragment:
            portraits.append(self.portrait[0])
            current = self
            while current.next[0].is_in_a_fragment:
                current = current.next[0]
                portraits.append(current.portrait[0])
            current = self
            while current.previous[0].is_in_a_fragment:
                current = current.previous[0]
                portraits.append(current.portrait[0])
        return portraits


def connect_blob_list(blobs_in_video):
    for frame_i in tqdm(xrange(1,len(blobs_in_video)), desc = 'Connecting blobs progress'):
        for (blob_0, blob_1) in itertools.product(blobs_in_video[frame_i-1], blobs_in_video[frame_i]):
            if blob_0.is_a_fish and blob_1.is_a_fish and blob_0.overlaps_with(blob_1):
                blob_0.now_points_to(blob_1)

def all_blobs_in_a_fragment(blobs_in_frame):
    return all([blob.is_in_a_fragment for blob in blobs_in_frame])

def is_a_global_fragment(blobs_in_frame, num_animals):
    """Returns True iff:
    * number of blobs equals num_animals
    """
    return len(blobs_in_frame)==num_animals

def check_global_fragments(blobs_in_video, num_animals):
    """Returns an array with True iff:
    * each blob has a unique blob intersecting in the past
    * number of blobs equals num_animals
    """
    return [all_blobs_in_a_fragment(blobs_in_frame) and len(blobs_in_frame)==num_animals for blobs_in_frame in blobs_in_video]

def apply_model_area(blob, model_area):
    if model_area(blob.area): #Checks if area is compatible with the model area we built
        blob.portrait = getPortrait(blob.bounding_box_image, blob.contour, blob.bounding_box_in_frame_coordinates)

def apply_model_area_to_blobs_in_frame(blobs_in_frame, model_area):
    for blob in blobs_in_frame:
        apply_model_area(blob, model_area)

def apply_model_area_to_video(blobs_in_video, model_area):
    # Parallel(n_jobs=-1)(delayed(apply_model_area_to_blobs_in_frame)(frame, model_area) for frame in tqdm(blobs_in_video, desc = 'Fragmentation progress'))
    for blobs_in_frame in tqdm(blobs_in_video, desc = 'Fragmentation progress'):
        apply_model_area_to_blobs_in_frame(blobs_in_frame, model_area)

def get_images_from_blobs_in_frame(blobs_in_frame):
    return [blob.portrait for blob in blobs_in_frame if blob.is_a_fish_in_a_fragment]

def get_images_from_blobs_in_video(blobs_in_video):
    portraits_in_video = Parallel(n_jobs=-1)(delayed(get_images_from_blobs_in_frame)(blobs_in_frame) for blobs_in_frame in tqdm(blobs_in_video, desc = 'Getting portraits'))
    return [portrait for portraits_in_frame in portraits_in_video for portrait in portraits_in_frame]

# if __name__ == "__main__":
#     contoura = np.array([ [[0,0]], [[1,1]], [[2,2]] ])
#     contourb = np.array([ [[0,1]], [[1,1]], [[2,3]] ])
#     contourc = np.array([ [[1,1]] ])
#     contourd = np.array([ [[0,1]], [[1,1]], [[2,3]] ])
#
#     print(contoura.shape)
#     a = Blob(np.array([0,0]), contoura, 0, None, None, None)
#     b = Blob(np.array([1,1]), contourb, 0, None, None, None)
#     c = Blob(np.array([2,2]), contourc, 0, None, None, None)
#     d = Blob(np.array([3,3]), contourd, 0, None, None, None)
#     print(a.overlaps_with(b))
#     print(b.overlaps_with(c))
#     list_of_blob = [[a],[b],[c],[d]]
#
#     connect_blob_list(list_of_blob)
#
#     print(check_global_fragments(list_of_blob, 1))
#     print(distance_travelled_in_fragment(a))
