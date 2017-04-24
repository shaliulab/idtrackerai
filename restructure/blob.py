from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./utils')
sys.path.append('./preprocessing')
from get_portraits import getPortrait
import itertools
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

STD_TOLERANCE = 4

class Blob(object):#NOTE: think better to overlap with pixels instead of contour (go for safe option)
    def __init__(self, centroid, contour, area, bounding_box_in_frame_coordinates, bounding_box_image = None, portrait = None, pixels = None):
        self.centroid = np.array(centroid)
        self.contour = contour
        self.area = area
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates
        self.bounding_box_image = bounding_box_image
        self.portrait = portrait
        self.pixels = pixels

        self.next = []
        self.previous = []
        self._identity_in_fragment = None
        self._identity = None

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
    def identity_in_fragment(self):
        return self._identity_in_fragment

    @identity_in_fragment.setter
    def identity_in_fragment(self, fragment_identifier):
        if self.is_a_fish_in_a_fragment:
            self._identity_in_fragment = fragment_identifier

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
            portraits.append(self.portrait)
            current = self
            while current.next[0].is_in_a_fragment:
                current = current.next[0]
                portraits.append(current.portrait)
            current = self
            while current.previous[0].is_in_a_fragment:
                current = current.previous[0]
                portraits.append(current.portrait)
        return portraits


def connect_blob_list(blob_list):
    for frame_i in tqdm(xrange(1,len(blob_list)), desc = 'Connecting blobs progress'):
        for (blob_0, blob_1) in itertools.product(blob_list[frame_i-1], blob_list[frame_i]):
            if blob_0.is_a_fish and blob_1.is_a_fish and blob_0.overlaps_with(blob_1):
                blob_0.now_points_to(blob_1)

def all_blobs_in_a_fragment(frame):
    return all([blob.is_in_a_fragment for blob in frame])

def is_a_global_fragment(blobs_in_frame, num_animals):
    """Returns True iff:
    * number of blobs equals num_animals
    """
    return len(blobs_in_frame)==num_animals

def check_global_fragments(blob_list, num_animals):
    """Returns an array with True iff:
    * each blob has a unique blob intersecting in the past
    * number of blobs equals num_animals
    """
    return [all_blobs_in_a_fragment(frame) and len(frame)==num_animals for frame in blob_list]


def apply_model_area(blob, model_area):
    if model_area(blob.area): #Checks if area is compatible with the model area we built
        blob.portrait = getPortrait(blob.bounding_box_image, blob.contour, blob.bounding_box_in_frame_coordinates)

def apply_model_area_to_blobs_in_frame(blobs_in_frame, model_area):
    for blob in blobs_in_frame:
        apply_model_area(blob, model_area)

def apply_model_area_to_video(blob_list, model_area):
    # Parallel(n_jobs=1, verbose = 5)(delayed(apply_model_area_to_blobs_in_frame)(frame, model_area) for frame in blob_list)
    for frame in tqdm(blob_list, desc = 'Fragmentation progress'):
        apply_model_area_to_blobs_in_frame(frame, model_area)


if __name__ == "__main__":
    contoura = np.array([ [[0,0]], [[1,1]], [[2,2]] ])
    contourb = np.array([ [[0,1]], [[1,1]], [[2,3]] ])
    contourc = np.array([ [[1,1]] ])
    contourd = np.array([ [[0,1]], [[1,1]], [[2,3]] ])

    print(contoura.shape)
    a = Blob(np.array([0,0]), contoura, 0, None, None, None)
    b = Blob(np.array([1,1]), contourb, 0, None, None, None)
    c = Blob(np.array([2,2]), contourc, 0, None, None, None)
    d = Blob(np.array([3,3]), contourd, 0, None, None, None)
    print(a.overlaps_with(b))
    print(b.overlaps_with(c))
    list_of_blob = [[a],[b],[c],[d]]

    connect_blob_list(list_of_blob)

    print(check_global_fragments(list_of_blob, 1))
    print(distance_travelled_in_fragment(a))
