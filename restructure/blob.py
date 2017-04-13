from __future__ import absolute_import, division, print_function
import itertools
import numpy as np

class Blob(object):
    def __init__(self, centroid, contour, area, bounding_box_in_frame_coordinates, bounding_box_image = None, portrait = None):
        self.centroid = centroid
        self.contour = contour         
        self.area = area
        self.bounding_box_in_frame_coordinates = bounding_box_in_frame_coordinates
        self.bounding_box_image = bounding_box_image
        self.portrait = portrait

        self.next = []
        self.previous = []

        self._identity = None
    
    @property
    def is_a_fish(self):
        return self.portrait is not None

    def overlaps_with(self, other):
        """Checks if contours are disjoint
        """
        overlaps = False
        for ([[x,y]],[[x1,y1]]) in itertools.product(self.contour, other.contour): 
            if x == x1 and y == y1:
                overlaps = True
                break
        return overlaps

    def now_points_to(self, other):
        self.next.append(other)
        other.previous.append(self)

    @property
    def is_in_a_fragment(self):
        return len(self.previous) == 1

    @property
    def is_a_fish_in_a_fragment(self):
        self.is_a_fish and self.is_in_a_fragment 

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

def connect_blob_list(blob_list):
    for frame_i in range(1,len(blob_list)):
        for (blob_0, blob_1) in itertools.product(blob_list[frame_i-1], blob_list[frame_i]):
            if blob_0.overlaps_with(blob_1):
                blob_0.now_points_to(blob_1)

def all_blobs_in_a_fragment(frame):
    return all([blob.is_in_a_fragment for blob in frame])

def check_global_fragments(blob_list, num_animals):
    """Returns an array with True iff: 
    * each blob has a unique blob intersecting in the past
    * number of blobs equals num_animals
    """
    return [all_blobs_in_a_fragment(frame) and len(frame)==num_animals for frame in blob_list]

def check_potential_global_fragments(blob_list, num_animals):
    """Returns an array with True iff: 
    * number of blobs equals num_animals
    """
    return [len(frame)==num_animals for frame in blob_list]

if __name__ == "__main__":
    contoura = np.array([ [[0,0]], [[1,1]], [[2,2]] ])
    contourb = np.array([ [[0,1]], [[1,1]], [[2,3]] ])
    contourc = np.array([ [[1,1]] ])
    print(contoura.shape)
    a = Blob(None, contoura, 0, None, None, None)
    b = Blob(None, contourb, 0, None, None, None)
    c = Blob(None, contourc, 0, None, None, None)
    print(a.overlaps_with(b))
    print(b.overlaps_with(c))
    list_of_blob = [[a],[b],[c]]
    
    connect_blob_list(list_of_blob) 
    
    print(check_global_fragments(list_of_blob, 1))






