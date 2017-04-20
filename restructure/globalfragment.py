from __future__ import absolute_import, division, print_function
import numpy as np

from blob import is_a_global_fragment, check_global_fragments

# class PotentialGlobalFragment(object):
#     def __init__(self, list_of_blobs, index_beginning_of_fragment):
#         self.index_beginning_of_fragment = index_beginning_of_fragment
#         self.list_of_blobs = list_of_blobs
#
# def give_me_list_of_potential_global_fragments(list_of_blobs, num_animals):
#     global_fragments_boolean_array = check_potential_global_fragments(list_of_blobs, num_animals)
#     indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
#     return [PotentialGlobalFragment(list_of_blobs,i) for i in indices_beginning_of_fragment]
#


def detect_beginnings(boolean_array):
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def compute_model_area(blobs_in_video, number_of_animals, std_tolerance = 4):
    blobs_in_core_global_fragments = [blobs_in_frame for blobs_in_frame in blobs_in_video if is_a_global_fragment(blobs_in_frame, number_of_animals)]
    areas = [blob.area for blob in blobs_in_frame for blobs_in_frame in blobs_in_core_global_fragments]
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    def model_area(area):
        return abs(area-mean_area) < std_tolerance * std_area
    return model_area


class GlobalFragment(object):
    def __init__(self, list_of_blobs, index_beginning_of_fragment):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.list_of_blobs = list_of_blobs
        self.average_distance_travelled = np.average([blob.distance_travelled_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ])
        self.portraits = [blob.portraits_in_fragment
            for blob in list_of_blobs[index_beginning_of_fragment] ]


def give_me_list_of_global_fragments(list_of_blobs, num_animals):
    global_fragments_boolean_array = check_global_fragments(list_of_blobs, num_animals)
    print('glob frag bool ', global_fragments_boolean_array)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    print('indices beginning frag ', indices_beginning_of_fragment)
    return [GlobalFragment(list_of_blobs,i) for i in indices_beginning_of_fragment]
