from __future__ import absolute_import, division, print_function
import numpy as np

from blob import is_a_global_fragment, check_global_fragments

STD_TOLERANCE = 4

def detect_beginnings(boolean_array):
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def compute_model_area(blobs_in_video, number_of_animals, std_tolerance = 4):
    blobs_in_core_global_fragments = [blobs_in_frame for blobs_in_frame in blobs_in_video if is_a_global_fragment(blobs_in_frame, number_of_animals)]
    areas = [blob.area for blob in blobs_in_frame for blobs_in_frame in blobs_in_core_global_fragments]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    return ModelArea(mean_area, std_area)

class ModelArea():
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, area, std_tolerance = STD_TOLERANCE):
    return (area - self.mean) < std_tolerance * self.std

class GlobalFragment(object):
    def __init__(self, list_of_blobs, index_beginning_of_fragment):
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.average_distance_travelled = np.mean([blob.distance_travelled_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ])
        self.portraits = [blob.portraits_in_fragment()
            for blob in list_of_blobs[index_beginning_of_fragment] ]
        # self.accuracy = None

def give_me_list_of_global_fragments(list_of_blobs, num_animals):
    global_fragments_boolean_array = check_global_fragments(list_of_blobs, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(list_of_blobs,i) for i in indices_beginning_of_fragment]
