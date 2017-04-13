from __future__ import absolute_import, division, print_function
import numpy as np

from blobs import check_global_fragments

class GlobalFragment(object):
    def __init__(self, list_of_blobs,index_beginning_of_fragment)
        self.index_beginning_of_fragment = index_beginning_of_fragment
        self.average_distance_travelled = np.average([blob.distance_travelled_in_segment 
            for blob in list_of_blobs[index_beginning_of_fragment] ])
        self.portraits = [blob.portraits_in_segment 
                for blob in list_of_blobs[index_beginning_of_fragment] ] 

def detect_beginnings(boolean_array):
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])] 

def give_me_list_of_global_fragments(list_of_blobs):
    global_fragments_boolean_array = check_global_fragments(list_of_blobs):
    index_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(list_of_blobs,i) for i in index_beginning_of_fragment] 



