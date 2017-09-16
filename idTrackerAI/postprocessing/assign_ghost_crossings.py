from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import collections
from blob import ListOfBlobs

import matplotlib.pyplot as plt
from py_utils import get_spaced_colors_util
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from video_utils import segmentVideo, filterContoursBySize, getPixelsList, getBoundigBox

def assign_ghost_crossings(blobs):
    for blobs_in_frame in tqdm(blobs, desc = 'Assign identity to individual fragments extremes'):
        # try:
            # print("\nframe number ", blobs_in_frame[0].frame_number)
        # except:
            # print("last frame")
        for blob in blobs_in_frame:
            #if a blob has not been assigned but it is a fish and overlaps with one fragment
            #assign it!
            if blob.identity == 0 and (blob.is_a_fish or blob.is_a_ghost_crossing):
                # print("is a ghost crossing ", blob.is_a_ghost_crossing)
                # print("num next ", len(blob.next))
                # print("num prev ", len(blob.previous))
                # print("identity ", blob.identity)
                if len(blob.next) == 1:
                    # print("next is 1")
                    # print("next id ", blob.next[0].identity)
                    blob._identity = blob.next[0].identity
                    blob._frequencies_in_fragment = blob.next[0].frequencies_in_fragment
                    blob._P1_vector = blob.next[0].P1_vector
                    blob._P2_vector = blob.next[0].P2_vector ### NOTE: this is not strictly correct as it should be recomputed
                    # blob.is_an_extreme_of_individual_fragment = True
                elif len(blob.previous) == 1:
                    # print("prev is 1")
                    # print("prev id ", blob.previous[0].identity)
                    blob._identity = blob.previous[0].identity
                    blob._frequencies_in_fragment = blob.previous[0].frequencies_in_fragment
                    blob._P1_vector = blob.previous[0].P1_vector
                    blob._P2_vector = blob.previous[0].P2_vector ### NOTE: this is not strictly correct as it should be recomputed
                    # blob.is_an_extreme_of_individual_fragment = True
                print("assigned during accumulation ----->", blob.assigned_during_accumulation)
