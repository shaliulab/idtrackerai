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

def assing_identity_to_individual_fragments_extremes(blobs):
    for blobs_in_frame in tqdm(blobs, desc = 'Assign identity to individual fragments extremes'):
        for blob in blobs_in_frame:
            #if a blob has not been assigned but it is a fish and overlaps with one fragment
            #assign it!
            if blob.identity == 0 and blob.is_a_fish:
                if len(blob.next) == 1:
                    blob.identity = blob.next[0].identity
                    blob.is_an_extreme_of_individual_fragment = True
                elif len(blob.previous) == 1:
                    blob.identity = blob.previous[0].identity
                    blob.is_an_extreme_of_individual_fragment = True
