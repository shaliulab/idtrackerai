import sys
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/CNN')
sys.path.append('IdTrackerDeep/tracker')
from py_utils import *
from video_utils import *
from idTrainerTracker import *
from cnn_utils import getCkptvideoPath
from cnn_utils import *

import time
import numpy as np
np.set_printoptions(precision=2)
import numpy.matlib
import argparse
import os
import glob
import pandas as pd
import re
from joblib import Parallel, delayed
import multiprocessing
import cPickle as pickle
import tensorflow as tf
from tf_utils import *
from input_data_cnn import *
from cnn_utils import *
from pprint import pprint
from collections import Counter
import collections
import datetime

IMSIZE = (32,32,1)

class IdAssigner(object):
    def __init__(self, video_path, portraits, video_fragmentation_data, ind_frags_to_identify, reference_ind_frags):
        self.video_path = video_path
        self.all_images = portraits
        self.fragments_data = video_fragmentation_data
        self.test_fragments_indices = ind_frags_to_identify
        self.ref_fragments_indices = reference_ind_frags
        self.imsize = IMSIZE

    def get_data(self, blob_index):
        portraitsFrag = np.asarray(portraits.loc[:,'images'].tolist())
        portsFragments = []
        # print 'indivFragments', indivFragments
        for indivFragment in indivFragments:
            print 'current indivFragment ', indivFragment
            # print 'indiv fragment, ', indivFragment
            portsFragment = []
            # print 'portsFragment, ', portsFragment
            for (frame, column) in indivFragment:
                portsFragment.append(portraitsFrag[frame][column])

            portsFragments.append(np.asarray(portsFragment))
        # print portsFragments
        images = np.vstack(portsFragments)
        images = np.expand_dims(images,axis=3)
        images = cropImages(images,32)
        self.images = standarizeImages(images)
        return self.images

video_path
portraits
video_fragmentation_data
ind_frags_to_identify
reference_ind_frags

a = IdAssigner()
