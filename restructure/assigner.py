from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import matplotlib.pyplot as plt
import numpy as np

from network_params import NetworkParams
from get_data import DataSet
from id_CNN import ConvNetwork
from get_predictions import GetPrediction
from blob import get_images_from_blobs_in_video

MAX_FLOAT = sys.float_info[0]
MIN_FLOAT = sys.float_info[3]

def assign(video, blobs_in_video, params, video_episodes_start_end, print_flag):

    net = ConvNetwork(params, training_flag = False)
    # Get images from the blob collection
    images = get_images_from_blobs_in_video(blobs_in_video, video_episodes_start_end)
    # build data object
    images = np.expand_dims(np.asarray(images), axis = 3)
    data = DataSet(params.number_of_animals, images)
    # Instantiate data_set
    data.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    data.crop_images(image_size = 32)
    # Restore network
    net.restore()
    # Train network
    assigner = GetPrediction(data, print_flag = print_flag)

    assigner.get_predictions(net.predict)

    assign_identity_to_blobs_in_video(blobs_in_video, assigner)

    assign_identity_to_blobs_in_fragment(video, blobs_in_video)
    

def assign_identity_to_blobs_in_video(blobs_in_video, assigner):
    counter = 0
    for frame in blobs_in_video:
        for blob in frame:
            if not blob.is_a_fish_in_a_fragment:
                blob._identity = 0
            else:
                blob._identity = int(assigner._predictions[counter])
                counter += 1

def assign_identity_to_blobs_in_fragment(video, blobs_in_video):
    print('In assign_identity_to_blobs_in_fragment')
    for frame in blobs_in_video:
        for blob in frame:
            # print(blob.is_a_fish_in_a_fragment)
            # print(len(blob.previous))
            if blob.is_a_fish_in_a_fragment and not blob.previous[0].is_a_fish_in_a_fragment:
                # Get per blob identities in the fragment
                identities_in_fragment = np.asarray(blob.identities_in_fragment())
                # Compute frequencies of assignation for each identity
                frequencies = np.asarray([np.sum(identities_in_fragment == i) for i in range(1,video.number_of_animals+1)])
                # Compute numerator of P1 and check that it is not inf
                numerator = 2.**frequencies
                if np.any(numerator == np.inf):
                    numerator[numerator == np.inf] = MAX_FLOAT
                # Compute denominator of P1
                denominator = np.sum(numerator)
                # Compute P1 and check that it is not 0. for any identity
                P1_of_fragment = numerator / denominator
                if np.any(P1_of_fragment == 0.):
                    P1_of_fragment[P1_of_fragment == 0.] = MIN_FLOAT
                if np.any(P1_of_fragment == 0.):
                    raise ValueError('P1_of_fragment cannot be 0')
                # Change P1 that are 1. for 0.9999 so that we do not have problems when computing P2
                P1_of_fragment[P1_of_fragment == 1.] = 0.9999
                if np.any(P1_of_fragment == 1.):
                    raise ValueError('P1_of_fragment cannot be 1')
                # Assign identity to the fragment
                identity_in_fragment = np.argmax(P1_of_fragment) + 1
                # Update identity of all blobs in fragment
                blob.update_identity_in_fragment(identity_in_fragment)
