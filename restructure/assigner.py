from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import matplotlib.pyplot as plt

from network_params import NetworkParams
from get_data import DataSet
from id_CNN import ConvNetwork
from get_predictions import GetPrediction
from blob import get_images_from_blobs_in_video

def assign(blobs, params, print_flag):

    net = ConvNetwork(params)
    # Get images from the blob collection
    images = get_images_from_blobs_in_video(blobs)
    # build data object
    data = DataSet(params.number_of_animals, images)
    # Instantiate data_set
    data.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    data.crop_images(image_size = 32)
    # Restore network
    net.restore()
    # Train network
    assigner = GetPrediction(data,
                            starting_epoch = 0,
                            print_flag = print_flag)

    assigner.get_predictinos(data, net.predict)
