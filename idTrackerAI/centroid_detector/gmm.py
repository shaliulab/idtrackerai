from __future__ import absolute_import, print_function, division
import numpy as np
from sklearn import mixture
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
from tqdm import tqdm
from blob import ListOfBlobs
import matplotlib.pyplot as plt
from py_utils import get_spaced_colors_util

def fit_samples(samples, n_components = 2, covariance_type = "full"):
    gmix = mixture.GMM(n_components= n_components, covariance_type = covariance_type)
    gmix.fit(samples)
    available_colors = get_spaced_colors_util(n_components, norm = True, black = False)
    return [available_colors[i] for i in gmix.predict(samples)]


#load video and list of blobs
video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_2/video_object.npy').item()
number_of_animals = video.number_of_animals
list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_2/preprocessing/blobs_collection.npy'
list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
blobs = list_of_blobs.blobs_in_video

crossings = []
jumps = []
num_animals_in_crossing = []
plt.ion()
fig, ax_arr = plt.subplots(1,2)

def count_animals_in_crossing(blobs, frame_num):
    previous_blobs = blobs[frame_num - 1]
    cur_blobs = blobs[frame_num]
    print("cur blobs ", cur_blobs)
    num_animals_per_current_blobs = np.zeros_like(cur_blobs)

    for i, cur_blob in enumerate(cur_blobs):
        print(i)
        for prev_blob in previous_blobs:
            if len(prev_blob.next) > 0 and prev_blob.next[0] is cur_blob:
                num_animals_per_current_blobs[i] += 1

    print("number of guys ", num_animals_per_current_blobs)
    return num_animals_per_current_blobs

# def count_animals_in_crossing(blobs, frame_num):



for frame_num, blobs_in_frame in tqdm(enumerate(blobs)):
    print("frame number ", frame_num)
    for blob_num, blob in enumerate(blobs_in_frame):
        print("blob number ", blob_num)
        if blob.identity == 0 and (len(blob.previous) == 0 or len(blob.next) == 0):
            print("is it a fish? ", blob.is_an_individual)
            print("is it in a fragment? ", blob.is_in_a_fragment)
            if blob.portrait is not None:
                print("Jump!")
                jumps.append(blob.portrait)
                plt.title("Jump")
                # plt.imshow(blob.portrait)
            else:
                print("Crossing!")
                number_of_animals_crossing_in_current_frame = count_animals_in_crossing(blobs, frame_num)[blob_num]
                # print(len(blobs_in_frame))
                # pxs = np.array(np.unravel_index(blob.pixels,(video._height,video._width))).T
                # # pxs = np.concatenate((pxs, blob.bounding_box_image))
                # crossings.append(pxs)
                ax_arr[0].imshow(blob.bounding_box_image)
                print("number of crossing animals: ", number_of_animals_crossing_in_current_frame)
                # colors = fit_samples(pxs, n_components = number_of_animals_crossing_in_current_frame, covariance_type = "full")
                #
            	# ax_arr[1].scatter(pxs[:,1], -pxs[:,0], c = colors, alpha=0.8)
                #
                # # plt.imshow(blob.bounding_box_image)
                plt.pause(0.5)
                # ax_arr[1].clear()
        #
        # elif blob.identity == 0 and len(blob.previous) > 0:
        #     crossings.append(blob.bounding_box_image)
        #     plt.imshow(blob.bounding_box_image)
        #     plt.pause(0.5)
        #     num_animals_in_crossing.append(max(len(blob.previous), len(blob.next)))
        #     print("----------------num animals = ", num_animals_in_crossing[-1])
