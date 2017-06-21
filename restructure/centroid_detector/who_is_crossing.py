from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
import numpy as np
from tqdm import tqdm
from blob import ListOfBlobs
import matplotlib.pyplot as plt
from py_utils import get_spaced_colors_util

"""if the blob was not assigned durign the standard procedure it could be a jump or a crossing
these blobs are blobs that are not in a fragment. We distinguish the two cases in which
they are single fish (blob.is_a_fish == True) or crossings (blob.is_a_fish == False)
"""
#load video and list of blobs
video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/video_object.npy').item()
number_of_animals = video.number_of_animals
list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/preprocessing/blobs_collection.npy'
list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
blobs = list_of_blobs.blobs_in_video

crossing_frames = []
jumps = [] # array of non assigned portrait to be sent to the network for one-shot recognition [to be conditioned wrt 2 * 99perc[velocity]]
num_animals_in_crossing = []
plt.ion()
# fig, ax_arr = plt.subplots(1,2)


for frame_num, blobs_in_frame in tqdm(enumerate(blobs)):
    print("frame number ", frame_num)
    for blob_num, blob in enumerate(blobs_in_frame):
        print("blob number ", blob_num)
        if blob.identity == 0:
            # if it is a fish, than it has a portrait and can be assigned
            if blob.is_a_fish:
                print("Jump!")
                jumps.append(blob.portrait[0])
                #plt.title("Jump")
                # plt.imshow(blob.portrait[0])
            else:
                print("Crossing!")
                crossing_frames.append(frame_num)

for crossing_frame in crossing_frames:
    print("frame num ", crossing_frame)
    previous_blobs = [blob.previous for blob in blobs[crossing_frame] if len(blob.previous) > 1]
    print("previous blobs ", previous_blobs)
    ids_in_crossings = [[blob.identity for blob in previous_blob_sublist] for previous_blob_sublist in previous_blobs]
    print("ids in crossing ", ids_in_crossings)
