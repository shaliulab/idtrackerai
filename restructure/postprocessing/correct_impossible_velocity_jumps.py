from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
import numpy as np
from tqdm import tqdm
import collections
from blob import ListOfBlobs
from assigner import assign
from id_CNN import ConvNetwork
from network_params import NetworkParams
from blob import Blob
from compute_velocity_model import compute_velocity_from_list_of_blobs, compute_model_velocity

VEL_PERCENTILE = 99 #percentile used to model velocity jumps

def compute_velocity_subsequent_frames(blob):
    return np.linalg.norm(blob.next[0].centroid - blob.centroid)

def correct_impossible_jumps(video_object, blobs_in_video):
    if not hasattr(video, "percentile_velocity_threshold"):
        video_object.percentile_velocity_threshold = compute_model_velocity(blobs, video_object.number_of_animals, percentile = VEL_PERCENTILE)
        video_object.save()

    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment:
                velocity = compute_velocity_subsequent_frames(blob)
                if velocity > video_object.percentile_velocity_threshold:
                    print("shit! it jumped")




if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    # video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/video_object.npy').item()
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Conflicto8/session_4/video_object.npy').item()
    number_of_animals = video.number_of_animals
    # list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/preprocessing/blobs_collection.npy'
    list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Conflicto8/session_2/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
