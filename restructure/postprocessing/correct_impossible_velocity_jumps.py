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

def compute_velocity_subsequent_frames(blob1, blob2):
    return np.linalg.norm(blob2.centroid - blob1.centroid)

def give_me_blob_next_frame_same_identity_not_in_a_fragment(list_of_blobs, blob):
    """Given a blob get the one in the next frame that has the same identity, it is a fish a not in a fragment """
    next_blobs = [_blob for _blob in list_of_blobs[blob.frame_number] if _blob.identity == blob.identity and blob.is_a_fish and not blob.is_in_a_fragment]
    return next_blobs[0] if len(next_blobs) > 0 else None

def correct_impossible_jumps(video_object, blobs_in_video):
    if not hasattr(video, "percentile_velocity_threshold"):
        video_object.percentile_velocity_threshold = compute_model_velocity(blobs, video_object.number_of_animals, percentile = VEL_PERCENTILE)
        video_object.save()

    for blobs_in_frame in tqdm(blobs_in_video, desc = "correcting impossible jumps"):
        for blob in blobs_in_frame:
            if blob.is_a_fish_in_a_fragment and blob.frame_number < video._num_frames:
                next_blob = give_me_blob_next_frame_same_identity_not_in_a_fragment(blobs_in_video, blob)
                if next_blob is not None:
                    velocity = compute_velocity_subsequent_frames(blob, next_blob)
                    print("velocity ", velocity, " velocity th ", video_object.percentile_velocity_threshold)
                    if velocity > video_object.percentile_velocity_threshold:
                        print("shit! it jumped")




if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    # video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/video_object.npy').item()
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    # list_of_blobs_path = '/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/session_4/preprocessing/blobs_collection.npy'
    list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    correct_impossible_jumps(video, blobs)
