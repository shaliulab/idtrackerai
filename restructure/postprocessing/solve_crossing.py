from __future__ import absolute_import, print_function, division
import matplotlib
matplotlib.use('TKAgg')
from sklearn import mixture
import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../utils')
sys.path.append('../network')
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

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_identities_in_crossing_forward(blob):
    blob._identity = list(flatten([previous_blob.identity for previous_blob in blob.previous]))
    blob.bad_crossing = False

    for previous_blob in blob.previous:
        if previous_blob.is_a_crossing:
            previous_has_more_than_one_crossing = sum([previous_blob_next.is_a_crossing for previous_blob_next in previous_blob.next]) > 1
            if previous_has_more_than_one_crossing:
                blob.bad_crossing = True
            if len(previous_blob.next) != 1: # the previous crossing_blob is splitting

                for previous_blob_next in previous_blob.next:
                    if previous_blob_next is not blob: # for every next of the previous that is not the current blob we remove the identities
                        if previous_blob_next.is_a_fish and previous_blob_next.identity != 0 and previous_blob_next.identity in blob._identity:
                            blob._identity.remove(previous_blob_next.identity)
    return blob

def get_identities_in_crossing_backward(blob, blobs_in_frame):
    has_more_than_one_crossing = sum([blob_previous.is_a_crossing for blob_previous in blob.previous]) > 1

    for blob_previous in blob.previous:
        if blob_previous.is_a_crossing and blob_previous.bad_crossing and has_more_than_one_crossing:
            blob_previous.bad_crossing = True
    blob._identity.extend(list(flatten([next_blob.identity for next_blob in blob.next])))
    blob._identity = list(np.unique(blob._identity))

    for next_blob in blob.next:
        if next_blob.is_a_crossing:
            if len(next_blob.previous) != 1: # the next crossing_blob is splitting
                for next_blob_previous in next_blob.previous:
                    if next_blob_previous is not blob:
                        if next_blob_previous.is_a_fish and next_blob_previous.identity != 0 and next_blob_previous.identity in blob._identity:
                            blob._identity.remove(next_blob_previous.identity)
                        elif next_blob_previous.is_a_crossing and not next_blob_previous.bad_crossing:
                            [blob._identity.remove(identity) for identity in next_blob_previous.identity if identity in blob._identity]

    identities_to_remove_from_crossing = [blob_to_remove.identity for blob_to_remove in blobs_in_frame if blob_to_remove.is_a_fish]
    identities_to_remove_from_crossing.extend([0])
    [blob._identity.remove(identity) for identity in identities_to_remove_from_crossing if identity in blob._identity]
    if blob.bad_crossing:
        blob.number_of_animals_in_crossing = None
    else:
        blob.number_of_animals_in_crossing = len(blob.identity)
    return blob

def give_me_identities_in_crossings(list_of_blobs):
    """Sweep through the video frame by frame to get the identities of individuals in each crossing whenever it is possible
    """
    for frame_number, blobs_in_frame in enumerate(tqdm(list_of_blobs, desc = "getting identities in crossing")):
        ''' from past to future '''
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                blob = get_identities_in_crossing_forward(blob)

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs[::-1], desc = "getting identities in crossings")):
        ''' from future to past '''
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                blob = get_identities_in_crossing_backward(blob, blobs_in_frame)

    return list_of_blobs

def find_crossing_fragments(list_of_blobs):
    """we define a crossing fragment as a crossing that in subsequent frames
    involves the same individuals"""
    crossing_identifier = 0
    # get crossings in a fragment
    for blobs_in_frame in list_of_blobs:
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                print('crossing number ', crossing_identifier)
                print('frame number ', blob.frame_number)
                if len(blob.next) == 1 and len(blob.previous) == 1:
                    print("cur crossing id ", blob.identity)
                    print("prev crossing id ", blob.previous[0].identity)
                    print("next crossing id ", blob.next[0].identity)

                    blob.is_a_crossing_in_a_fragment = True
                    blob.crossing_identifier = crossing_identifier
                else:
                    print("-----------------------------------")
                    crossing_identifier += 1
            else:
                blob.is_a_crossing_in_a_fragment = False




if __name__ == "__main__":
    from GUI_utils import frame_by_frame_identity_inspector
    NUM_CHUNKS_BLOB_SAVING = 10

    #load video and list of blobs
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/video_object.npy').item()
    # video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_1/preprocessing/blobs_collection.npy'
    # list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    blobs = give_me_identities_in_crossings(blobs)
    crossing_identifier = find_crossing_fragments(blobs)

    # blobs_list = ListOfBlobs(blobs_in_video = blobs, path_to_save = video.blobs_path)
    # blobs_list.generate_cut_points(NUM_CHUNKS_BLOB_SAVING)
    # blobs_list.cut_in_chunks()
    # blobs_list.save()
    # frame_by_frame_identity_inspector(video, blobs)
