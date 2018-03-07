from __future__ import absolute_import, division, print_function
import cv2
import numpy as np

from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.video import Video

def put_pixels_in_frame(frame, blob, color):
    pxs = np.array(np.unravel_index(blob.pixels,(video.height, video.width))).T
    if color == 'r':
        frame[pxs[:,0], pxs[:,1], 1] -= 50
        frame[pxs[:,0], pxs[:,1], 2] -= 50
    elif color == 'g':
        frame[pxs[:,0], pxs[:,1], 0] -= 50
        frame[pxs[:,0], pxs[:,1], 2] -= 50
    elif color == 'k':
        frame[pxs[:,0], pxs[:,1], :] -= 50

    return frame


if __name__ == '__main__':

    print("loading list_of_blobs and video_object")
    path_to_list_of_blobs = '/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_test/preprocessing/blobs_collection_no_gaps.npy'
    path_to_video_object = '/home/chronos/Desktop/IdTrackerDeep/videos/conflicto_short/session_test/video_object.npy'
    video = np.load(path_to_video_object).item()
    list_of_blobs = np.load(path_to_list_of_blobs).item()

    frame = np.ones((video.height, video.width, 3)) * 255

    print("plottig frame")
    for frame_number in range(44,56,2):
        blobs_in_frame = list_of_blobs.blobs_in_video[frame_number]
        for blob in blobs_in_frame:
            if isinstance(blob.assigned_identity, int):
                if blob.assigned_identity == 1:
                    frame = put_pixels_in_frame(frame, blob, 'r')
                elif blob.assigned_identity == 4:
                    frame = put_pixels_in_frame(frame, blob, 'g')
            elif isinstance(blob.assigned_identity, list) and (1 in blob.assigned_identity or 4 in blob.assigned_identity):
                print("pass")
                frame = put_pixels_in_frame(frame, blob, 'k')

    cv2.imwrite('frame_all.png', frame)
