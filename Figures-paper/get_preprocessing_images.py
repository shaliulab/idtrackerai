import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn
seaborn.set_style('white')
import sys
sys.path.append('../idTrackerAI')
sys.path.append('../idTrackerAI/preprocessing')
sys.path.append('../idTrackerAI/utils')
import os
from video import Video
from blob import Blob, ListOfBlobs


if __name__ == "__main__":
    #load video and list of blobs
    video = np.load( '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_4/video_object.npy' ).item()
    # video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/video_object.npy').item()
    number_of_animals = video.number_of_animals
    list_of_blobs_path = video._blobs_path
    # list_of_blobs_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_1/preprocessing/blobs_collection.npy'
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)
    list_of_segmented_blobs_path = video._blobs_path_segmented
    list_of_segmented_blobs = ListOfBlobs.load(list_of_segmented_blobs_path)
    blobs = list_of_blobs.blobs_in_video
    segmented_blobs = list_of_segmented_blobs.blobs_in_video
    plt.ion()
    plt.figure(0)
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    figs_path = os.path.join(video._session_folder, 'preprocessing_images')
    if not os.path.isdir(figs_path):
        os.makedirs(figs_path)

    for i in range(10):
        blobs_in_frame = blobs[i]
        segmented_blobs_in_frame = segmented_blobs[i]
        for blob, segmented_blob in zip(blobs_in_frame, segmented_blobs_in_frame):
            if not blob.is_a_crossing:
                ax1.imshow(blob.portrait)
                ax1.set_yticks([])
                ax1.set_xticks([])
                ax2.imshow(segmented_blob.bounding_box_image)
                ax2.set_yticks([])
                ax2.set_xticks([])
                # plt.show()
                fig_name = 'frame_' + str(blob.frame_number) + '_blob_number_' + str(blob.blob_index) + '.pdf'
                fig_path = os.path.join(figs_path, fig_name)
                plt.savefig(fig_path)
                # _ = raw_input("Press [enter] to continue.")
