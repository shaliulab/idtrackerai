from __future__ import absolute_import, division, print_function
import os
import sys
from glob import glob
sys.path.append('./')
sys.path.append('./utils')

import numpy as np
import cv2

from matplotlib import pyplot as plt
import seaborn as sns

from idtrackerai.video import Video
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.blob import Blob
from idtrackerai.list_of_fragments import ListOfFragments
from idtrackerai.fragment import Fragment
from idtrackerai.utils.GUI_utils import selectDir
from idtrackerai.utils.py_utils import  get_spaced_colors_util

def save_identification_images(video, list_of_fragments, number_of_images):

    for identity in range(1, video.number_of_animals + 1):

        fragment = [fragment for fragment in list_of_fragments.fragments if fragment.final_identity == identity and fragment.start_end[0] == 0][0]

        for image_number in range(number_of_images):
            fig, ax = plt.subplots(1,1)
            ax.imshow(fragment.images[image_number], cmap = 'gray')
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(left = True, right = True, top = True, bottom = True)
            fig.savefig(os.path.join(save_folder, 'identity_%i' %identity, '_image_%i.pdf' %image_number), transparent=True, bbox_inches='tight')

def get_crossings_detector_image(video, blob, downsampling_factor = .5):
    _, _, _, image = blob.get_image_for_identification(video)
    image = cv2.resize(image, None,
                            fx = downsampling_factor,
                            fy = downsampling_factor,
                            interpolation = cv2.INTER_CUBIC)
    return image

if __name__ == '__main__':
    number_of_images = 20
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    save_folder = os.path.join(video.session_folder, 'crossing_detector_images')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    list_of_blobs_segmented = ListOfBlobs.load(video, video.blobs_path_segmented)
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)

    individuals = [bs for bfs, bf in zip(list_of_blobs_segmented.blobs_in_video, list_of_blobs.blobs_in_video) for bs, b in zip(bfs, bf) if b.is_an_individual]
    crossings = [bs for bfs, bf in zip(list_of_blobs_segmented.blobs_in_video, list_of_blobs.blobs_in_video) for bs, b in zip(bfs, bf) if b.is_a_crossing]

    for image_number, individual in enumerate(individuals[:20]):
        fig, ax = plt.subplots(1,1)
        image = get_crossings_detector_image(video, individual)
        ax.imshow(image, cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(left = True, right = True, top = True, bottom = True)
        fig.savefig(os.path.join(save_folder, 'individual_%i.pdf' %image_number), transparent=True, bbox_inches='tight')

    for image_number, crossing in enumerate(crossings):
        fig, ax = plt.subplots(1,1)
        image = get_crossings_detector_image(video, crossing)
        ax.imshow(image, cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(left = True, right = True, top = True, bottom = True)
        fig.savefig(os.path.join(save_folder, 'crossing_%i.pdf' %image_number), transparent=True, bbox_inches='tight')
