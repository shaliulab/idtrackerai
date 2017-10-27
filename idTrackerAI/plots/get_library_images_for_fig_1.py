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

from video import Video
from list_of_blobs import ListOfBlobs
from blob import Blob
from list_of_fragments import ListOfFragments
from fragment import Fragment
from GUI_utils import selectDir
from py_utils import get_spaced_colors_util

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

def save_video_frame(video, save_folder, frame_number = 0):

    cap = cv2.VideoCapture(video.video_path)
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cv2.imwrite(os.path.join(save_folder, 'frame_%i.png' %frame_number), frame[:, :int(frame.shape[1]/2)])

def save_preprocesing_images(video, save_folder, list_of_blobs_segmented, list_of_blobs, frame_number = 0):

    for blob_segmented, blob in zip(list_of_blobs_segmented.blobs_in_video[frame_number], list_of_blobs.blobs_in_video[frame_number]):
        folder_to_save_for_paper_figure = os.path.join(save_folder, 'identity_%i' %blob.final_identity)
        print("***\n %i" %blob.final_identity)
        if not os.path.isdir(folder_to_save_for_paper_figure):
            os.makedirs(folder_to_save_for_paper_figure)
        blob_segmented.get_image_for_identification(video, folder_to_save_for_paper_figure = folder_to_save_for_paper_figure)


if __name__ == '__main__':
    number_of_images = 20
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    video = np.load(video_path).item(0)
    save_folder = os.path.join(video.session_folder, 'identification_images_and_video_frame')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    # save frame snapshot
    save_video_frame(video, save_folder, frame_number = 0)
    # save preprocessing images
    list_of_blobs_segmented = ListOfBlobs.load(video.blobs_path_segmented)
    list_of_blobs = ListOfBlobs.load(video.blobs_path)
    save_preprocesing_images(video, save_folder, list_of_blobs_segmented, list_of_blobs, frame_number = 0)

    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_fragments_dictionaries = np.load(os.path.join(video.accumulation_folder,'light_list_of_fragments.npy'))
    fragments = [Fragment(number_of_animals = video.number_of_animals) for fragment_dictionary in list_of_fragments_dictionaries]
    [fragment.__dict__.update(fragment_dictionary) for fragment, fragment_dictionary in zip(fragments, list_of_fragments_dictionaries)]
    light_list_of_fragments = ListOfFragments(video, fragments)
    save_identification_images(video, list_of_fragments, number_of_images)
