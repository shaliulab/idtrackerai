# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os

import numpy as np
import cv2
import h5py
from confapp import conf
import matplotlib.pyplot as plt

from idtrackerai.blob import Blob
from idtrackerai.utils.segmentation_utils import blob_extractor

''' erosion '''


def compute_erosion_disk(video, blobs_in_video):
    min_frame_distance_transform = []
    for blobs_in_frame in blobs_in_video:
        if len(blobs_in_frame) > 0:
            min_frame_distance_transform.append(compute_min_frame_distance_transform(video, blobs_in_frame))

    return np.ceil(np.nanmedian(min_frame_distance_transform)).astype(np.int)
    # return np.ceil(np.nanmedian([compute_min_frame_distance_transform(video, blobs_in_frame)
    #                              for blobs_in_frame in blobs_in_video
    #                              if len(blobs_in_frame) > 0])).astype(np.int)


def compute_min_frame_distance_transform(video, blobs_in_frame):
    max_distance_transform = []
    for blob in blobs_in_frame:
        if blob.is_an_individual:
            try:
                max_distance_transform.append(compute_max_distance_transform(video, blob))
            except:
                print("WARNING: Could not compute distance transform for this blob")

    # max_distance_transform = [compute_max_distance_transform(video, blob)
    #                           for blob in blobs_in_frame
    #                           if blob.is_an_individual]
    return np.min(max_distance_transform) if len(max_distance_transform) > 0 else np.nan


def generate_temp_image(video, pixels, bounding_box_in_frame_coordinates):
    pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
    pxs = np.array([pxs[:, 0] - bounding_box_in_frame_coordinates[0][1],
                    pxs[:, 1] - bounding_box_in_frame_coordinates[0][0]])
    temp_image = np.zeros((bounding_box_in_frame_coordinates[1][1] -
                            bounding_box_in_frame_coordinates[0][1],
                            bounding_box_in_frame_coordinates[1][0] -
                            bounding_box_in_frame_coordinates[0][0])).astype('uint8')
    temp_image[pxs[0,:], pxs[1,:]] = 255
    return temp_image


def compute_max_distance_transform(video, blob):
    temp_image = generate_temp_image(video, blob.pixels, blob.bounding_box_in_frame_coordinates)
    return np.max(cv2.distanceTransform(temp_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))

def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image,kernel,iterations = 1)


def get_eroded_blobs(video, blobs_in_frame, frame_number):
    episode = video.in_which_episode(frame_number)
    pixels_path = None
    if conf.SAVE_PIXELS == 'DISK':
        pixels_path = os.path.join(video._segmentation_data_folder,
                                   'episode_pixels_{}.hdf5'.format(str(episode)))

    # logger.debug('Getting eroded blobs')
    segmented_frame = np.zeros((video.height, video.width)).astype('uint8')

    for blob in blobs_in_frame:
        pixels = blob.eroded_pixels if (hasattr(blob, 'has_eroded_pixels') and blob.has_eroded_pixels) else blob.pixels
        pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
        segmented_frame[pxs[:,0], pxs[:,1]] = 255

    segmented_eroded_frame = erode(segmented_frame, video.erosion_kernel_size)
    boundingBoxes, _, centroids, _, pixels_all, contours, _ = blob_extractor(segmented_eroded_frame, segmented_eroded_frame, 0, np.inf)
    # logger.debug('Finished getting eroded blobse')
    eroded_blobs_in_frame = []
    for i, (centroid, contour, pixels, bounding_box) in enumerate(zip(centroids, contours, pixels_all, boundingBoxes)):
        eroded_blob = Blob(centroid,
                           contour,
                           None,
                           bounding_box,
                           bounding_box_image=None,
                           number_of_animals=video.number_of_animals,
                           frame_number=frame_number,
                           pixels=None,
                           pixels_path=pixels_path,
                           in_frame_index=i,
                           video_height=video.height,
                           video_width=video.width,
                           video_path=video.video_path,
                           pixels_are_from_eroded_blob=True,
                           resolution_reduction=video.resolution_reduction)
        eroded_blob.eroded_pixels = pixels
        eroded_blobs_in_frame.append(eroded_blob)


    return eroded_blobs_in_frame
