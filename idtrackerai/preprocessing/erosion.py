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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)
 

from __future__ import absolute_import, print_function, division
import os
import sys
import numpy as np
import cv2
from idtrackerai.utils.video_utils import blob_extractor
from idtrackerai.blob import Blob

def get_eroded_blobs_with_pixels_in_original_blob(video, original_blob, eroded_blobs_in_frame):
    eroded_blobs_in_original_blob = [eroded_blob for eroded_blob in eroded_blobs_in_frame
                                if cv2.pointPolygonTest(original_blob.contour,
                                np.unravel_index(eroded_blob.pixels[0],(video.height, video.width))[::-1], False) >= 0]

    if len(eroded_blobs_in_original_blob) <= 1:
        return [original_blob]
    else:
        return eroded_blobs_in_original_blob

def compute_erosion_disk(video, blobs_in_video):
    return np.ceil(np.nanmedian([compute_min_frame_distance_transform(video, blobs_in_frame)
                    for blobs_in_frame in blobs_in_video if len(blobs_in_frame) > 0])).astype(np.int)

def compute_min_frame_distance_transform(video, blobs_in_frame):
    max_distance_transform = [compute_max_distance_transform(video, blob)
                    for blob in blobs_in_frame]
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
    return np.max(cv2.distanceTransform(temp_image, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE))

def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image,kernel,iterations = 1)

def get_eroded_blobs(video, blobs_in_frame):
    segmented_frame = np.zeros((video.height, video.width)).astype('uint8')

    for blob in blobs_in_frame:
        pixels = blob.eroded_pixels if hasattr(blob,'eroded_pixels') else blob.pixels
        pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
        segmented_frame[pxs[:,0], pxs[:,1]] = 255

    segmented_eroded_frame = erode(segmented_frame, video.erosion_kernel_size)
    bounding_boxes, miniframes, centroids, \
    areas, pixels, contours, estimated_body_lengths = blob_extractor(segmented_eroded_frame, segmented_eroded_frame, video.min_area*.55, np.inf)
    return [Blob(centroid,
                contour,
                area,
                bounding_box,
                bounding_box_image = miniframe,
                estimated_body_length = estimated_body_length,
                pixels = pixel,
                number_of_animals = video.number_of_animals)
                for bounding_box, miniframe, centroid, area, pixel, contour, estimated_body_length in zip(bounding_boxes, miniframes, centroids,
                                                                    areas, pixels, contours, estimated_body_lengths)]

def get_new_blobs_in_frame_after_erosion(video, blobs_in_frame, eroded_blobs_in_frame):
    new_blobs_in_frame = []

    for blob in blobs_in_frame:
        new_blobs = get_eroded_blobs_with_pixels_in_original_blob(video, blob, eroded_blobs_in_frame)
        new_blobs_in_frame.extend(new_blobs)

    return new_blobs_in_frame
