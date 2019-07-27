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

import sys

import cv2
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from confapp import conf

# from idtrackerai.utils.py_utils import *
from idtrackerai.utils.py_utils import set_mkl_to_single_thread, set_mkl_to_multi_thread

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.segmentation_utils")


"""
The utilities to segment and extract the blob information
"""


def sum_frames_for_bkg_per_episode_in_single_file_video(starting_frame,
                                                        ending_frame,
                                                        video_path, bkg):
    """Computes the sum of frames (1 every 100 frames) for a particular episode of
    the video when the video is a single file.

    Parameters
    ----------
    starting_frame : int
        First frame of the episode
    ending_frame : int
        Last frame of the episode
    video_path : string
        Path to the single file of the video
    bkg : nd.array
        Zeros array with same width and height as the frame of the video.

    Returns
    -------
    bkg : nd.array
        Array with same width and height as the frame of the video. Contains the
        sum of (ending_frame - starting_frame) / 100 frames for the given episode
    number_of_frames_for_bkg_in_episode : int
        Number of frames used to compute the background in the current episode
    """
    cap = cv2.VideoCapture(video_path)
    logger.debug('Adding from starting frame %i to background' %starting_frame)
    number_of_frames_for_bkg_in_episode = 0
    frameInds = range(starting_frame,ending_frame, conf.BACKGROUND_SUBTRACTION_PERIOD)
    for ind in frameInds:
        logger.debug('Frame %i' %ind)
        cap.set(1,ind)
        ret, frameBkg = cap.read()
        if conf.SIGMA_GAUSSIAN_BLURRING is not None:
            frameBkg = cv2.GaussianBlur(frameBkg, (0, 0), conf.SIGMA_GAUSSIAN_BLURRING)
        if ret:
            gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
            gray = np.true_divide(gray,np.mean(gray))
            bkg = bkg + gray
            number_of_frames_for_bkg_in_episode += 1

    cap.release()
    return bkg, number_of_frames_for_bkg_in_episode


def sum_frames_for_bkg_per_episode_in_multiple_files_video(video_path, bkg):
    """Computes the sum of frames (1 every 100 frames) for a particular episode of
    the video when the video is splitted in several files

    Parameters
    ----------
    video_path : string
        Path to the file of the episode to be added to the background
    bkg : nd.array
        Zeros array with same width and height as the frame of the video.

    Returns
    -------
    bkg : nd.array
        Array with same width and height as the frame of the video. Contains the
        sum of (ending_frame - starting_frame) / 100 frames for the given episode
    number_of_frames_for_bkg_in_episode : int
        Number of frames used to compute the background in the current episode
    """
    logger.debug('Adding video %s to background' % video_path)
    cap = cv2.VideoCapture(video_path)
    numFrame = int(cap.get(7))
    number_of_frames_for_bkg_in_episode = 0
    frameInds = range(0,numFrame, conf.BACKGROUND_SUBTRACTION_PERIOD)
    for ind in frameInds:
        cap.set(1,ind)
        ret, frameBkg = cap.read()
        if conf.SIGMA_GAUSSIAN_BLURRING is not None:
            frameBkg = cv2.GaussianBlur(frameBkg, (0, 0), conf.SIGMA_GAUSSIAN_BLURRING)
        if ret:
            gray = cv2.cvtColor(frameBkg, cv2.COLOR_BGR2GRAY)
            gray = np.true_divide(gray,np.mean(gray))
            bkg = bkg + gray
            number_of_frames_for_bkg_in_episode += 1

    return bkg, number_of_frames_for_bkg_in_episode


def cumpute_background(video):
    """Computes a model of the background by averaging multiple frames of the video.
    In particular 1 every 100 frames is used for the computation.

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading

    Returns
    -------
    bkg : nd.array
        Array with the model of the background.

    See Also
    --------
    sum_frames_for_bkg_per_episode_in_single_file_video
    sum_frames_for_bkg_per_episode_in_multiple_files_video
    """
    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    bkg = np.zeros((video.original_height, video.original_width))

    set_mkl_to_single_thread()
    if video.paths_to_video_segments is None: # one single file
        logger.debug('one single video, computing bkg in parallel from single video')
        output = Parallel(n_jobs=conf.NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION)(delayed(
                    sum_frames_for_bkg_per_episode_in_single_file_video)(
                    starting_frame, ending_frame, video.video_path, bkg)
                    for (starting_frame, ending_frame) in video.episodes_start_end)
        logger.debug('Finished parallel loop for bkg subtraction')
    else: # multiple video files
        logger.debug('multiple videos, computing bkg in parallel from every episode')
        output = Parallel(n_jobs=conf.NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION)(delayed(
                    sum_frames_for_bkg_per_episode_in_multiple_files_video)(
                    videoPath,bkg) for videoPath in video.paths_to_video_segments)
        logger.debug('Finished parallel loop for bkg subtraction')
    set_mkl_to_multi_thread()

    partialBkg = [bkg for (bkg,_) in output]
    totNumFrame = np.sum([numFrame for (_,numFrame) in output])
    bkg = np.sum(np.asarray(partialBkg),axis=0)
    bkg = np.true_divide(bkg, totNumFrame)
    return bkg.astype('float32')

def segment_frame(frame, min_threshold, max_threshold, bkg, ROI, useBkg):
    """Applies the intensity thresholds (`min_threshold` and `max_threshold`) and the
    mask (`ROI`) to a given frame. If `useBkg` is True, the background subtraction
    operation is applied before thresholding with the given `bkg`.

    Parameters
    ----------
    frame : nd.array
        Frame to be segmented
    min_threshold : int
        Minimum intensity threshold for the segmentation (value from 0 to 255)
    max_threshold : int
        Maximum intensity threshold for the segmentation (value from 0 to 255)
    bkg : nd.array
        Background model to be used in the background subtraction operation
    ROI : nd.array
        Mask to be applied after thresholding. Ones in the array are pixels to be
        considered, zeros are pixels to be discarded.
    useBkg : bool
        Flag indicating whether background subtraction must be performed or not

    Returns
    -------
    frame_segmented_and_masked : nd.array
        Frame with zeros and ones after applying the thresholding and the mask.
        Pixels with value 1 are valid pixels given the thresholds and the mask.
    """
    if useBkg:
        frame = cv2.absdiff(bkg,frame) #only step where frame normalization is important, because the background is normalised
        p99 = np.percentile(frame, 99.95)*1.001
        frame = np.clip(255 - frame * (255.0/p99), 0, 255)
        frame_segmented = cv2.inRange(frame, min_threshold, max_threshold) #output: 255 in range, else 0
    elif not useBkg:
        p99 = np.percentile(frame, 99.95)*1.001
        frame_segmented = cv2.inRange(np.clip(frame * (255.0/p99), 0, 255), min_threshold, max_threshold) #output: 255 in range, else 0
    frame_segmented_and_masked = cv2.bitwise_and(frame_segmented,frame_segmented, mask=ROI) #Applying the mask
    return frame_segmented_and_masked


def filter_contours_by_area(contours, min_area, max_area):
    """Filters out contours which number of pixels is smaller than `min_area` or
    greater than `max_area`

    Parameters
    ----------
    contours : list
        List of OpenCV contours
    min_area : int
        Minimum number of pixels for a contour to be acceptable
    max_area : int
        Maximum number of pixels for a contours to be acceptable

    Returns
    -------
    good_contours : list
        List of OpenCV contours that fulfill both area thresholds
    """

    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            good_contours.append(contour)
    return good_contours


def cnt2BoundingBox(cnt,bounding_box):
    """Transforms the coordinates of the contour in the full frame to the the
    bounding box of the blob.

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the full
        frame of the video
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))


    Returns
    -------
    contour_in_bounding_box : nd.array
        Array with the pairs of coordinates of the contour in the bounding box
    """
    return cnt - np.asarray([bounding_box[0][0],bounding_box[0][1]])


def get_bounding_box(cnt, width, height, crossing_detector = False):
    """Computes the bounding box of a given contour with an extra margin of 45
    pixels. If the function is called with the crossing_detector set to True the
    margin of the bounding box is set to 55.

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the full
        frame of the video
    width : int
        Width of the video frame
    height : int
        Height of the video frame
    crossing_detector : bool
        Flag to indicate whether the function is being called from the crossing_detector module

    Returns
    -------
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))
    original_diagonal : int
        Diagonal of the original bounding box computed with OpenCv that serves as
        an estimate for the body length of the animal.
    """
    x,y,w,h = cv2.boundingRect(cnt)
    original_diagonal = int(np.ceil(np.sqrt(w**2 + h**2)))
    n = 45 if not crossing_detector else 55
    if x - n > 0: # We only expand the
        x = x - n
    else:
        x = 0
    if y - n > 0:
        y = y - n
    else:
        y = 0
    if x + w + 2*n < width:
        w = w + 2*n
    else:
        w = width - x
    if y + h + 2*n < height:
        h = h + 2*n
    else:
        h = height - y
    return ((x, y),(x + w, y + h)), original_diagonal


def getCentroid(cnt):
    """Computes the centroid of the contour

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the full
        frame of the video

    Returns
    -------
    centroid : tuple
        (x,y) coordinates of the center of mass of the contour.
    """
    M = cv2.moments(cnt)
    x = M['m10']/M['m00']
    y = M['m01']/M['m00']
    return (x,y)


def get_pixels(cnt, width, height):
    """Gets the coordinates list of the pixels inside the contour

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in a give
        width and height (it can either be the video width and heigh or the
        bounding box width and height)
    width : int
        Width of the frame
    height : int
        Height of the frame

    Returns
    -------
    pixels_coordinates_list : list
        List of the coordinates of the pixels in a given width and height
    """
    cimg = np.zeros((height, width))
    cv2.drawContours(cimg, [cnt], -1, color=255, thickness = -1)
    pts = np.where(cimg == 255)
    return np.asarray(list(zip(pts[0],pts[1])))


def get_bounding_box_image(frame, cnt, save_pixels, save_segmentation_image):
    """Computes the `bounding_box_image`from a given frame and contour. It also
    returns the coordinates of the `bounding_box`, the ravelled `pixels` inside of
    the contour and the diagonal of the `bounding_box` as an `estimated_body_length`

    Parameters
    ----------
    frame : nd.array
        frame from where to extract the `bounding_box_image`
    cnt : list
        List of the coordinates that defines the contour of the blob in the full
        frame of the video

    Returns
    -------
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))
    bounding_box_image : nd.array
        Part of the `frame` defined by the coordinates in `bounding_box`
    pixels_in_full_frame_ravelled : list
        List of ravelled pixels coordinates inside of the given contour
    estimated_body_length : int
        Estimated length of the contour in pixels.

    See Also
    --------
    get_bounding_box
    cnt2BoundingBox
    get_pixels
    """
    height = frame.shape[0]
    width = frame.shape[1]
    bounding_box, estimated_body_length = get_bounding_box(cnt, width, height) # the estimated body length is the diagonal of the original bounding_box
    if save_segmentation_image == 'RAM' or save_segmentation_image == 'DISK':
        bounding_box_image = frame[bounding_box[0][1]:bounding_box[1][1],
                                bounding_box[0][0]:bounding_box[1][0]]
    elif save_segmentation_image == 'NONE':
        bounding_box_image = None
    contour_in_bounding_box = cnt2BoundingBox(cnt, bounding_box)
    if save_pixels == 'RAM' or save_pixels == 'DISK':
        pixels_in_bounding_box = get_pixels(contour_in_bounding_box,
                                np.abs(bounding_box[0][0] - bounding_box[1][0]),
                                np.abs(bounding_box[0][1] - bounding_box[1][1]))
        pixels_in_full_frame = pixels_in_bounding_box + \
                                np.asarray([bounding_box[0][1], bounding_box[0][0]])
        pixels_in_full_frame_ravelled = np.ravel_multi_index(
                                        [pixels_in_full_frame[:,0], pixels_in_full_frame[:,1]],
                                        (height,width))
    elif save_pixels == 'NONE':
        pixels_in_full_frame_ravelled = None

    return bounding_box, bounding_box_image, pixels_in_full_frame_ravelled, estimated_body_length


def get_blobs_information_per_frame(frame, contours, save_pixels,
                                    save_segmentation_image):
    """Computes a set of properties for all the `contours` in a given frame.

    Parameters
    ----------
    frame : nd.array
        Frame from where to extract the `bounding_box_image` of every contour
    contours : list
        List of OpenCV contours for which to compute the set of properties

    Returns
    -------
    bounding_boxes : list
        List with the `bounding_box` for every contour in `contours`
    bounding_box_images : list
        List with the `bounding_box_image` for every contour in `contours`
    centroids : list
        List with the `centroid` for every contour in `contours`
    areas : list
        List with the `area` in pixels for every contour in `contours`
    pixels : list
        List with the `pixels` for every contour in `contours`
    estimated_body_lengths : list
        List with the `estimated_body_length` for every contour in `contours`

    See Also
    --------
    get_bounding_box_image
    getCentroid
    get_pixels
    """
    bounding_boxes = []
    bounding_box_images = []
    centroids = []
    areas = []
    pixels = []
    estimated_body_lengths = []

    for i, cnt in enumerate(contours):
        bounding_box, \
        bounding_box_image, \
        pixels_in_full_frame_ravelled, \
        estimated_body_length = get_bounding_box_image(frame, cnt, save_pixels,
                                                       save_segmentation_image)
        #bounding boxes
        bounding_boxes.append(bounding_box)
        # bounding_box_images
        bounding_box_images.append(bounding_box_image)
        # centroids
        centroids.append(getCentroid(cnt))
        areas.append(cv2.contourArea(cnt))
        # pixels lists
        pixels.append(pixels_in_full_frame_ravelled)
        # estimated body lengths list
        estimated_body_lengths.append(estimated_body_length)

    return bounding_boxes, bounding_box_images, centroids, areas, \
        pixels, estimated_body_lengths


def blob_extractor(segmented_frame, frame, min_area, max_area, save_pixels='DISK',
                   save_segmentation_image='DISK'):
    """Given a `segmented_frame` it extracts the blobs with area greater than
    `min_area` and smaller than `max_area` and it computes a set of relevant
    properties for every blob.

    Parameters
    ----------
    segmented_frame : nd.array
        Frame with zeros and ones where ones are valid pixels.
    frame : nd.array
        Frame from where to extract the `bounding_box_image` of every blob
    min_area : int
        Minimum number of blobs above which a blob is considered to be valid
    max_area : int
        Maximum number of blobs below which a blob is considered to be valid

    Returns
    -------
    bounding_boxes : list
        List with the `bounding_box` for every contour in `contours`
    bounding_box_images : list
        List with the `bounding_box_image` for every contour in `contours`
    centroids : list
        List with the `centroid` for every contour in `contours`
    areas : list
        List with the `area` in pixels for every contour in `contours`
    pixels : list
        List with the `pixels` for every contour in `contours`
    good_contours_in_full_frame:
        List with the `contours` which area is in between `min_area` and `max_area`
    estimated_body_lengths : list
        List with the `estimated_body_length` for every contour in `contours`

    See Also
    --------
    filter_contours_by_area
    get_blobs_information_per_frame
    """
    _, contours, hierarchy = cv2.findContours(segmented_frame,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_NONE)
    # Filter contours by size
    good_contours_in_full_frame = filter_contours_by_area(contours,
                                                          min_area,
                                                          max_area)
    # get contours properties
    bounding_boxes, bounding_box_images, \
        centroids, areas, pixels, \
        estimated_body_lengths = \
        get_blobs_information_per_frame(frame, good_contours_in_full_frame,
                                        save_pixels, save_segmentation_image)

    return bounding_boxes, bounding_box_images, centroids, areas, pixels, \
        good_contours_in_full_frame, estimated_body_lengths
