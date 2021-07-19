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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

from typing import Iterable, Optional, Tuple, List
import logging
from tqdm import tqdm

import cv2
import numpy as np
from confapp import conf
from joblib import Parallel, delayed

# TODO: importing video here for typing creates a circular import
# from idtrackerai.video import Video
from idtrackerai.utils.py_utils import (
    set_mkl_to_multi_thread,
    set_mkl_to_single_thread,
)

logger = logging.getLogger("__main__.segmentation_utils")


"""
The utilities to segment and extract the blob information
"""


def _update_bkg_stat(
    bkg: np.ndarray, gray: np.ndarray, stat: str
) -> np.ndarray:
    if stat == "mean":
        # We only sum the frames and divide by the number of samples
        # outside of the loop
        bkg = bkg + gray
    elif stat == "min":
        bkg = np.min(np.asarray([bkg, gray]), axis=0)
    elif stat == "max":
        bkg = np.max(np.asarray([bkg, gray]), axis=0)
    return bkg


def _compute_bkg_for_episode(
    cap: cv2.VideoCapture,
    bkg: np.ndarray,
    frames_range: Iterable,  # sample frames in the given episode
    mask: np.ndarray,  # values are 1 (valid) 0 (invalid)
    sigma: float,
    stat: str,
):
    number_of_sample_frames_in_episode = 0
    for ind in frames_range:
        logger.debug("Frame %i" % ind)
        cap.set(1, ind)
        ret, frame = cap.read()
        frame = gaussian_blur(frame, sigma=sigma)
        if ret:
            gray = to_gray_scale(frame)
            gray = gray / get_frame_average_intensity(gray, mask)
            bkg = _update_bkg_stat(bkg, gray, stat)
            number_of_sample_frames_in_episode += 1
    return bkg, number_of_sample_frames_in_episode


def _get_episode_frames_for_bkg(cap, starting_frame, ending_frame, period):
    if ending_frame is None:
        # ending_frame is None when the video is splitted in chunks
        ending_frame = int(cap.get(7))  # number of frames in video
        if period > ending_frame:
            # TODO: Find a better implementation that does not change the
            # effective BACKGROUND_SUBTRACTION_PERIOD when the video is
            # splitted in multiple files
            logger.warning(
                "In this video episode "
                "BACKGROUND_SUBTRACTION_PERIOD > num_frames in video file "
                f"({period} > {ending_frame}). "
                f"The effective period will be num_frames ({ending_frame})."
            )
    return range(starting_frame, ending_frame, period)


def _compute_episode_bkg(
    video_path: str,
    bkg: np.ndarray,
    mask: np.ndarray,
    period: int,
    stat: str = "mean",
    sigma: Optional[float] = None,
    starting_frame: Optional[int] = 0,
    ending_frame: Optional[int] = None,
) -> Tuple[np.ndarray, int]:

    cap = cv2.VideoCapture(video_path)

    frames_for_bkg = _get_episode_frames_for_bkg(
        cap, starting_frame, ending_frame, period
    )

    bkg, number_of_sample_frames_in_episode = _compute_bkg_for_episode(
        cap,
        bkg,
        frames_for_bkg,
        mask,
        sigma,
        stat,
    )

    cap.release()
    return bkg, number_of_sample_frames_in_episode


def compute_background(
    video_paths,
    original_height,
    original_width,
    video_path,
    original_ROI,
    episodes_start_end,
    background_sampling_period=conf.BACKGROUND_SUBTRACTION_PERIOD,
    background_subtraction_stat=conf.BACKGROUND_SUBTRACTION_STAT,
    parallel_period=conf.FRAMES_PER_EPISODE,
    num_jobs_parallel=conf.NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION,
    sigma_gaussian_blur=conf.SIGMA_GAUSSIAN_BLURRING,
):
    """
    Computes the background model by sampling frames from the video with a
    period `background_sampling_period` and computing the stat
    `background_subtraction_stat` across the sampled frames.
    If the video comes in a single file it computes the background in parallel
    splitting the video in `parallel_period`.
    This parameter is ignored if the the video comes in multiple files.

    Parameters
    ----------
    video : idtrackerai.video.VideoObject
    background_sampling_period : int
        sampling period to compute the background model
    background_subtraction_stat: str
        statistic to compute over the sampled frames ("mean", "min", or "max)
    parallel_period: int
        video chunk size (in frames) for the parallel computation
    num_jobs_parallel: int
        number of jobs for the parallel computation
    sigma_gaussian_blur: float
        sigma of the gaussian kernel to blur each frame

    Returns
    -------
    bkg : np.ndarray
        Background model

    """
    bkg_geq_episode_period = background_sampling_period >= parallel_period
    single_video_file = video_paths is None
    if single_video_file and bkg_geq_episode_period:
        logger.warning(
            f"BACKGROUND_SUBTRACTION_PERIOD "
            f"({background_sampling_period}) >= "
            f"FRAMES_PER_EPISODE ({parallel_period}): "
            f"This effectively makes "
            f"BACKGROUND_SUBTRACTION_PERIOD=FRAMES_PER_EPISODE."
        )
        logger.warning(
            "To get a higher BACKGROUND_SUBTRACTION_PERIOD make "
            "FRAMES_PER_EPISODE > BACKGROUND_SUBTRACTION_PERIOD"
        )

    # This holds even if we have not selected a ROI because then the ROI is
    # initialized as the full frame
    if background_subtraction_stat in ["mean", "max"]:
        bkg = np.zeros((original_height, original_width))
    else:
        bkg = np.ones((original_height, original_width)) * 10

    set_mkl_to_single_thread()
    if video_paths is None:  # one single file
        logger.debug(
            "one single video, computing bkg in parallel from single video"
        )
        output = Parallel(n_jobs=num_jobs_parallel)(
            delayed(_compute_episode_bkg)(
                video_path,
                bkg,
                original_ROI,
                background_sampling_period,
                stat=background_subtraction_stat,
                sigma=sigma_gaussian_blur,
                starting_frame=starting_frame,
                ending_frame=ending_frame,
            )
            for (starting_frame, ending_frame) in tqdm(
                episodes_start_end, desc="Computing background model"
            )
        )
        logger.debug("Finished parallel loop for bkg subtraction")
    else:  # multiple video files
        logger.debug(
            "multiple videos, computing bkg in parallel from every episode"
        )
        output = Parallel(n_jobs=num_jobs_parallel)(
            delayed(_compute_episode_bkg)(
                video_path,
                bkg,
                original_ROI,
                background_sampling_period,
                stat=background_subtraction_stat,
                sigma=sigma_gaussian_blur,
            )
            for video_path in tqdm(
                video_paths,
                desc="Computing bakcground model",
            )
        )
        logger.debug("Finished parallel loop for bkg subtraction")
    set_mkl_to_multi_thread()

    logger.info(
        f"Computing background with stat={background_subtraction_stat} "
        f"and period={background_sampling_period} frames"
    )
    partial_bkg = np.asarray([bkg for (bkg, _) in output])
    if background_subtraction_stat == "mean":
        num_samples_bkg = np.sum([numFrame for (_, numFrame) in output])
        bkg = np.sum(partial_bkg, axis=0)
        bkg = bkg / num_samples_bkg
    elif background_subtraction_stat == "min":
        bkg = np.min(partial_bkg, axis=0)
    elif background_subtraction_stat == "max":
        bkg = np.max(partial_bkg, axis=0)

    return bkg.astype("float32")


def gaussian_blur(frame, sigma=None):
    if sigma is not None and sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), sigma)
    return frame


def to_gray_scale(frame):
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def get_frame_average_intensity(frame: np.ndarray, mask: np.ndarray):
    """Computes the average intensity of a given frame considering the maks.
    Only pixels with values
    different than zero in the mask are considered to compute the average
    intensity

    Parameters
    ----------
    frame : nd.array
        Frame from which to compute the average intensity
    mask : nd.array
        Mask to be applied. Pixels with value 0 will be ignored to compute the
        average intensity.

    Returns
    -------

    """
    assert mask is not None
    assert frame.shape == mask.shape
    return np.float32(np.mean(np.ma.array(frame, mask=mask == 0)))


def segment_frame(frame, min_threshold, max_threshold, bkg, ROI, useBkg):
    """Applies the intensity thresholds (`min_threshold` and `max_threshold`)
    and the mask (`ROI`) to a given frame. If `useBkg` is True,
    the background subtraction operation is applied before
    thresholding with the given `bkg`.

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
        Mask to be applied after thresholding. Ones in the array are pixels to
        be considered, zeros are pixels to be discarded.
    useBkg : bool
        Flag indicating whether background subtraction must be performed or not

    Returns
    -------
    frame_segmented_and_masked : nd.array
        Frame with zeros and ones after applying the thresholding and the mask.
        Pixels with value 1 are valid pixels given the thresholds and the mask.
    """
    if useBkg:
        # only step where frame normalization is important,
        # because the background is normalised
        frame = cv2.absdiff(bkg, frame)
        p99 = np.percentile(frame, 99.95) * 1.001
        frame = np.clip(255 - frame * (255.0 / p99), 0, 255)
        frame_segmented = cv2.inRange(
            frame, min_threshold, max_threshold
        )  # output: 255 in range, else 0
    else:
        p99 = np.percentile(frame, 99.95) * 1.001
        frame_segmented = cv2.inRange(
            np.clip(frame * (255.0 / p99), 0, 255),
            min_threshold,
            max_threshold,
        )  # output: 255 in range, else 0
    frame_segmented_and_masked = cv2.bitwise_and(
        frame_segmented, frame_segmented, mask=ROI
    )  # Applying the mask
    return frame_segmented_and_masked


def _filter_contours_by_area(
    contours: List, min_area: int, max_area: int
) -> List[np.ndarray]:  # (cnt_points, 1, 2)
    """Filters out contours which number of pixels is smaller than `min_area`
    or greater than `max_area`

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


def _cnt2BoundingBox(cnt, bounding_box):
    """Transforms the coordinates of the contour in the full frame to the
    bounding box of the blob.

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))


    Returns
    -------
    contour_in_bounding_box : nd.array
        Array with the pairs of coordinates of the contour in the bounding box
    """
    return cnt - np.asarray([bounding_box[0][0], bounding_box[0][1]])


def _get_bounding_box(
    cnt: np.ndarray,
    width: int,
    height: int,
) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]:
    """Computes the bounding box of a given contour with an extra margin of
    constants.EXTRA_PIXELS_BBOX pixels.
    The extra margin is given so that the image can be rotated without adding
    artifacts in the borders. The image will be rotated when setting the
    identification image in the crossing detector step.

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video
    width : int
        Width of the video frame
    height : int
        Height of the video frame

    Returns
    -------
    bounding_box : tuple
        Tuple with the coordinates of the bounding box (x, y),(x + w, y + h))
    original_diagonal : int
        Diagonal of the original bounding box computed with OpenCv that serves
        as estimate for the body length of the animal.
    """
    # TODO: rethink whether the expansion is really needed
    x, y, w, h = cv2.boundingRect(cnt)
    original_diagonal = int(np.ceil(np.sqrt(w ** 2 + h ** 2)))
    n = conf.EXTRA_PIXELS_BBOX
    if x - n > 0:  # We only expand the
        x = x - n
    else:
        x = 0
    if y - n > 0:
        y = y - n
    else:
        y = 0
    if x + w + 2 * n < width:
        w = w + 2 * n
    else:
        w = width - x
    if y + h + 2 * n < height:
        h = h + 2 * n
    else:
        h = height - y
    expanded_bbox = ((x, y), (x + w, y + h))
    return expanded_bbox, original_diagonal


def _getCentroid(cnt):
    """Computes the centroid of the contour

    Parameters
    ----------
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video

    Returns
    -------
    centroid : tuple
        (x,y) coordinates of the center of mass of the contour.
    """
    M = cv2.moments(cnt)
    x = M["m10"] / M["m00"]
    y = M["m01"] / M["m00"]
    return (x, y)


def _get_pixels(cnt: np.ndarray, width: int, height: int) -> np.ndarray:
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
    cv2.drawContours(cimg, [cnt], -1, color=255, thickness=-1)
    pts = np.where(cimg == 255)
    return np.asarray(list(zip(pts[0], pts[1])))


def _get_bounding_box_image(
    frame: np.ndarray,
    cnt: np.ndarray,
    save_pixels: str,
    save_segmentation_image: str,
):
    """Computes the `bounding_box_image`from a given frame and contour. It also
    returns the coordinates of the `bounding_box`, the ravelled `pixels`
    inside of the contour and the diagonal of the `bounding_box` as
    an `estimated_body_length`

    Parameters
    ----------
    frame : nd.array
        frame from where to extract the `bounding_box_image`
    cnt : list
        List of the coordinates that defines the contour of the blob in the
        full frame of the video

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
    _get_bounding_box
    _cnt2BoundingBox
    _get_pixels
    """
    height = frame.shape[0]
    width = frame.shape[1]
    # Coordinates of an expanded bounding box
    bounding_box, estimated_body_length = _get_bounding_box(
        cnt, width, height
    )  # the estimated body length is the diagonal of the original bounding_box
    # Get bounding box from frame
    if save_segmentation_image == "RAM" or save_segmentation_image == "DISK":
        bounding_box_image = frame[
            bounding_box[0][1] : bounding_box[1][1],
            bounding_box[0][0] : bounding_box[1][0],
        ]
    elif save_segmentation_image == "NONE":
        bounding_box_image = None
    else:
        raise ValueError(
            f"Invalid `save_segmentation_image` = {save_segmentation_image}"
        )
    contour_in_bounding_box = _cnt2BoundingBox(cnt, bounding_box)
    if save_pixels == "RAM" or save_pixels == "DISK":
        pixels_in_bounding_box = _get_pixels(
            contour_in_bounding_box,
            np.abs(bounding_box[0][0] - bounding_box[1][0]),
            np.abs(bounding_box[0][1] - bounding_box[1][1]),
        )
        pixels_in_full_frame = pixels_in_bounding_box + np.asarray(
            [bounding_box[0][1], bounding_box[0][0]]
        )
        pixels_in_full_frame_ravelled = np.ravel_multi_index(
            [pixels_in_full_frame[:, 0], pixels_in_full_frame[:, 1]],
            (height, width),
        )
    elif save_pixels == "NONE":
        pixels_in_full_frame_ravelled = None
    else:
        raise

    return (
        bounding_box,
        bounding_box_image,
        pixels_in_full_frame_ravelled,
        estimated_body_length,
    )


def _get_blobs_information_per_frame(
    frame: np.ndarray,
    contours: List[np.ndarray],
    save_pixels: str,
    save_segmentation_image: str,
):
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
    _get_bounding_box_image
    _getCentroid
    _get_pixels
    """
    bounding_boxes = []
    bounding_box_images = []
    centroids = []
    areas = []
    pixels = []
    estimated_body_lengths = []

    for i, cnt in enumerate(contours):
        (
            bounding_box,
            bounding_box_image,
            pixels_in_full_frame_ravelled,
            estimated_body_length,
        ) = _get_bounding_box_image(
            frame, cnt, save_pixels, save_segmentation_image
        )
        # bounding boxes
        bounding_boxes.append(bounding_box)
        # bounding_box_images
        bounding_box_images.append(bounding_box_image)
        # centroids
        centroids.append(_getCentroid(cnt))
        areas.append(cv2.contourArea(cnt))
        # pixels lists
        pixels.append(pixels_in_full_frame_ravelled)
        # estimated body lengths list
        estimated_body_lengths.append(estimated_body_length)

    return (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        estimated_body_lengths,
    )


def blob_extractor(
    segmented_frame: np.ndarray,
    frame: np.ndarray,
    min_area: int,
    max_area: int,
    save_pixels: Optional[str] = "DISK",
    save_segmentation_image: Optional[str] = "DISK",
) -> Tuple[
    List[Tuple],
    List[np.ndarray],
    List[Tuple],
    List[int],
    List[List],
    List[List],
    List[float],
]:
    # TODO: Document
    _, contours, hierarchy = cv2.findContours(
        segmented_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    # Filter contours by size
    good_contours_in_full_frame = _filter_contours_by_area(
        contours, min_area, max_area
    )
    # get contours properties
    (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        estimated_body_lengths,
    ) = _get_blobs_information_per_frame(
        frame,
        good_contours_in_full_frame,
        save_pixels,
        save_segmentation_image,
    )

    return (
        bounding_boxes,
        bounding_box_images,
        centroids,
        areas,
        pixels,
        good_contours_in_full_frame,
        estimated_body_lengths,
    )
