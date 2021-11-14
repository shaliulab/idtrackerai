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

from typing import Tuple, List, Dict, Optional
import gc
import logging
import multiprocessing
import os

import cv2
import h5py

from scipy import ndimage
from confapp import conf
from joblib import Parallel, delayed
from tqdm import tqdm

from idtrackerai.blob import Blob
from idtrackerai.utils.py_utils import (
    flatten,
    set_mkl_to_multi_thread,
    set_mkl_to_single_thread,
)
from idtrackerai.animals_detection.segmentation_utils import (
    blob_extractor,
    get_frame_average_intensity,
    segment_frame,
    to_gray_scale,
    gaussian_blur,
)
import idtrackerai.constants as cons

logger = logging.getLogger("__main__.segmentation")

"""
The segmentation module
"""


def _get_blobs_in_frame(
    cap,
    video_params_to_store,
    segmentation_parameters,
    max_number_of_blobs,
    global_frame_number,
    frame_number_in_video_path,
    bounding_box_images_path,
    video_path,
    pixels_path,
    save_pixels,
    save_segmentation_image,
):
    """Segments a frame read from `cap` according to the preprocessing parameters
    in `video`. Returns a list `blobs_in_frame` with the Blob objects in the frame
    and the `max_number_of_blobs` found in the video so far. Frames are segmented
    in gray scale.

    Parameters
    ----------
    cap : <VideoCapture object>
        OpenCV object used to read the frames of the video
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    segmentation_thresholds : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process
    frame_number : int
        Number of the frame being segmented. It is used to print in the terminal the frames
        where the segmentation fails. This frame is the frame of the episode if the video
        is chuncked.
    global_frame_number : int
        This is the frame number in the whole video. It will be different to the frame_number
        if the video is chuncked.


    Returns
    -------
    blobs_in_frame : list
        List of <Blob object> segmented in the current frame
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process

    See Also
    --------
    Video
    Blob
    segment_frame
    blob_extractor
    """
    ret, frame = cap.read()
    (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
    ) = _process_frame(
        frame,
        segmentation_parameters,
        global_frame_number,
        save_pixels,
        save_segmentation_image,
    )

    blobs_in_frame = _create_blobs_objects(
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
        save_segmentation_image,
        bounding_box_images_path,
        save_pixels,
        pixels_path,
        global_frame_number,
        frame_number_in_video_path,
        video_params_to_store,
        video_path,
        segmentation_parameters,
    )

    max_number_of_blobs = max(max_number_of_blobs, len(centroids))

    return blobs_in_frame, max_number_of_blobs


def _process_frame(
    frame,
    segmentation_parameters,
    frame_number,
    save_pixels,
    save_segmentation_image,
):

    try:
        frame = gaussian_blur(frame, sigma=conf.SIGMA_GAUSSIAN_BLURRING)
        bkg = segmentation_parameters["bkg_model"]
        mask = segmentation_parameters["mask"]

        # Apply resolution reduction
        if segmentation_parameters["resolution_reduction"] != 1:
            frame = cv2.resize(
                frame,
                None,
                fx=segmentation_parameters["resolution_reduction"],
                fy=segmentation_parameters["resolution_reduction"],
                interpolation=cv2.INTER_AREA,
            )
            if bkg is not None:
                bkg = cv2.resize(
                    bkg,
                    None,
                    fx=segmentation_parameters["resolution_reduction"],
                    fy=segmentation_parameters["resolution_reduction"],
                    interpolation=cv2.INTER_AREA,
                )
            if mask is not None:
                mask = cv2.resize(
                    mask,
                    None,
                    fx=segmentation_parameters["resolution_reduction"],
                    fy=segmentation_parameters["resolution_reduction"],
                    interpolation=cv2.INTER_AREA,
                )
        # Convert the frame to gray scale
        gray = to_gray_scale(frame)
        # Normalize frame
        normalized_framed = gray / get_frame_average_intensity(gray, mask)
        # Binarize frame
        segmentedFrame = segment_frame(
            normalized_framed,
            segmentation_parameters["min_threshold"],
            segmentation_parameters["max_threshold"],
            bkg,
            mask,
            segmentation_parameters["subtract_bkg"],
        )
        # Fill holes in the segmented frame to avoid duplication of contours
        segmentedFrame = ndimage.binary_fill_holes(segmentedFrame).astype(
            "uint8"
        )
        # Extract blobs info
        (
            bounding_boxes,
            miniframes,
            centroids,
            areas,
            pixels,
            contours,
            estimated_body_lengths,
        ) = blob_extractor(
            segmentedFrame,
            gray,
            segmentation_parameters["min_area"],
            segmentation_parameters["max_area"],
            save_pixels,
            save_segmentation_image,
        )
    except Exception as e:
        print(f"Frame {frame_number}: {e}")
        logger.info(
            "An error occurred while reading frame number : %i" % frame_number
        )
        bounding_boxes = []
        miniframes = []
        centroids = []
        areas = []
        pixels = []
        contours = []
        estimated_body_lengths = []

    return (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
    )


def _create_blobs_objects(
    bounding_boxes,
    miniframes,
    centroids,
    areas,
    pixels,
    contours,
    estimated_body_lengths,
    save_segmentation_image,
    bounding_box_images_path,
    save_pixels,
    pixels_path,
    global_frame_number,
    frame_number_in_video_path,
    video_params_to_store,
    video_path,
    segmentation_parameters,
):
    blobs_in_frame = []
    # create blob objects
    for i, bounding_box in enumerate(bounding_boxes):
        if save_segmentation_image == "DISK":
            with h5py.File(bounding_box_images_path, "a") as f1:
                f1.create_dataset(
                    str(global_frame_number) + "-" + str(i), data=miniframes[i]
                )
            miniframes[i] = None
        if save_pixels == "DISK":
            with h5py.File(pixels_path, "a") as f2:
                f2.create_dataset(
                    str(global_frame_number) + "-" + str(i), data=pixels[i]
                )
            pixels[i] = None

        blob = Blob(
            centroids[i],
            contours[i],
            areas[i],
            bounding_box,
            bounding_box_image=miniframes[i],
            bounding_box_images_path=bounding_box_images_path,
            estimated_body_length=estimated_body_lengths[i],
            number_of_animals=video_params_to_store["number_of_animals"],
            frame_number=global_frame_number,
            pixels=pixels[i],
            pixels_path=pixels_path,
            in_frame_index=i,
            video_height=video_params_to_store["height"],
            video_width=video_params_to_store["width"],
            video_path=video_path,
            frame_number_in_video_path=frame_number_in_video_path,
            resolution_reduction=segmentation_parameters[
                "resolution_reduction"
            ],
        )
        blobs_in_frame.append(blob)

    return blobs_in_frame


def _frame_in_intervals(frame_number, intervals):
    """Returns True if a frame is inside of one of the frame intervals. Otherwise
    returns False

    Parameters
    ----------
    frame_number : int
        Number of the frame to be checked.
    intervals : list
        List of intervals where to check for the frame
    """
    if intervals:
        for interval in intervals:
            if frame_number >= interval[0] and frame_number <= interval[1]:
                return True
    return False


def _segment_episode(
    episode_number,
    start,
    end,
    video_path,
    segmentation_parameters,
    segmentation_data_folder,
    video_params_to_store,
    single_video_file,
    save_pixels=None,
    save_segmentation_image=None,
):
    """Gets list of blobs segmented in every frame of the episode of the video
    given by `path` (if the video is splitted in different files) or by
    `episode_start_end_frames` (if the video is given in a single file)

    Parameters
    ----------
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    segmentation_thresholds : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    path : string
        Path to the video file from where to get the VideoCapture (OpenCV) object
    episode_start_end_frames : tuple
        Tuple (starting_frame, ending_frame) indicanting the start and end of the episode
        when the video is given in a single file

    Returns
    -------
    blobs_in_episode : list
        List of `blobs_in_frame` of the episode of the video being segmented
    max_number_of_blobs : int
        Maximum number of blobs found in the episode of the video being segmented

    See Also
    --------
    Video
    Blob
    _get_videoCapture
    segment_frame
    blob_extractor
    """
    # Set file path to store blobs segmentation image and blobs pixels
    if save_segmentation_image == "DISK":
        bounding_box_images_path = os.path.join(
            segmentation_data_folder,
            f"episode_images_{episode_number}.hdf5",
        )
        if os.path.isfile(bounding_box_images_path):
            os.remove(bounding_box_images_path)
    else:
        bounding_box_images_path = None
    if save_pixels == "DISK":
        pixels_path = os.path.join(
            segmentation_data_folder,
            f"episode_pixels_{episode_number}.hdf5",
        )
        if os.path.isfile(pixels_path):
            os.remove(pixels_path)
    else:
        pixels_path = None
    # Read video for the episode
    cap = cv2.VideoCapture(video_path)

    # Get number of frames in the episode
    if single_video_file:
        number_of_frames_in_episode = end - start
        # Moving to first frame of the episode from the single file
        cap.set(1, start)
    else:
        number_of_frames_in_episode = int(cap.get(7))

    max_number_of_blobs = 0
    frame_number = 0
    blobs_in_episode = []
    while frame_number < number_of_frames_in_episode:

        # Compute the global fragment number in the video
        global_frame_number = start + frame_number
        # Compute the frame number in the video file
        if single_video_file:
            frame_number_in_video_path = global_frame_number
        else:
            frame_number_in_video_path = frame_number
        if single_video_file:
            assert global_frame_number == frame_number_in_video_path

        if _frame_in_intervals(
            global_frame_number, segmentation_parameters["tracking_interval"]
        ):
            blobs_in_frame, max_number_of_blobs = _get_blobs_in_frame(
                cap,
                video_params_to_store,
                segmentation_parameters,
                max_number_of_blobs,
                global_frame_number,
                frame_number_in_video_path,
                bounding_box_images_path,
                video_path,
                pixels_path,
                save_pixels,
                save_segmentation_image,
            )
        else:
            ret, _ = cap.read()
            blobs_in_frame = []

        # store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)
        frame_number += 1

    cap.release()
    gc.collect()
    return blobs_in_episode, max_number_of_blobs


def _segment_video_in_parallel(
    episodes_sublists,
    segmentation_data_folder,
    segmentation_parameters,
    video_params_to_store,
    single_video_file,
):
    # init variables to store data
    blobs_in_video = []
    maximum_number_of_blobs_in_episode = []
    logger.info("There is only one path, segmenting by frame indices")
    logger.info(f"Pixels stored in {conf.SAVE_PIXELS}")
    logger.info(
        f"Segmentation images stored in {conf.SAVE_SEGMENTATION_IMAGE}"
    )
    for episodes_sublist in tqdm(episodes_sublists, desc="Segmenting video"):
        OupPutParallel = Parallel(n_jobs=conf.NUMBER_OF_JOBS_FOR_SEGMENTATION)(
            delayed(_segment_episode)(
                episode_number,
                start_end[0],
                start_end[1],
                episode_path,
                segmentation_parameters,
                segmentation_data_folder,
                video_params_to_store,
                single_video_file,
                conf.SAVE_PIXELS,
                conf.SAVE_SEGMENTATION_IMAGE,
            )
            for episode_path, episode_number, start_end in episodes_sublist
        )
        blobs_in_episode = [out[0] for out in OupPutParallel]
        maximum_number_of_blobs_in_episode.append(
            [out[1] for out in OupPutParallel]
        )
        blobs_in_video.append(blobs_in_episode)
    return blobs_in_video, maximum_number_of_blobs_in_episode


def segment(
    video_path: str,
    segmentation_parameters: Dict[str, any],
    video_attributes_to_store_in_each_blob: Dict[str, any],
    episodes_start_end: List[Tuple[int, int]],
    segmentation_data_folder: str,
    video_paths: Optional[List[str]] = None,
) -> Tuple[List[List[Blob]], int]:
    """
    Computes a list of blobs for each frame of the video and the maximum
    number of blobs found in a frame.

    Parameters
    ----------
    video_path
    segmentation_parameters
    video_attributes_to_store_in_each_blob
    episodes_start_end
    segmentation_data_folder
    video_paths

    Returns
    -------

    See Also
    --------
    _segment_video_in_parallel

    """
    # avoid computing with all the cores in very large videos. It fills the RAM.
    num_cpus = int(multiprocessing.cpu_count())
    num_jobs = conf.NUMBER_OF_JOBS_FOR_SEGMENTATION
    if conf.NUMBER_OF_JOBS_FOR_SEGMENTATION is None:
        num_jobs = 1
    elif num_jobs < 0:
        num_jobs = num_cpus + 1 + num_jobs

    if len(video_paths) == 1:
        logger.debug("Single video paths")
        episodes_sublists = []
        for i in range(0, len(episodes_start_end), num_jobs):
            episode_numbers = range(i, i + num_jobs)
            episode_start_ends = episodes_start_end[i : i + num_jobs]
            episode_paths = [video_path] * len(episode_start_ends)
            episodes_sublists.append(
                zip(episode_paths, episode_numbers, episode_start_ends)
            )
        single_video_file = True
    else:
        logger.debug("Many video paths")
        episodes_sublists = []
        for i in range(0, len(episodes_start_end), num_jobs):
            episode_numbers = range(i, i + num_jobs)
            episode_paths = video_paths[i : i + num_jobs]
            episode_start_ends = episodes_start_end[i : i + num_jobs]
            episodes_sublists.append(
                zip(episode_paths, episode_numbers, episode_start_ends)
            )
        single_video_file = False
    # print("******************************************************************")
    # print(num_jobs)
    # print(len(episodes_start_end))
    # print(episode_numbers)
    # print(episode_start_ends)
    # print([list(episodes_sublist) for episodes_sublist in episodes_sublists])
    # print("******************************************************************")
    set_mkl_to_single_thread()
    blobs_in_video, maximum_number_of_blobs = _segment_video_in_parallel(
        episodes_sublists,
        segmentation_data_folder,
        segmentation_parameters,
        video_attributes_to_store_in_each_blob,
        single_video_file,
    )
    set_mkl_to_multi_thread()

    # blobs_in_video is flattened to obtain a list of blobs per
    # episode and then the list of all blobs
    blobs_in_video = flatten(flatten(blobs_in_video))
    maximum_number_of_blobs = max(flatten(maximum_number_of_blobs))
    return blobs_in_video, maximum_number_of_blobs
