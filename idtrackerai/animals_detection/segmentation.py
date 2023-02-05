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

from typing import Tuple, List, Dict, Optional, Union
import time
import gc
import logging
import multiprocessing
import os
import traceback

import cv2
import h5py
import numpy as np

from scipy import ndimage
from confapp import conf
from joblib import Parallel, delayed
from torch import normal
from tqdm import tqdm

# this is needed, otherwise conf.SIGMA_GAUSSIAN_BLURRING is the default
try:
    import local_settings # type: ignore
    conf += local_settings
except:
    pass

from idtrackerai.blob import Blob
from idtrackerai.utils.py_utils import (
    flatten,
    set_mkl_to_multi_thread,
    set_mkl_to_single_thread,
    find_blob,
)
from idtrackerai.animals_detection.segmentation_utils import (
    blob_extractor,
    get_frame_average_intensity,
    segment_frame,
    to_gray_scale,
    gaussian_blur,
)

try:
    from imgstore.interface import VideoCapture
except ModuleNotFoundError:
    from cv2 import VideoCapture

import idtrackerai.constants as cons

logger = logging.getLogger("__main__.segmentation")

"""
The segmentation module
"""


class BlobsInFrame(List):

    def __getitem__(self, k):
        if isinstance(k, int):
            return super(BlobsInFrame, self).__getitem__(k)

        elif isinstance(k, tuple):
            return find_blob(self, k)
        else:
            print(f"Passed invalid key {k} of type {type(k)}")

            raise Exception(
                """
                BlobsInFrame can only be indexed by:
                  int: return the blob under that position of the list
                  list: return the blob with the bounding box given by the list (errors if no match)
                """
            )


def _get_blobs_in_frame(
    cap,
    video_params_to_store,
    segmentation_parameters,
    max_number_of_blobs,
    global_frame_number,
    frame_number_in_video_path,
    bounding_box_images_path,
    video_path,
    chunk,
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
        chunk,
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
    iteration=0,
    erosion=None,
    number_of_animals=None,
):

    orig_frame = frame.copy()

    try:

        bkg = segmentation_parameters["bkg_model"]
        mask = segmentation_parameters["mask"]

        assert frame.shape[:2] == mask.shape

        frame = gaussian_blur(frame, sigma=conf.SIGMA_GAUSSIAN_BLURRING)
        # Convert the frame to gray scale
        gray = to_gray_scale(frame)
        # Normalize frame
        avg_intensity = get_frame_average_intensity(gray, mask)
        # print(avg_intensity)
        # normalized_framed = gray / avg_intensity
        normalized_framed = gray

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
        # Binarize frame
        segmentedFrame = segment_frame(
            normalized_framed,
            segmentation_parameters["min_threshold"],
            segmentation_parameters["max_threshold"],
            bkg,
            mask,
            segmentation_parameters["subtract_bkg"],
        )

        if erosion is not None:
            segmentedFrame = cv2.erode(segmentedFrame, erosion["kernel"], iterations=erosion["iterations"])
            fraction = 1 - erosion["iterations"]*0.2
            min_area = max(int(segmentation_parameters["min_area"] * fraction), erosion["iterations"]*0.2)
        else:
            min_area = segmentation_parameters["min_area"]


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
            min_area,
            segmentation_parameters["max_area"],
            save_pixels,
            save_segmentation_image,
        )
        segmentation_parameters=segmentation_parameters.copy()


        if conf.ADVANCED_SEGMENTATION:
            if len(contours) == 0:
                np.save(str(frame_number) + "_segmentation_parameters.npy", segmentation_parameters)
                np.save(str(frame_number) + "_normalized.npy", normalized_framed)
                cv2.imwrite(str(frame_number) + "_frame.png", gray)
                cv2.imwrite(str(frame_number) + "_segmented_frame.png", segmentedFrame)
                raise Exception("No contour found")
            
            if number_of_animals is not None and len(contours) != number_of_animals:

                return perform_advanced_segmentation(
                    orig_frame.copy(),
                    frame_number=frame_number,
                    contours=contours,
                    segmentation_parameters=segmentation_parameters,
                    save_pixels=save_pixels,
                    save_segmentation_image=save_segmentation_image,
                    iteration=iteration,
                )

    except Exception as error:
        print(f"Error on frame {frame_number}: {error}")
        logger.warning(traceback.print_exc())
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


def perform_advanced_segmentation(frame, frame_number, contours, segmentation_parameters, iteration, **kwargs):
    
    bounding_boxes = []
    miniframes = []
    centroids = []
    areas = []
    pixels = []
    contours = []
    estimated_body_lengths = []
    result = (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
    )

    if (
        # iteration < 20 and
        iteration < 10 and
        len(contours) < segmentation_parameters["number_of_animals"] and
        segmentation_parameters["max_threshold"] > (segmentation_parameters["min_threshold"]+1)
    ):
        # print(frame_number, len(contours), segmentation_parameters)
        iteration+=1
        # print(frame_number, f"Iteration: {iteration}")
        segmentation_parameters["max_threshold"] -= 1
        if (segmentation_parameters["max_threshold"] - segmentation_parameters["min_threshold"]) < 10:
            segmentation_parameters["min_threshold"] -= 1
            
            
        logger.debug("Dynamic segmentation parameters: ")
        logger.debug(segmentation_parameters)

        result = _process_frame(
            frame,
            segmentation_parameters,
            frame_number,
            iteration=iteration,
            **kwargs
        )

    if iteration != 0:
        logger.debug(f"""
            Calling _process_frame with segmentation thresholds
                min: {segmentation_parameters['min_threshold']}
                max: {segmentation_parameters['max_threshold']}
                frame_number: {frame_number}
                iteration: {iteration}
                # contours: {len(contours)}
            """
        )

    return result


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
    chunk,
    segmentation_parameters,
    modified=False,
):
    """
    Generate idtrackerai.Blob instances from the segmentation results
    
    Args:
        bounding_boxes (list): Bounding box of each segmented contour, in format ((x1, y1), (x2, y2))
        miniframes (list):Image crops, one per passed bounding box.
            Their width and height must agree with the width and height of the corresponding bounding box
        centroids (list): Centroid of the contour in coordinates of the image, in format (x, y). x and y may be floats
        areas (list): Area of each contour, may be float 
        pixels (list): Pixels of the contour, in raveled format (i.e. in 1D)
        contours (list): Contour of the segmented blob 
        estimated_body_lengths (list): Approximate length of the longest side of each animal, in int format
        save_segmentation_image (str): Whether to save the segmentation data to disk (DISK), RAM (RAM) or not (NONE)
        bounding_box_images_path (str): Path to a segmentation_data/episode_images_X.hdf5 file to which the segmentation image will be saved if required
        save_pixels (str): Whether to save the pixels to disk (DISK), RAM (RAM) or not (NONE)
        pixels_path (str): Path to a segmentation_data/episode_pixels_X.hdf5 file to which the pixels will be saved if required
        global_frame_number (int): Frame number in the full imgstore or video
        frame_number_in_video_path (int): Frame number in the video chunk or video
        video_params_to_store (dict): Parameters of the video to be saved. Must contain width, height and number_of_animals
        video_path (str): Path to the video or metadata.yaml (imgstore)
        chunk (int): Chunk of the imgstore
        segmentation_parameters (dict):
            min_threshold, max_threshold, min_area, max_area, apply_ROI (bool), rois (list), mask (np.ndarray),
            subtract_bkg (bool), bkg_model, resolution_reduction, tracking_interval (list of length 2 lists), number_of_animals (int)

    Returns:
        blobs_in_frame (list): Collection of blobs generated from the segmentation results of a single frame 
    """

    blobs_in_frame = BlobsInFrame()
       
    # create blob objects
    for i, bounding_box in enumerate(bounding_boxes):
        if not modified:
            dataset_name = str(global_frame_number) + "-" + str(i)
        else:
            dataset_name = str(global_frame_number) + "-" + str(i) + "-modified"
            print(dataset_name)

        if save_segmentation_image == "DISK":
            with h5py.File(bounding_box_images_path, "a") as f1:
                try:
                    f1.create_dataset(
                        dataset_name, data=miniframes[i]
                    )
                except ValueError as error:
                    if dataset_name.endswith("-modified"):
                        f1[dataset_name][:] = miniframes[i]
                    else:
                        raise error

            miniframes[i] = None
        if save_pixels == "DISK":
            with h5py.File(pixels_path, "a") as f2:
                try:
                    f2.create_dataset(
                        dataset_name, data=pixels[i]
                    )
                except ValueError as error:
                    if dataset_name.endswith("-modified"):
                        f2[dataset_name][:] = pixels[i]
                    else:
                       raise error

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
            chunk=chunk,
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
    chunk,
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
    cap = VideoCapture(video_path, chunk)

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
    called = 0
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
        ) and (global_frame_number % conf.SKIP_EVERY_FRAME  == 0 or frame_number == 0 or frame_number >= (number_of_frames_in_episode-conf.SKIP_EVERY_FRAME)):
            try:
                blobs_in_frame, max_number_of_blobs = _get_blobs_in_frame(
                    cap,
                    video_params_to_store,
                    segmentation_parameters,
                    max_number_of_blobs,
                    global_frame_number,
                    frame_number_in_video_path,
                    bounding_box_images_path,
                    video_path,
                    chunk,
                    pixels_path,
                    save_pixels,
                    save_segmentation_image,
                )
                called += 1
            except Exception as error:
                # print(f"Start {start}")
                # print(global_frame_number)
                # print(f"Called {called} times")
                raise error
        else:
            ret, _ = cap.read()
            blobs_in_frame = BlobsInFrame()

        # store all the blobs encountered in the episode
        blobs_in_episode.append(blobs_in_frame)
        frame_number += 1

    cap.release()
    gc.collect()
    return blobs_in_episode, max_number_of_blobs


def _segment_video_in_parallel(
    episodes_sublists,
    chunk,
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
                chunk,
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
    chunk: Union[int, None],
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
        chunk,
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
