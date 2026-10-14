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
import gc
import logging
import os

import cv2
import h5py
import numpy as np

from confapp import conf
from joblib import Parallel, delayed

# this is needed, otherwise conf.SIGMA_GAUSSIAN_BLURRING is the default
try:
    import local_settings  # type: ignore
    conf += local_settings
except ImportError:
    pass

from idtrackerai.blob import Blob
from idtrackerai.utils.py_utils import (
    flatten,
    set_mkl_to_multi_thread,
    set_mkl_to_single_thread,
    find_blob,
    effective_n_jobs,
)
from idtrackerai.animals_detection.segmentation_utils import (
    blob_extractor,
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


def get_segmentation_storage(number_of_animals):
    """
    Where to store the pixels and the image of each blob (DISK, RAM or NONE)

    When tracking a single animal, the identification images are never needed
    and the pixels can be recomputed cheaply from the contour of each blob,
    so SINGLE_ANIMAL_SAVE_PIXELS and SINGLE_ANIMAL_SAVE_SEGMENTATION_IMAGE are used instead

    Returns:
        save_pixels (str)
        save_segmentation_image (str)
    """
    if number_of_animals == 1:
        return (
            getattr(conf, "SINGLE_ANIMAL_SAVE_PIXELS", conf.SAVE_PIXELS),
            getattr(conf, "SINGLE_ANIMAL_SAVE_SEGMENTATION_IMAGE", conf.SAVE_SEGMENTATION_IMAGE),
        )
    return conf.SAVE_PIXELS, conf.SAVE_SEGMENTATION_IMAGE


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
    images_file=None,
    pixels_file=None,
):
    """Segments a frame read from `cap` according to the preprocessing parameters
    in `video`. Returns a list `blobs_in_frame` with the Blob objects in the frame
    and the `max_number_of_blobs` found in the video so far. Frames are segmented
    in gray scale.

    Parameters
    ----------
    cap : <VideoCapture object>
        OpenCV object used to read the frames of the video
    segmentation_parameters : dict
        Dictionary with the thresholds used for the segmentation: `min_threshold`,
        `max_threshold`, `min_area`, `max_area`
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process
    global_frame_number : int
        This is the frame number in the whole video.
    images_file, pixels_file: h5py.File
        Open hdf5 files where the segmentation images and pixels are saved (if saved to DISK)

    Returns
    -------
    blobs_in_frame : list
        List of <Blob object> segmented in the current frame
    max_number_of_blobs : int
        Maximum number of blobs found in the whole video so far in the segmentation process
    success: bool
        False if the frame could not be read or segmented

    See Also
    --------
    Video
    Blob
    segment_frame
    blob_extractor
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Could not read frame %s from %s (chunk %s)", global_frame_number, video_path, chunk)
        return BlobsInFrame(), max_number_of_blobs, False

    if conf.ADVANCED_SEGMENTATION:
        result, success = _process_frame_with_advanced_segmentation(
            frame,
            segmentation_parameters,
            global_frame_number,
            save_pixels,
            save_segmentation_image,
            return_success=True,
        )
    else:
        result, success = _process_frame(
            frame,
            segmentation_parameters,
            global_frame_number,
            save_pixels,
            save_segmentation_image,
            return_success=True,
        )

    (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths,
    ) = result

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
        images_file=images_file,
        pixels_file=pixels_file,
    )

    max_number_of_blobs = max(max_number_of_blobs, len(centroids))

    return blobs_in_frame, max_number_of_blobs, success


def _resize(image, resolution_reduction):
    return cv2.resize(
        image,
        None,
        fx=resolution_reduction,
        fy=resolution_reduction,
        interpolation=cv2.INTER_AREA,
    )


def _process_frame(
    frame,
    segmentation_parameters,
    frame_number,
    save_pixels,
    save_segmentation_image,
    iteration=0,
    erosion=None,
    number_of_animals=None,
    return_success=False,
):
    """
    Segment a frame

    Returns:
        (bounding_boxes, miniframes, centroids, areas, pixels, contours, estimated_body_lengths)
        If any error occurs, all these lists are empty

        if return_success is True, the tuple above and a bool which is False if an error occurred

    iteration and number_of_animals are not used anymore and kept for backwards compatibility
    (see _process_frame_with_advanced_segmentation)
    """

    try:
        bkg = segmentation_parameters["bkg_model"]
        mask = segmentation_parameters["mask"]
        resolution_reduction = segmentation_parameters["resolution_reduction"]

        frame = gaussian_blur(frame, sigma=conf.SIGMA_GAUSSIAN_BLURRING)
        # Convert the frame to gray scale
        gray = to_gray_scale(frame)

        if gray.shape != mask.shape:
            raise ValueError(f"Frame and mask don't have same shape. {gray.shape} != {mask.shape}")

        # Apply resolution reduction
        # the frame that is segmented and the frame where the blob images
        # are taken from must have the same resolution as the mask and the background
        if resolution_reduction != 1:
            gray = _resize(gray, resolution_reduction)
            if bkg is not None:
                bkg = _resize(bkg, resolution_reduction)
            if mask is not None:
                mask = _resize(mask, resolution_reduction)

        # Binarize frame
        segmentedFrame = segment_frame(
            gray,
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

        # Extract blobs info
        # Only the outermost contours are retrieved, which is equivalent to
        # filling the holes of the segmented frame (and much faster)
        result = blob_extractor(
            segmentedFrame,
            gray,
            min_area,
            segmentation_parameters["max_area"],
            save_pixels,
            save_segmentation_image,
            external_only=True,
        )
        success = True

    except Exception:
        logger.exception("Error while segmenting frame number %s", frame_number)
        result = ([], [], [], [], [], [], [])
        success = False

    if return_success:
        return result, success
    return result


def _process_frame_with_advanced_segmentation(
    frame,
    segmentation_parameters,
    frame_number,
    save_pixels,
    save_segmentation_image,
    max_iterations=10,
    return_success=False,
):
    """
    Segment a frame like _process_frame.
    If fewer blobs than animals are found,
    the maximum intensity threshold is progressively lowered
    (and the minimum threshold too, if the range becomes narrower than 10)
    until enough blobs are found or max_iterations are done
    """
    number_of_animals = segmentation_parameters["number_of_animals"]
    segmentation_parameters = segmentation_parameters.copy()

    result, success = _process_frame(
        frame, segmentation_parameters, frame_number,
        save_pixels, save_segmentation_image, return_success=True,
    )
    iteration = 0
    while (
        success
        and iteration < max_iterations
        and len(result[5]) < number_of_animals
        and segmentation_parameters["max_threshold"] > (segmentation_parameters["min_threshold"] + 1)
    ):
        iteration += 1
        segmentation_parameters["max_threshold"] -= 1
        if (segmentation_parameters["max_threshold"] - segmentation_parameters["min_threshold"]) < 10:
            segmentation_parameters["min_threshold"] -= 1

        logger.debug(
            "Frame %s: %s contours found. Trying thresholds %s - %s (iteration %s)",
            frame_number, len(result[5]),
            segmentation_parameters["min_threshold"], segmentation_parameters["max_threshold"],
            iteration,
        )
        result, success = _process_frame(
            frame, segmentation_parameters, frame_number,
            save_pixels, save_segmentation_image, return_success=True,
        )

    if return_success:
        return result, success
    return result


def _save_dataset(file, dataset_name, data):
    try:
        file.create_dataset(dataset_name, data=data)
    except ValueError as error:
        if dataset_name.endswith("-modified"):
            file[dataset_name][:] = data
        else:
            raise error


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
    images_file=None,
    pixels_file=None,
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
        images_file, pixels_file (h5py.File): Already open hdf5 files for bounding_box_images_path and pixels_path.
            If not passed, the files are opened (once per call) if anything needs to be saved to them

    Returns:
        blobs_in_frame (list): Collection of blobs generated from the segmentation results of a single frame
    """

    blobs_in_frame = BlobsInFrame()

    files_to_close = []
    if len(bounding_boxes) > 0:
        if save_segmentation_image == "DISK" and images_file is None:
            images_file = h5py.File(bounding_box_images_path, "a")
            files_to_close.append(images_file)
        if save_pixels == "DISK" and pixels_file is None:
            pixels_file = h5py.File(pixels_path, "a")
            files_to_close.append(pixels_file)

    try:
        # create blob objects
        for i, bounding_box in enumerate(bounding_boxes):
            if not modified:
                dataset_name = str(global_frame_number) + "-" + str(i)
            else:
                dataset_name = str(global_frame_number) + "-" + str(i) + "-modified"

            if save_segmentation_image == "DISK":
                _save_dataset(images_file, dataset_name, miniframes[i])
                miniframes[i] = None

            if save_pixels == "DISK":
                _save_dataset(pixels_file, dataset_name, pixels[i])
                pixels[i] = None

            blob = Blob(
                centroids[i],
                contours[i],
                areas[i],
                bounding_box,
                bounding_box_image=miniframes[i],
                bounding_box_images_path=bounding_box_images_path if save_segmentation_image == "DISK" else None,
                estimated_body_length=estimated_body_lengths[i],
                number_of_animals=video_params_to_store["number_of_animals"],
                frame_number=global_frame_number,
                pixels=pixels[i],
                pixels_path=pixels_path if save_pixels == "DISK" else None,
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
    finally:
        for file in files_to_close:
            file.close()

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


def _episode_file(segmentation_data_folder, kind, episode_number, save):
    """Path to the hdf5 file where the pixels or images of an episode are saved,
    or None if they are not saved to DISK. Any previous file is removed"""
    if save != "DISK":
        return None
    path = os.path.join(segmentation_data_folder, f"episode_{kind}_{episode_number}.hdf5")
    if os.path.isfile(path):
        os.remove(path)
    return path


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
    # Each episode runs in its own process,
    # so OpenCV should not spawn more threads
    previous_number_of_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)

    # Set file path to store blobs segmentation image and blobs pixels
    bounding_box_images_path = _episode_file(segmentation_data_folder, "images", episode_number, save_segmentation_image)
    pixels_path = _episode_file(segmentation_data_folder, "pixels", episode_number, save_pixels)
    images_file = h5py.File(bounding_box_images_path, "a") if bounding_box_images_path else None
    pixels_file = h5py.File(pixels_path, "a") if pixels_path else None

    # Read video for the episode
    cap = VideoCapture(video_path, chunk)

    try:
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
        segmented_frames = 0
        failed_frames = 0
        reset=None
        while frame_number < number_of_frames_in_episode:

            # Compute the global fragment number in the video
            global_frame_number = start + frame_number
            # Compute the frame number in the video file
            if single_video_file:
                frame_number_in_video_path = global_frame_number
            else:
                frame_number_in_video_path = frame_number

            if _frame_in_intervals(
                global_frame_number, segmentation_parameters["tracking_interval"]
            ) and ((global_frame_number % conf.SKIP_EVERY_FRAME) == 0 or frame_number == 0):
                reset=True
                blobs_in_frame, max_number_of_blobs, success = _get_blobs_in_frame(
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
                    images_file=images_file,
                    pixels_file=pixels_file,
                )
                segmented_frames += 1
                failed_frames += int(not success)
            else:
                if conf.SKIP_EVERY_FRAME > 5:
                    if reset is not None and reset==True:
                        next_fn=conf.SKIP_EVERY_FRAME*(global_frame_number//conf.SKIP_EVERY_FRAME)+conf.SKIP_EVERY_FRAME
                        logger.debug(f"{global_frame_number}: Skipping to frame {next_fn}")
                        cap.set(1, next_fn)
                        reset=False
                else:
                    ret, _ = cap.read()

                blobs_in_frame = BlobsInFrame()

            # store all the blobs encountered in the episode
            blobs_in_episode.append(blobs_in_frame)
            frame_number += 1

    finally:
        cap.release()
        for file in [images_file, pixels_file]:
            if file is not None:
                file.close()
        cv2.setNumThreads(previous_number_of_threads)

    if failed_frames > 0:
        logger.warning(
            "Episode %s (frames %s - %s): %s of %s frames could not be read or segmented",
            episode_number, start, end, failed_frames, segmented_frames
        )
        if failed_frames == segmented_frames:
            raise RuntimeError(
                f"None of the frames of episode {episode_number} (frames {start} - {end}) "
                f"of {video_path} (chunk {chunk}) could be segmented. See the log for details"
            )

    gc.collect()
    return blobs_in_episode, max_number_of_blobs


def _segment_video_in_parallel(
    episodes,
    chunk,
    segmentation_data_folder,
    segmentation_parameters,
    video_params_to_store,
    single_video_file,
    n_jobs,
    save_pixels,
    save_segmentation_image,
):
    """
    Segment all the episodes in parallel

    Arguments:
        episodes (list): (episode_path, episode_number, (start, end)) of each episode

    Returns:
        blobs_in_video (list): blobs_in_frame of each frame of the episodes, in order
        maximum_number_of_blobs (list): maximum number of blobs in a frame of each episode
    """
    logger.info(f"Segmenting {len(episodes)} episodes with {n_jobs} jobs")
    logger.info(f"Pixels stored in {save_pixels}")
    logger.info(f"Segmentation images stored in {save_segmentation_image}")

    output = Parallel(n_jobs=n_jobs)(
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
            save_pixels,
            save_segmentation_image,
        )
        for episode_path, episode_number, start_end in episodes
    )
    blobs_in_video = flatten([blobs_in_episode for blobs_in_episode, _ in output])
    maximum_number_of_blobs = [max_number_of_blobs for _, max_number_of_blobs in output]
    return blobs_in_video, maximum_number_of_blobs


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

    See Also
    --------
    _segment_video_in_parallel

    """
    num_jobs = effective_n_jobs(conf.NUMBER_OF_JOBS_FOR_SEGMENTATION)

    if len(video_paths) == 1:
        logger.debug("Single video paths")
        episode_paths = [video_path] * len(episodes_start_end)
        single_video_file = True
    else:
        logger.debug("Many video paths")
        episode_paths = video_paths
        single_video_file = False

    episodes = list(zip(episode_paths, range(len(episodes_start_end)), episodes_start_end))

    save_pixels, save_segmentation_image = get_segmentation_storage(
        segmentation_parameters["number_of_animals"]
    )

    set_mkl_to_single_thread()
    blobs_in_video, maximum_number_of_blobs = _segment_video_in_parallel(
        episodes,
        chunk,
        segmentation_data_folder,
        segmentation_parameters,
        video_attributes_to_store_in_each_blob,
        single_video_file,
        num_jobs,
        save_pixels,
        save_segmentation_image,
    )
    set_mkl_to_multi_thread()

    maximum_number_of_blobs = max(maximum_number_of_blobs)
    return blobs_in_video, maximum_number_of_blobs
