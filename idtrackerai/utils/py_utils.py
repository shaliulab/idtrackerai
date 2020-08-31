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

import glob
import logging
import multiprocessing
import os
import re

import numpy as np
from matplotlib import cm

logger = logging.getLogger("__main__.py_utils")

### MKL
def set_mkl_to_single_thread():
    logger.info("Setting MKL library to use single thread")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"


def set_mkl_to_multi_thread():
    logger.info("Setting MKL library to use multiple threads")
    os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
    os.environ["MKL_DYNAMIC"] = "TRUE"


### Object utils ###
def append_values_to_lists(values, list_of_lists):
    list_of_lists_updated = []

    for list_, value in zip(list_of_lists, values):
        list_.append(value)
        list_of_lists_updated.append(list_)

    return list_of_lists_updated


def set_attributes_of_object_to_value(
    object_to_modify, attributes_list, value=None
):
    [
        setattr(object_to_modify, attribute, value)
        for attribute in attributes_list
        if hasattr(object_to_modify, attribute)
    ]


def delete_attributes_from_object(object_to_modify, list_of_attributes):
    [
        delattr(object_to_modify, attribute)
        for attribute in list_of_attributes
        if hasattr(object_to_modify, attribute)
    ]


### Dict utils ###
def flatten(list_):
    """ flatten a list of lists """
    try:
        ans = [inner for outer in list_ for inner in outer]
    except TypeError:
        ans = [y for x in list_ for y in (x if isinstance(x, tuple) else (x,))]
    return ans


def natural_sort(list_):
    def convert(text):
        if text.isdigit():
            return int(text)
        else:
            return text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(list_, key=alphanum_key)


def scanFolder(path):
    ### NOTE if the video selected does not finish with '_1' the scanFolder
    # function won't select all of them. This can be improved
    paths = [path]
    video_path = os.path.basename(path)
    filename, extension = os.path.splitext(video_path)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-2:] == "_1":
        paths = natural_sort(
            glob.glob(folder + "/" + filename[:-1] + "*" + extension)
        )
    return paths


def get_spaced_colors_util(n, norm=False, black=True, cmap="jet"):
    RGB_tuples = cm.get_cmap(cmap)
    if norm:
        colors = [RGB_tuples(i / n) for i in range(n)]
    else:
        RGB_array = np.asarray([RGB_tuples(i / n) for i in range(n)])
        BRG_array = np.zeros(RGB_array.shape)
        BRG_array[:, 0] = RGB_array[:, 2]
        BRG_array[:, 1] = RGB_array[:, 1]
        BRG_array[:, 2] = RGB_array[:, 0]
        colors = [tuple(BRG_array[i, :] * 256) for i in range(n)]
    if black:
        black = (0.0, 0.0, 0.0)
        colors.insert(0, black)
    return colors


def check_and_change_video_path(video_object, old_video):
    current_video_folder = os.path.split(video_object.video_path)[0]
    old_video_folder = os.path.split(old_video.video_path)[0]
    old_video_session_name = old_video.session_folder
    print("in check and change video path")
    print("current_video_folder: ", current_video_folder)
    print("old_video_folder: ", old_video_folder)
    print("old_video_session_name: ", old_video_session_name)
    if current_video_folder != old_video_folder:
        print("updating attributes")
        attributes_to_modify = {
            key: getattr(old_video, key)
            for key in old_video.__dict__
            if isinstance(getattr(old_video, key), str)
            and old_video_folder in getattr(old_video, key)
        }

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(
                old_video_folder, current_video_folder
            )
            setattr(old_video, key, new_value)

        if (
            old_video.paths_to_video_segments is not None
            and len(old_video.paths_to_video_segments) != 0
        ):
            logger.info("Updating paths_to_video_segments")
            new_paths_to_video_segments = []
            for path in old_video.paths_to_video_segments:
                new_path = os.path.join(
                    old_video.video_folder, os.path.split(path)[1]
                )
                new_paths_to_video_segments.append(new_path)
            old_video._paths_to_video_segments = new_paths_to_video_segments

        ### update checkpoint files
        current_video_session_name = old_video.session_folder
        folders_to_check = [
            "_crossings_detector_folder",
            "_pretraining_folder",
            "_accumulation_folder",
        ]
        for folder in folders_to_check:
            if (
                hasattr(old_video, folder)
                and getattr(old_video, folder) is not None
            ):
                if folder == "crossings_detector_folder":
                    checkpoint_path = os.path.join(
                        old_video.crossings_detector_folder, "checkpoint"
                    )
                    if os.path.isfile(checkpoint_path):
                        old_video.update_tensorflow_checkpoints_file(
                            checkpoint_path,
                            old_video_session_name,
                            current_video_session_name,
                        )
                    else:
                        logger.warn("No checkpoint found in %s " % folder)
                else:
                    for sub_folder in ["conv", "softmax"]:
                        checkpoint_path = os.path.join(
                            getattr(old_video, folder),
                            sub_folder,
                            "checkpoint",
                        )
                        if os.path.isfile(checkpoint_path):
                            old_video.update_tensorflow_checkpoints_file(
                                checkpoint_path,
                                old_video_session_name,
                                current_video_session_name,
                            )
                        else:
                            logger.warn(
                                "No checkpoint found in %s "
                                % os.path.join(
                                    getattr(old_video, folder), sub_folder
                                )
                            )
    return old_video


def set_load_previous_dict(old_video, processes, computed_processes):
    print("in set load previous dict")
    processes_to_attributes = {
        "preprocessing": ["has_been_preprocessed"],
        "protocols1_and_2": ["first_accumulation_finished"],
        "protocol3_pretraining": ["has_been_pretrained"],
        "protocol3_accumulation": ["second_accumulation_finished"],
        "residual_identification": ["has_been_assigned"],
        "post_processing": [
            "has_crossings_solved",
            "has_trajectories",
            "has_trajectories_wo_gaps",
        ],
    }
    for process in processes:
        attributes = processes_to_attributes[process]
        attributes_values = []
        print(process)
        for attribute in attributes:
            print(attribute, getattr(old_video, attribute))
            attributes_values.append(getattr(old_video, attribute))
        if None in attributes_values:
            computed_processes[process] = "-1"
        elif all(attributes_values):
            logger.debug(attribute)
            computed_processes[process] = "1"
        else:
            computed_processes[process] = "0"

    return computed_processes


def get_computed_processes(video_object, processes):
    """get processes already computed in a previous session
    preprocessing: segmentation, fragmentation and creation of blobs and individual/global fragments
    knowledge_transfer: knowledge transferred from a model trained on a different video_object
    first_accumulation: first accumulation attempt
    pretraining: building the filters in a global-identity-agnostic way
    second_accumulation: accumulation by transferring knowledge from pre-training
    assignment: assignment of the idenitity to each individual fragment
    solving_duplications: solve eventual identity duplications
    crossings: assign identity to single animals during occlusions
    trajectories: compute the individual trajectories
    """
    computed_processes = {name: False for name in processes}
    old_video = None
    print(video_object._previous_session_folder)
    if os.path.isdir(video_object._previous_session_folder):
        logger.info("loading old video object from get_computed_processes")
        previous_video_object_path = os.path.join(
            video_object._previous_session_folder, "video_object.npy"
        )
        print(previous_video_object_path)
        if os.path.isfile(previous_video_object_path):
            old_video = np.load(
                previous_video_object_path, allow_pickle=True
            ).item()
            old_video.update_paths(previous_video_object_path)
            logger.info("old video_object loaded")
        else:
            logger.info(
                f"The folder {video_object._previous_session_folder} is empty. "
                f"The tracking cannot be resumed."
            )
            return computed_processes, old_video
        print(video_object)
        print(old_video)
        old_video = check_and_change_video_path(video_object, old_video)
        print(processes)
        print(computed_processes)
        computed_processes = set_load_previous_dict(
            old_video, processes, computed_processes
        )

    return computed_processes, old_video


def interpolate_nans(t):
    """Interpolates nans linearly in a trajectory

    :param t: trajectory
    :returns: interpolated trajectory
    """
    shape_t = t.shape
    reshaped_t = t.reshape((shape_t[0], -1))
    for timeseries in range(reshaped_t.shape[-1]):
        y = reshaped_t[:, timeseries]
        nans, x = _nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    # Ugly slow hack, as reshape seems not to return a view always
    back_t = reshaped_t.reshape(shape_t)
    t[...] = back_t


def _nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
