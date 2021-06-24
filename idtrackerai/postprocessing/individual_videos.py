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


import logging
import os
import sys
from math import sqrt

import cv2
import numpy as np
from confapp import conf
from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger("__main__.video")


def get_frame(frame, centroid, height, width):
    if not np.all(np.isnan(centroid)):
        X, Y = int(centroid[1]), int(centroid[0])
        r0, r1 = X - width // 2, X + width // 2
        c0, c1 = Y - height // 2, Y + height // 2
        miniframe = frame[r0:r1, c0:c1]
        if miniframe.shape[0] == height and miniframe.shape[1] == width:
            return miniframe
        else:
            return np.zeros((height, width))
    else:
        return np.zeros((height, width))


def initialize_video_writer(video_object, height, width, identity):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    file_name = os.path.join(
        video_object.individual_videos_folder,
        "minivideo_{}.avi".format(identity),
    )
    out = cv2.VideoWriter(
        file_name, fourcc, video_object.frames_per_second, (height, width)
    )
    return out


def generate_individual_video(
    video_object, trajectories, identity, width, height
):
    # Initialize video writer
    out = initialize_video_writer(video_object, height, width, identity)
    # Initialize cap reader
    if len(video_object.video_paths) > 1:
        current_segment = 0
        cap = cv2.VideoCapture(video_object.video_paths[current_segment])
        start = video_object._episodes_start_end[current_segment][0]
    else:
        cap = cv2.VideoCapture(video_object.video_path)

    for frame_number in range(video_object.number_of_frames):
        # Update cap if necessary.
        if len(video_object.video_paths) > 1:
            segment_number = video_object.in_which_episode(frame_number)
            if current_segment != segment_number:
                print(video_object.video_paths[segment_number])
                cap = cv2.VideoCapture(
                    video_object.video_paths[segment_number]
                )
                start = video_object._episodes_start_end[segment_number][0]
                current_segment = segment_number
            cap.set(1, frame_number - start)
        # Read frame
        try:
            ret, frame = cap.read()
        except cv2.error:
            raise Exception("could not read frame")
        # Generate frame for individual
        individual_frame = get_frame(
            frame, trajectories[frame_number, identity - 1], height, width
        )
        # Write frame in video
        out.write(individual_frame.astype("uint8"))
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def compute_width_height_individual_video(video_object):
    if conf.INDIVIDUAL_VIDEO_WIDTH_HEIGHT is None:
        height, width = 2 * [
            int(video_object.median_body_length_full_resolution * 1.5 / 2) * 2
        ]
    else:
        height, width = 2 * [conf.INDIVIDUAL_VIDEO_WIDTH_HEIGHT]
    return height, width


def generate_individual_videos(video_object, trajectories):
    """
    Generates one individual-centered video for every individual in the video.
    The video will contain black frames where the trajectories have NaN, or when
    the fish is too close to the border of the original video frame.
    """
    # Cretae folder to store videos
    video_object.create_individual_videos_folder()
    # Calculate width and height of the video from the estimated body length
    height, width = compute_width_height_individual_video(video_object)
    logger.info("Generating individual videos ...")
    Parallel(n_jobs=-2)(
        delayed(generate_individual_video)(
            video_object,
            trajectories,
            identity=i + 1,
            width=width,
            height=height,
        )
        for i in range(
            video_object.user_defined_parameters["number_of_animals"]
        )
    )
    logger.info("Invididual videos generated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video_object_path",
        type=str,
        help="Path to the video object created during the tracking session",
    )
    parser.add_argument(
        "-t",
        "--trajectories_path",
        type=str,
        help="Path to the trajectory file",
    )
    args = parser.parse_args()

    print("Loading video information from {}".format(args.video_object_path))
    video_object = np.load(args.video_object_path, allow_pickle=True).item()
    print("Loading trajectories from {}".format(args.trajectories_path))
    trajectories_dict = np.load(
        args.trajectories_path, allow_pickle=True
    ).item()

    generate_individual_videos(video_object, trajectories_dict["trajectories"])
