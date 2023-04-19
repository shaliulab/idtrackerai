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
from tkinter import N
import warnings
from math import sqrt
import matplotlib.pyplot as plt
import cv2
import codetiming
try:
    import cv2cuda # type: ignore
    CV2CUDA_AVAILABLE=True
    
except:
    CV2CUDA_AVAILABLE=False

import numpy as np
from confapp import conf
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial.distance import euclidean
import imgstore.stores
from idtrackerai.blob import _get_rotation_angle
from imgstore.util import load_config
logger = logging.getLogger("__main__.video")

try:
    from imgstore.interface import VideoCapture
except ModuleNotFoundError:
    from cv2 import VideoCapture

if CV2CUDA_AVAILABLE:
    ENCODER_FORMAT="h264_nvenc/mp4"
else:
    ENCODER_FORMAT="divx/avi"

from idtrackerai.list_of_blobs import ListOfBlobs

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


def initialize_video_writer_cpu(video_object, height, width, identity):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    extension = ".avi"    
    file_name = os.path.join(
        video_object.individual_videos_folder,
        f"minivideo_{identity}{extension}",
    )
    out = cv2.VideoWriter(
        file_name, fourcc, video_object.frames_per_second, (height, width)
    )

    print(f"--> {file_name}")
    return out

def initialize_video_writer_gpu(video_object, height, width, identity):
    assert CV2CUDA_AVAILABLE

    extension = ".mp4"    
    file_name = os.path.join(
        video_object.individual_videos_folder,
        f"minivideo_{identity}{extension}",
    )
    out = cv2cuda.VideoWriter(
        filename=file_name,
        apiPreference="FFMPEG",
        fourcc="h264_nvenc",
        fps=video_object.frames_per_second,
        frameSize=(width, height),
        isColor=False,        
    )
    print(f"--> {file_name}")
    return out

def initialize_store(video_object, height, width, identity, prefix="fly"):

    basedir = os.path.join(
        video_object.individual_videos_folder,
        f"{prefix}_{identity}",
    )

    store = imgstore.new_for_format(
        mode="w",
        fmt=ENCODER_FORMAT,
        framerate=video_object.frames_per_second,
        basedir=basedir,
        imgshape=(height, width),
        chunksize=video_object.chunksize,
        imgdtype=np.uint8
    )
    
    return store


def initialize_video_writer(*args, **kwargs):
    
    if CV2CUDA_AVAILABLE:
        return initialize_video_writer_gpu(*args, **kwargs)
    else:
        return initialize_video_writer_cpu(*args, **kwargs)

def rotate_image(img, angle):
    s = 1
    img=cv2.copyMakeBorder(img, img.shape[0] // s, img.shape[0] // s, img.shape[1] // s, img.shape[1] // s, borderType=cv2.BORDER_CONSTANT, value=0)
    center=tuple([e//2 for e in img.shape[::-1]])
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle+45+180, scale=1)
    # apply the rotation
    rotated = cv2.warpAffine(src=img, M=rotate_matrix, dsize=img.shape[:2][::-1])
    return rotated
        

def clean_mask(mask):

    contours = cv2.findContours(cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    contour = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
    mask = np.zeros_like(mask)
    mask=cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask

def postprocess_frame(config, video_object, individual_frame):
    
    body_length = video_object.median_body_length
    
   
    assert "_wing" in config, "Please enter a threshold value for the wings"
    min_intensity = config["_intensity"]["value"][0]
    min_wing_intensity = config["_wing"]["value"][0]
    max_wing_intensity = config["_wing"]["value"][1]
    
    

    animal_mask = cv2.bitwise_and(
        cv2.threshold(individual_frame, min_intensity, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(individual_frame, max_wing_intensity, 255, cv2.THRESH_BINARY_INV)[1]
    )
    
    wing_mask = cv2.bitwise_and(
        cv2.threshold(individual_frame, min_wing_intensity, 255, cv2.THRESH_BINARY)[1],
        cv2.threshold(individual_frame, max_wing_intensity, 255, cv2.THRESH_BINARY_INV)[1]
    )
    
    cleaned_mask=clean_mask(animal_mask)
    

    kernel = np.ones((int(body_length/20), int(body_length/20)))
    wing_mask_eroded = cv2.erode(wing_mask, kernel, 2)

    cv2.imshow("wing mask", wing_mask_eroded)

    central_mask_column = wing_mask_eroded[:, int(wing_mask_eroded.shape[1] / 3):int(2*wing_mask_eroded.shape[1] / 3)]
    
    mean_y = np.mean(np.where(central_mask_column == 255)[0])
    if mean_y < wing_mask_eroded.shape[0] // 2:
        flip = True
    else:
        flip = False
        
    # remove background
    individual_frame[cleaned_mask==0]=255
    return individual_frame, flip, {"wing_mask_eroded": wing_mask_eroded}

import math
DEBUG_FN=math.inf # no debug


def process_missing_data(out, frame_number, height, width, fts, i, identity=None):
    logger.debug(f"No blob found in frame number {frame_number} with identity {identity}")
    # TODO Make this code faster
    # by getting the right frame time
    # without having to call get_image
    out.add_image(np.zeros((height, width), np.uint8), frame_number, fts[i])
    out.add_extra_data(
        blob_missing=True,
        light=None,
        temperature=None,
        humidity=None,
        flip=None,
        nearby_flies={},
    )

def generate_individual_video_rotation_invariant(video_object, trajectories, lists_of_blobs, identity, width, height):
    
    identity_in_first_chunk=identity
    del identity

    # Initialize video writer
    out = initialize_store(video_object, height, width, identity_in_first_chunk)
    # Initialize cap reader
    if len(video_object.video_paths) > 1:
        current_segment = 0
        cap = VideoCapture(video_object.video_paths[current_segment], chunk=video_object._chunk)
        start = video_object._episodes_start_end[current_segment][0]
    else:
        cap = VideoCapture(video_object.video_path, chunk=video_object._chunk)

    store = VideoCapture(video_object.video_path, video_object._chunk)
    config = load_config(store)
    extra_data = store.get_extra_data()
    index = {row["frame_time"]: row.name for i, row in extra_data.iterrows()}

    first_chunk=video_object.concatenation[0,0]-1
    chunk = 0
    started=False

    while True:
        try:
            list_of_blobs = lists_of_blobs[chunk]
            started=True
        except:
            if not started and chunk == 2000:
                raise Exception("No data found")
            elif not started:
                chunk += 1
            else:
                break

        list_of_blobs._annotate_location_of_blobs()
        start, end = list_of_blobs._start_end_with_blobs
        end+=1
        fts = store._index.get_chunk_metadata(chunk)["frame_time"]

        single_trajectory = []
        print(start, end)
        for i, frame_number in enumerate(tqdm(range(start, end), desc=f"Cropping chunk {chunk}")):

            if DEBUG_FN != math.inf and frame_number < DEBUG_FN:
                continue

            blobs_in_frame = list_of_blobs.blobs_in_video[frame_number]

            if chunk == first_chunk:
                identity = identity_in_first_chunk
            else:
                row_id = video_object.concatenation[:,0] == chunk
                if row_id.sum() == 0:
                    process_missing_data(out, frame_number, height, width, fts, i, identity=None)
                    single_trajectory.append([np.nan, np.nan])
                    continue

                identity = video_object.concatenation[row_id,1:].flatten()[identity_in_first_chunk-1]

            candidate_blobs = [blob for blob in blobs_in_frame if blob.final_identities[0] == identity]

            if candidate_blobs:
                blob = candidate_blobs[0]
                assert blob.frame_number == frame_number
            else:
                process_missing_data(out, frame_number, height, width, fts, i, identity=identity)
                single_trajectory.append([np.nan, np.nan])
                continue

            with codetiming.Timer(text="{milliseconds:.8f} to compute angle", logger=logger.debug):
                rot_angle = _get_rotation_angle(
                    blob.pixels,
                    video_object.height,
                    video_object.width
                )[0]
                if rot_angle > 180:
                    rot_angle -= 360

            # Read frame
            try:
                frame, (frame_number, frame_time) = cap.get_image(frame_number)

            except cv2.error:
                raise Exception("could not read frame")

            centroid=trajectories[frame_number, identity_in_first_chunk-1]
            
            
            nearby_flies={}
            for k, other_centroid in enumerate(trajectories[frame_number, :]):
                if np.isnan(other_centroid).all():
                    continue

                dist=euclidean(centroid, other_centroid)
                if 0 < dist < 1.5 * video_object.median_body_length:
                    other_identity = k+1
                    nearby_flies[other_identity]=dist
                    
            # Generate frame for individual
            individual_frame = get_frame(
                frame, centroid, height*2, width*2
            )
            individual_frame = rotate_image(individual_frame, rot_angle)
            individual_frame_centroid = (individual_frame.shape[1] // 2, individual_frame.shape[0] // 2)
            
            tl_x = int(individual_frame_centroid[0]-width / 2)
            tl_y = int(individual_frame_centroid[1]-height / 2)
            
            individual_frame=individual_frame[
                tl_y:(tl_y+height),
                tl_x:(tl_x+width)
            ]


            # folder = "/home/vibflysleep/individual_videos/debug"
            # os.makedirs(folder, exist_ok=True)
            # blob.reset_next()
            # blob.reset_previous()
            # np.save(os.path.join(folder, f"blob_{frame_number}-{identity_in_first_chunk}.npy"), blob)
            # cv2.imwrite(os.path.join(folder, f"fly_{frame_number}-{identity_in_first_chunk}.png"), individual_frame)
            
            if frame_number >= DEBUG_FN:
                import ipdb; ipdb.set_trace()

            individual_frame, flip, debug_dict=postprocess_frame(config, video_object, individual_frame)

            # for k in debug_dict:
            #     np.save(os.path.join(folder, f"{k}-{frame_number}-{identity_in_first_chunk}.png"), debug_dict[k])
                

            def flip_image(blob, individual_frame, rot_angle):
                logger.debug("Flipping")
                rot_angle += 180
                rot_angle=rot_angle % 360
                
                blob._rotation_angle = rot_angle
                individual_frame=individual_frame[::-1,:]
                return individual_frame, rot_angle

            if flip:
                individual_frame, rot_angle= flip_image(blob, individual_frame, rot_angle)
                if rot_angle > 180:
                    rot_angle -= 360
                elif rot_angle < -180:
                    rot_angle += 360                   

                blob._rotation_angle = rot_angle

            # cv2.imwrite(os.path.join(folder, f"fly_processed_{frame_number}-{identity}.png"), individual_frame)
            cv2.imshow(f"individual frame", individual_frame)
            cv2.waitKey(1)

            assert individual_frame.shape[0] == height
            assert individual_frame.shape[1] == width

            # Write frame in video
            out.add_image(individual_frame.astype("uint8"), frame_number, frame_time)
            try:
                extra_data_row = extra_data.loc[index[frame_time]]
                out.add_extra_data(
                    blob_missing=False, 
                    light=extra_data_row["light"],
                    temperature=extra_data_row["temperature"],
                    humidity=extra_data_row["humidity"],
                    flip=flip,
                    nearby_flies=nearby_flies,
                )
            except KeyError as error:
                print(error)

            single_trajectory.append(centroid)
            
            
        single_trajectory = np.array(single_trajectory)
        np.save(
            os.path.join(out._basedir, f"{str(chunk).zfill(6)}.npy"),
            single_trajectory
        )

        # lists_of_blobs[chunk].save(lists_of_blobs._blobs_path)
        chunk+=1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def generate_individual_video(
    video_object, trajectories, identity, width, height
):
    # Initialize video writer
    out = initialize_video_writer(video_object, height, width, identity)
    # Initialize cap reader
    if len(video_object.video_paths) > 1:
        current_segment = 0
        cap = VideoCapture(video_object.video_paths[current_segment], chunk=video_object._chunk)
        start = video_object._episodes_start_end[current_segment][0]
    else:
        cap = VideoCapture(video_object.video_path, chunk=video_object._chunk)

    for frame_number in range(video_object.number_of_frames_in_chunk):
        # Update cap if necessary.
        if len(video_object.video_paths) > 1:
            segment_number = video_object.in_which_episode(frame_number)
            if current_segment != segment_number:
                print(video_object.video_paths[segment_number])
                cap = VideoCapture(
                    video_object.video_paths[segment_number],
                    chunk=video_object._chunk
                )
                start = video_object._episodes_start_end[segment_number][0]
                current_segment = segment_number
            cap.set("CAP_PROP_POS_FRAMES", frame_number - start)
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

def generate_individual_videos_rotation_invariant(video_object, trajectories, identity=None):
    
    list_of_blobs = ListOfBlobs.load(video_object.get_blobs_path("tracking"))
    
    # Cretae folder to store videos
    video_object.create_individual_videos_folder()
    # Calculate width and height of the video from the estimated body length
    height, width = compute_width_height_individual_video(video_object)
    n_jobs=conf.NUMBER_OF_JOBS_TO_GENERATE_INDIVIDUAL_VIDEOS
    logger.info(f"Generating individual videos using {n_jobs} jobs ...")
    
    Parallel(n_jobs=n_jobs)(
        delayed(generate_individual_video_rotation_invariant)(
            video_object,
            trajectories,
            list_of_blobs,
            identity=i,
            width=width,
            height=height
        )

        for i in range(
            video_object.user_defined_parameters["number_of_animals"]
        )
    )
    logger.info("Invididual videos generated")



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
    n_jobs=conf.NUMBER_OF_JOBS_TO_GENERATE_INDIVIDUAL_VIDEOS
    logger.info(f"Generating individual videos using {n_jobs} jobs ...")
    
    Parallel(n_jobs=n_jobs)(
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
