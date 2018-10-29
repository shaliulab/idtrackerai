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


import os
import sys
import numpy as np
from tqdm import tqdm
from idtrackerai.list_of_blobs import ListOfBlobs
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.get_trajectories")

"""
Usage: get_trajectories.py

Contains tools to process tracked ListOfBlobs
and output trajectories as numpy files with dimensions:

    [Frame number x Individual number  x  coordinate (x,y)]

When a certain individual was not identified in the frame
a NaN appears instead of the coordinates
"""

def assign_point_to_identity(centroid, identity, frame_number, centroid_trajectories):
    """Populate the matrix of individual trajectories with the centroid of a
    selected Blob object (see :class:`~blob.Blob`)

    Parameters
    ----------
    centroid : tuple
        (x, y)
    identity : int
        Identity to be associated with the centroid
    frame_number : int
        Frame number in the tracked video
    centroid_trajectories : ndarray
        array of shape [number of frame in video x number of animals  x  2]

    Returns
    -------
    ndarray
        centroid_trajectories

    """
    if identity is not None and identity != 0:
        centroid_trajectories[frame_number, identity - 1, :] = centroid
    return centroid_trajectories


def assign_P2_to_identity(P2_vector, identity, frame_number, id_probabilities):
    """Populate the matrix of P2 trajectories with the argmax of the P2_vector
    of a selected Blob object (see :class:`~blob.Blob`)

    Parameters
    ----------
    P2_vector : array
        Array with P2 values for a Blob
    identity : int
        Identity to be associated with the centroid
    frame_number : int
        Frame number in the tracked video
    centroid_trajectories : ndarray
        array of shape [number of frame in video x number of animals  x  2]

    Returns
    -------
    ndarray
        centroid_trajectories

    """
    if identity is not None and identity != 0:
        id_probabilities[frame_number, identity - 1, :] = np.max(P2_vector)
    return id_probabilities


def produce_trajectories(blobs_in_video, number_of_frames, number_of_animals):
    """Produce trajectories array from ListOfBlobs

    Parameters
    ----------
    blobs_in_video : <ListOfBlobs object>
        See :class:`list_of_blobs.ListOfBlobs`
    number_of_frames : int
        Total number of frames in video
    number_of_animals : int
        Number of animals to be tracked

    Returns
    -------
    dict
        Dictionary with np.array as values (trajectories organised by identity)

    """
    centroid_trajectories = np.ones((number_of_frames, number_of_animals, 2)) * np.NaN
    id_probabilities = np.ones((number_of_frames, number_of_animals, 1)) * np.NaN

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video)):

        for blob in blobs_in_frame:

            if isinstance(blob.final_identity, int) or isinstance(blob.final_identity, np.integer):
                centroid_trajectories = assign_point_to_identity(blob.centroid,
                                                                blob.final_identity,
                                                                blob.frame_number,
                                                                centroid_trajectories)
                id_probabilities = assign_P2_to_identity(blob._P2_vector,
                                                        blob.final_identity,
                                                        blob.frame_number,
                                                        id_probabilities)
            elif isinstance(blob.final_identity, list):
                for identity, centroid in zip(blob.final_identity, blob.interpolated_centroids):
                    centroid_trajectories = assign_point_to_identity(centroid,
                                                                    identity,
                                                                    blob.frame_number,
                                                                    centroid_trajectories)

    return centroid_trajectories, id_probabilities


def produce_trajectories_wo_identities(blobs_in_video, number_of_frames, number_of_animals):
    centroid_trajectories = np.ones((number_of_frames, number_of_animals, 2))*np.nan
    identifiers_prev = np.arange(number_of_animals).astype(np.float32)
    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video, "creating trajectories")):
        if frame_number != len(blobs_in_video)-1:
            identifiers_next = [b.fragment_identifier for b in blobs_in_video[frame_number+1]]
        else:
            identifiers_next = [b.fragment_identifier for b in blobs_in_video[frame_number]]
        for blob_number, blob in enumerate(blobs_in_frame):
            if blob.is_an_individual:
                if blob.fragment_identifier in identifiers_prev:
                    column = np.where(identifiers_prev == blob.fragment_identifier)[0][0]
                else:
                    column = np.where(np.isnan(identifiers_prev))[0][0]
                    identifiers_prev[column] = blob.fragment_identifier

                blob._identity = int(column+1)
                centroid_trajectories[frame_number, column, :] = blob.centroid

                if blob.fragment_identifier not in identifiers_next:
                    identifiers_prev[column] = np.nan
    return centroid_trajectories, None


def produce_output_dict(blobs_in_video, video):
    """Outputs the dictionary with keys: trajectories, git_commit, video_path,
    frames_per_second

    Parameters
    ----------
    blobs_in_video : list
        List of all blob objects (see :class:`~blob.Blobs`) generated by
        considering all the blobs segmented from the video
    video : <Video object>
        See :class:`~video.Video`

    Returns
    -------
    dict
        Output dictionary containing trajectories as values

    """
    if not video.track_wo_identities:
        centroid_trajectories, id_probabilities = \
            produce_trajectories(blobs_in_video, video.number_of_frames,
                                 video.number_of_animals)
    else:
        video._number_of_animals = np.max([len(bf) for bf in blobs_in_video])
        centroid_trajectories, id_probabilities = \
            produce_trajectories_wo_identities(blobs_in_video,
                                                video.number_of_frames,
                                                video.number_of_animals)
    output_dict = {'trajectories': centroid_trajectories,
                   'id_probabilities': id_probabilities,
                   'git_commit': video.git_commit,
                   'video_path': video.video_path,
                   'frames_per_second': video.frames_per_second,
                   'body_length': video.median_body_length}
    return output_dict

if __name__ == "__main__":
    from idtrackerai.utils.GUI_utils import selectDir
    # #SIMPLE USAGE EXAMPLE
    # BLOB_FILE_NAME = "blobs_collection.npy"
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    logger.info("loading video object...")
    video = np.load(video_path).item(0)
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)

    video.create_trajectories_wo_gaps_folder()
    logger.info("Generating trajectories. The trajectories files are stored in %s" %video.trajectories_wo_gaps_folder)
    trajectories_wo_gaps_file = os.path.join(video.trajectories_wo_gaps_folder, 'output_dict.npy')
    trajectories_wo_gaps = produce_output_dict(list_of_blobs.blobs_in_video, video)
    np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)
    logger.info("Saving trajectories")
    video._has_trajectories_wo_gaps = True
    video.save()

    #trajectories = produce_trajectories(blobs_list.blobs_in_video, video.number_of_frames, video.number_of_animals)
    #save_trajectories(trajectories, trajectories_folder)
