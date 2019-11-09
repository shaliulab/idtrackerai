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
import itertools
import numpy as np
from tqdm import tqdm

from idtrackerai.blob import Blob
from idtrackerai.preprocessing.model_area import ModelArea
from idtrackerai.utils.py_utils import interpolate_nans

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.list_of_blobs")


class ListOfBlobs(object):
    """ Collects all the instances of the class :class:`~blob.Blob` generated
    from the blobs extracted from the video during segmentation
    (see :mod:`~segmentation`)

    Attributes
    ----------

    blobs_in_video : list
        List of instances of :class:`~blob.Blob` segmented from the video and
        organised framewise
    number_of_frames : int
        number of frames in the video
    blobs_are_connected :bool
        True if the blobs have already being organised in fragments (see
        :class:`~fragment.Fragment`). False otherwise
    """

    def __init__(self, blobs_in_video=None, number_of_frames=None):
        self.blobs_in_video = blobs_in_video
        self.number_of_frames = len(self.blobs_in_video)
        self.blobs_are_connected = False

    def disconnect(self):
        """Reinitialise the previous and next list of blobs
        (see :attr:`~blob.Blob.next` and :attr:`~blob.Blob.previous`)
        """
        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                blob.next, blob.previous = [], []

    def connect(self):
        """Connects blobs in subsequent frames by computing their overlapping
        """
        logger.info("Connecting list of blob objects")
        for frame_i in tqdm(range(1, self.number_of_frames), desc='connecting blobs'):
            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)

    def reconnect(self):
        """Connects blobs in subsequent frames by computing their overlapping
        and sets blobs_are_connected to True
        """
        logger.info("re-Connecting list of blob objects")
        for frame_i in tqdm(range(1, self.number_of_frames), desc='re-connecting blobs'):
            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.fragment_identifier == blob_1.fragment_identifier:
                    blob_0.now_points_to(blob_1)
        self.blobs_are_connected = True

    def save(self, video, path_to_save=None, number_of_chunks=1):
        """save instance"""
        self.disconnect()
        logger.info("saving blobs list at %s" % path_to_save)
        np.save(path_to_save, self)
        if video.has_been_segmented and not video.has_been_preprocessed:
            self.connect()
        self.blobs_are_connected = True

    @classmethod
    def load(cls, video, path_to_load_blob_list_file):
        """Short summary.

        Parameters
        ----------
        video : <Video object>
            See :class:`~video.Video`
        path_to_load_blob_list_file : str
            path to load a list of blobs

        Returns
        -------
        <ListOfBlobs object>
            an instance of ListOfBlobs

        """
        logger.info("loading blobs list from %s" % path_to_load_blob_list_file)
        list_of_blobs = np.load(path_to_load_blob_list_file, allow_pickle=True).item()
        list_of_blobs.blobs_are_connected = False
        return list_of_blobs

    def compute_fragment_identifier_and_blob_index(self, number_of_animals):
        """Associates a unique fragment identifier to the fragments computed by
        overlapping see method
        :meth:`compute_overlapping_between_subsequent_frames` and sets the blob
        index as the hierarchy of the blob in the first frame of the fragment to
        which blob belongs to.

        Parameters
        ----------
        number_of_animals : int
            number of animals to be tracked
        """
        counter = 0
        possible_blob_indices = range(number_of_animals)

        for blobs_in_frame in tqdm(self.blobs_in_video, desc='assigning fragment identifier'):
            used_blob_indices = [blob.blob_index for blob in blobs_in_frame if blob.blob_index is not None]
            missing_blob_indices = list(set(possible_blob_indices).difference(set(used_blob_indices)))
            for blob in blobs_in_frame:
                if blob.fragment_identifier is None and blob.is_an_individual:
                    blob._fragment_identifier = counter
                    blob_index = missing_blob_indices.pop(0)
                    blob._blob_index = blob_index
                    blob.non_shared_information_with_previous = 1.
                    if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_an_individual:
                        blob.next[0]._fragment_identifier = counter
                        blob.next[0]._blob_index = blob_index
                        # blob.next[0].compute_overlapping_with_previous_blob()
                        if blob.next[0].is_an_individual_in_a_fragment:
                            blob = blob.next[0]

                            while len(blob.next) == 1 and blob.next[0].is_an_individual_in_a_fragment:
                                blob = blob.next[0]
                                blob._fragment_identifier = counter
                                blob._blob_index = blob_index
                                # compute_overlapping_with_previous_blob
                                # blob.compute_overlapping_with_previous_blob()

                            if len(blob.next) == 1 and len(blob.next[0].previous) == 1 and blob.next[0].is_an_individual:
                                blob.next[0]._fragment_identifier = counter
                                blob.next[0]._blob_index = blob_index
                                # blob.next[0].compute_overlapping_with_previous_blob()
                    counter += 1

        self.number_of_individual_fragments = counter
        logger.info("number_of_individual_fragments, %i" %counter)

    def compute_crossing_fragment_identifier(self):
        """Assign a unique identifier to fragments associated to a crossing
        """
        def propagate_crossing_identifier(blob, fragment_identifier):
            """Propagates the identifier throughout the entire fragment to which
            blob belongs to. It crawls by overlapping in both the past and
            future "overlapping histories" of blob

            Parameters
            ----------
            blob : <Blob object>
                an instance of :class:`~blob.Blob`
            fragment_identifier : int
                unique fragment identifier associated
            """
            assert blob.fragment_identifier is None
            blob._fragment_identifier = fragment_identifier
            cur_blob = blob

            while len(cur_blob.next) == 1 and len(cur_blob.next[0].previous) == 1 and cur_blob.next[0].is_a_crossing:
                cur_blob = cur_blob.next[0]
                cur_blob._fragment_identifier = fragment_identifier

            cur_blob = blob

            while len(cur_blob.previous) == 1 and len(cur_blob.previous[0].next) == 1 and cur_blob.previous[0].is_a_crossing:
                cur_blob = cur_blob.previous[0]
                cur_blob._fragment_identifier = fragment_identifier

        fragment_identifier = self.number_of_individual_fragments

        for blobs_in_frame in self.blobs_in_video:
            for blob in blobs_in_frame:
                if blob.is_a_crossing and blob.fragment_identifier is None:
                    propagate_crossing_identifier(blob, fragment_identifier)
                    fragment_identifier += 1
        logger.info("number_of_crossing_fragments: %i" % (fragment_identifier - self.number_of_individual_fragments))
        logger.info("total number of fragments: %i" % fragment_identifier)

    def compute_overlapping_between_subsequent_frames(self):
        """Computes overlapping between self and the blobs generated during
        segmentation in the next frames
        """
        def set_frame_number_to_blobs_in_frame(blobs_in_frame, frame_number):
            """Sets the attribute frame number to the instances of
            :class:`~blob.Blob` collected in `blobs_in_frame`

            Parameters
            ----------
            blobs_in_frame : list
                list of instances of :class:`~blob.Blob`
            frame_number : int
                number of the frame from which `blobs_in_frame` has been
                generated
            """
            for blob in blobs_in_frame:
                blob.frame_number = frame_number
        self.disconnect()

        for frame_i in tqdm(range(1, self.number_of_frames), desc='Connecting blobs '):
            set_frame_number_to_blobs_in_frame(self.blobs_in_video[frame_i-1], frame_i-1)

            for (blob_0, blob_1) in itertools.product(self.blobs_in_video[frame_i-1], self.blobs_in_video[frame_i]):
                if blob_0.overlaps_with(blob_1):
                    blob_0.now_points_to(blob_1)

        set_frame_number_to_blobs_in_frame(self.blobs_in_video[frame_i], frame_i)

    def compute_model_area_and_body_length(self, number_of_animals):
        """computes the median and standard deviation of all the blobs of the video
        and the median_body_length estimated from the diagonal of the bounding box.
        These values are later used to discard blobs that are not fish and potentially
        belong to a crossing.
        """
        #areas are collected throughout the entire video in the cores of the global fragments
        areas_and_body_length = np.asarray([(blob.area,blob.estimated_body_length)
                                            for blobs_in_frame in self.blobs_in_video
                                            for blob in blobs_in_frame
                                            if len(blobs_in_frame) == number_of_animals])
        if areas_and_body_length.shape[0] == 0:
            raise ValueError('There is not part in the video where the {} '
                             'animals are visible. '
                             'Try a different segmentation or check the '
                             'number of animals in the video.'.format(number_of_animals))
        median_area = np.median(areas_and_body_length[:,0])
        mean_area = np.mean(areas_and_body_length[:,0])
        std_area = np.std(areas_and_body_length[:,0])
        median_body_length = np.median(areas_and_body_length[:,1])
        return ModelArea(mean_area, median_area, std_area), median_body_length

    def apply_model_area_to_video(self, video, model_area, identification_image_size, number_of_animals):
        """Applies `model_area` to every blob extracted from video

        Parameters
        ----------
        video : <Video object>
            See :class:`~video.Video`
        model_area : <ModelArea object>
            See :class:`~model_area.ModelArea`
        identification_image_size : int
            size of the identification image (see
            :meth:`~blob.Blob.get_image_for_identification` and
            :attr:`~blob.Blob.image_for_identification`)
        number_of_animals : int
            number of animals to be tracked
        """
        def apply_model_area_to_blobs_in_frame(video, number_of_animals,
                                               blobs_in_frame, model_area,
                                               identification_image_size):
            """Applies `model_area` to the collection of Blob instances in
            `blobs_in_frame`

            Parameters
            ----------
            video : <Video object>
                See :class:`~video.Video`
            number_of_animals : int
                number of animals to be tracked
            blobs_in_frame : list
                list of instances of :class:`~blob.Blob`
            model_area : <ModelArea object>
                See :class:`~model_area.ModelArea`
            identification_image_size : int
                size of the identification image (see
                :meth:`~blob.Blob.get_image_for_identification` and
                :attr:`~blob.Blob.image_for_identification`)
            """
            number_of_blobs = len(blobs_in_frame)
            for blob in blobs_in_frame:
                blob.apply_model_area(video, number_of_animals, model_area,
                                      identification_image_size,
                                      number_of_blobs)
        for blobs_in_frame in tqdm(self.blobs_in_video, desc='Applying model area'):
            apply_model_area_to_blobs_in_frame(video, number_of_animals,
                                               blobs_in_frame, model_area,
                                               identification_image_size)

    # def get_data_plot(self):
    #     """Gets the areas of all the blobs segmented in the video
    #
    #     Returns
    #     -------
    #     list
    #         Areas of all the blobs segmented in the video
    #
    #     """
    #     return [blob.area for blobs_in_frame in self.blobs_in_video for blob in blobs_in_frame]


    def check_maximal_number_of_blob(self, number_of_animals, return_maximum_number_of_blobs = False):
        """Checks that the amount of blobs per frame is not greater than the
        number of animals to track

        Parameters
        ----------
        number_of_animals : int
            number of animals to be tracked

        Returns
        -------
        list
            List of indices of frames in which more blobs than animals to track
            have been segmented

        """
        maximum_number_of_blobs = 0
        frames_with_more_blobs_than_animals = []
        for frame_number, blobs_in_frame in enumerate(self.blobs_in_video):

            if len(blobs_in_frame) > number_of_animals:
                frames_with_more_blobs_than_animals.append(frame_number)
            maximum_number_of_blobs = len(blobs_in_frame) if len(blobs_in_frame) > maximum_number_of_blobs else maximum_number_of_blobs

        if len(frames_with_more_blobs_than_animals) > 0:
            logger.error('There are frames with more blobs than animals, this can be detrimental for the proper functioning of the system.')
            logger.error("Frames with more blobs than animals: %s" %str(frames_with_more_blobs_than_animals))
        if return_maximum_number_of_blobs:
            return frames_with_more_blobs_than_animals, maximum_number_of_blobs
        else:
            return frames_with_more_blobs_than_animals


    def update_from_list_of_fragments(self, fragments, fragment_identifier_to_index):
        """Updates the blobs objects generated from the video with the attributes
        computed for each fragment

        Parameters
        ----------
        fragments : list
            List of all the fragments
        fragment_identifier_to_index : int
            index to retrieve the fragment corresponding to a certain fragment
            identifier (see
            :meth:`~blob.Blob.compute_fragment_identifier_and_blob_index`)

        """
        attributes = ['identity',
                        'P2_vector',
                        'identity_corrected_solving_jumps',
                        'user_generated_identity', 'used_for_training',
                        'accumulation_step']

        for blobs_in_frame in tqdm(self.blobs_in_video, desc = 'updating list of blobs from list of fragments'):
            for blob in blobs_in_frame:
                fragment = fragments[fragment_identifier_to_index[blob.fragment_identifier]]
                [setattr(blob, '_' + attribute, getattr(fragment, attribute)) for attribute in attributes if hasattr(fragment, attribute)]

    def next_frame_to_validate(self, current_frame, direction):
        """Returns the next frame to be validated given the current frame an
        the direction to iterate

        Parameters
        ----------
        current_frame : int
            Frame from which to start checking for frames to validate
        direction : string
            Direction towards where to start checking. 'future' will check for
            upcoming frames, and 'past' for previous frames.

        Returns
        -------
        frame_number : int

        """
        logger.debug('next_frame_to_validate: {0}'.format(current_frame))

        if not (current_frame > 0 and current_frame < len(self.blobs_in_video)):
            raise Exception("The frame number must be between 0 and the number of frames in the video")
        if direction == 'future':
            blobs_in_frame_to_check = self.blobs_in_video[current_frame+1:]
        elif direction == 'past':
            blobs_in_frame_to_check = self.blobs_in_video[0:current_frame][::-1]
        for blobs_in_frame in blobs_in_frame_to_check:
            for blob in blobs_in_frame:
                if check_tracking(blobs_in_frame):
                    return blob.frame_number

    def interpolate_from_user_generated_centroids(self, video, identity,
                                                  start_frame, end_frame):
        """
        Interpolates linearly the centroids of the blobs of identity identity between
        start_frame and end_frame. The interpolation is done using the
        user_generated_centroids. The centroid of the blobs without
        user_generated_centroids are assumed to be nan and are interpolated
        accordingly.

        Parameters
        ----------
        identity : int
            Identity of the blobs to be interpolated
        start_frame : int
            Frame from which to start interpolation
        end_frame : int
            Frame where to end the interpolation
        """
        def check_extreme_blob(extreme_blob):
            if extreme_blob and len(extreme_blob) > 1:
                raise Exception('The identity must be unique in the first and last frames')
            elif not extreme_blob:
                raise Exception('There must be a blob with the identity to be\
                                 interpolated in the first and last frames')

        end_frame = end_frame + 1
        if start_frame >= end_frame:
            raise Exception('The initial frame has to be higher than the last frame.')

        first_blobs = [blob for blob in self.blobs_in_video[start_frame]
                       if identity in blob.final_identities]
        last_blobs = [blob for blob in self.blobs_in_video[end_frame-1]
                      if identity in blob.final_identities]

        check_extreme_blob(first_blobs)
        check_extreme_blob(last_blobs)

        # Check if they exited or are generated
        both_generated_blobs = first_blobs[0].is_a_generated_blob and last_blobs[0].is_a_generated_blob
        both_existed_blobs = not first_blobs[0].is_a_generated_blob and not last_blobs[0].is_a_generated_blob

        if not (both_existed_blobs or both_generated_blobs):
            raise Exception('The blobs in the first and last frames should be of the same type, \
                            either generated by the user or by segmentation')

        # Collect centroids of blobs with identity identity that were modified by the user
        centroids_to_interpolate = []
        blobs_of_id = []
        for blobs_in_frame in self.blobs_in_video[start_frame:end_frame]:
            possible_blobs = [blob for blob in blobs_in_frame if identity in blob.final_identities]

            if len(possible_blobs) == 1:
                blobs_of_id.append(possible_blobs[0])
                identity_index = possible_blobs[0].final_identities.index(identity)
                fixed_centroid = (None, None)
                if possible_blobs[0].user_generated_centroids is not None:
                    fixed_centroid = possible_blobs[0].user_generated_centroids[identity_index]
                if (fixed_centroid[0] is not None and fixed_centroid[0] > 0):
                    centroids_to_interpolate.append(fixed_centroid)
                elif possible_blobs[0].user_generated_identities is not None and possible_blobs[0].user_generated_identities[identity_index] is not None:
                    centroids_to_interpolate.append(possible_blobs[0].final_centroids[identity_index])
                else:
                    centroids_to_interpolate.append((np.nan, np.nan))
            elif not possible_blobs:
                blobs_of_id.append(None)
                centroids_to_interpolate.append((np.nan, np.nan))
            else:
                raise Exception('Make sure that the identnties of the user \
                                generated centroids (marked with u-) are unique \
                                in the interpolation interval.')

        if not (len(centroids_to_interpolate) == len(blobs_of_id) == end_frame - start_frame):
            raise Exception("The number of user generated centroids before interpolation does \
                            not match the number of frames interpolated.")
        if np.isnan(centroids_to_interpolate[0][0]) or np.isnan(centroids_to_interpolate[-1][0]):
            raise Exception("The first and last frame of the interpolation interval must contain a \
                            user generated centroid marked with the 'u-' prefix")

        centroids_to_interpolate = np.asarray(centroids_to_interpolate)
        # interpolate linearlry the centroids not generated by the user
        interpolate_nans(centroids_to_interpolate)
        # assign the new centroids to the blobs with identity identity.
        frames = range(start_frame, end_frame)
        for i, (blob, frame) in enumerate(zip(blobs_of_id, frames)):
            if blob is not None:
                identity_index = blob.final_identities.index(identity)
                if blob._user_generated_centroids is None:
                    blob._user_generated_centroids = [(None, None)]*len(blob.final_centroids)
                if blob._user_generated_identities is None:
                    blob._user_generated_identities = [None]*len(blob.final_centroids)
                blob._user_generated_centroids[identity_index] = tuple(centroids_to_interpolate[i, :])
                blob._user_generated_identities[identity_index] = identity
            else:
                if both_existed_blobs:
                    blob_index = np.argmin([candidate_blob.distance_from_countour_to(tuple(centroids_to_interpolate[i, :]))
                                            for candidate_blob in self.blobs_in_video[frame]])
                    nearest_blob = self.blobs_in_video[frame][blob_index]
                    nearest_blob.add_centroid(video, tuple(centroids_to_interpolate[i, :]), identity,
                                              apply_resolution_reduction=False)
                elif both_generated_blobs:
                    self.add_blob(video, frame,
                                  centroids_to_interpolate[i, :],
                                  identity)

        video.is_centroid_updated = True


    def reset_user_generated_identities_and_centroids(self, video, start_frame,
                                                      end_frame, identity=None):
        """
        Resets the identities and centroids generetad by the user to the ones assigned by the software

        Parameters
        ----------
        start_frame : int
            Frame from which to start reseting identities and centroids
        end_frame : int
            Frame where to end reseting identities and centroids
        identity : int, optional
            Identity of the blobs to be reseted (default None). If None,
            all the blobs are reseted
        """
        if start_frame > end_frame:
            raise Exception("Initial frame number must be smaller than the final frame number")
        if not (identity is None or identity >= 0):
            # missing identity <= self.number_of_animals but the attribute does not exist
            raise Exception("Identity must be None, zero or a positive integer")

        for blobs_in_frame in self.blobs_in_video[start_frame:end_frame+1]:
            if identity is None:
                # Reset all user generated identities and centroids
                for blob in blobs_in_frame:
                    if blob.is_a_generated_blob:
                        self.blobs_in_video[blob.frame_number].remove(blob)
                    else:
                        blob._user_generated_identities = None
                        blob._user_generated_centroids = None
            else:
                possible_blobs = [blob for blob in blobs_in_frame
                                  if identity in blob.final_identities]
                for blob in possible_blobs:
                    if blob.is_a_generated_blob:
                        self.blobs_in_video[blob.frame_number].remove(blob)
                    else:
                        indices = [i for i, final_id in enumerate(blob.final_identities)
                                   if final_id == identity]
                        for index in indices:
                            if blob._user_generated_centroids is not None:
                                blob._user_generated_centroids[index] = (None, None)
                            if blob._user_generated_identities is not None:
                                blob._user_generated_identities[index] = None

        video._is_centroid_updated = any([any([cent[0] is not None
                                               for cent in blob.user_generated_centroids])
                                          for blobs_in_frame in self.blobs_in_video
                                          for blob in blobs_in_frame
                                          if blob.user_generated_centroids is not None])

    def add_blob(self, video, frame_number, centroid, identity, apply_resolution_reduction=True):
        """
        Adds a Blob object the frame number.

        Adds a Blob object to a given frame_number with a given centroid and
        identity. Note that this Blob won't have most of the features (e.g.
        area, contour, fragment_identifier, bounding_box, ...). It is only
        intended to be used for validation and correction of trajectories.
        The new blobs generated are considered to be individuals.

        Args:
            frame_number (int): frame number where the Blob
            centroid (tuple): tuple with two float number (x, y).
            identity (int): identity of the blob

        Raises:
            Exception: If `identity` is greater of the number of animals in the
            video.

        """
        logger.info("Calling add_blob")
        if apply_resolution_reduction:
            centroid = (centroid[0]*video.resolution_reduction,
                        centroid[1]*video.resolution_reduction)
        if not (isinstance(centroid, tuple) and len(centroid) == 2):
            raise Exception("The centroid must be a tuple of length 2")
        if not (isinstance(identity, int) and identity > 0 and identity <= video.number_of_animals):
            raise Exception("The identity must be an integer between 1 and the number of animals in the video")

        new_blob = Blob(centroid=None, contour=None, area=None,
                        bounding_box_in_frame_coordinates=None)
        new_blob._user_generated_centroids = [(centroid[0], centroid[1])]
        new_blob._user_generated_identities = [identity]
        new_blob.frame_number = frame_number
        new_blob._is_an_individual = True
        new_blob._is_a_crossing = False
        new_blob._resolution_reduction = video.resolution_reduction
        new_blob.number_of_animals = video.number_of_animals
        self.blobs_in_video[frame_number].append(new_blob)
        video._is_centroid_updated=True



def check_tracking(blobs_in_frame):
    """Returns True if the list of blobs blobs_in_frame needs to be checked.

    Parameters
    ----------
    blobs_in_frame : list
        List of Blob objects

    Returns
    -------
    check_tracking_flag : boolean


    """
    there_are_crossings = any([blob.is_a_crossing for blob in blobs_in_frame]) #check whether there is a crossing in the frame
    missing_identity = any([None in blob.final_identities or 0 in blob.final_identities
                            for blob in blobs_in_frame]) # Check whether there is some missing identities (0 or None)
    return there_are_crossings or missing_identity
