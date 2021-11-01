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

import cv2
import h5py
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger("__main__.blob")


class Blob(object):
    """Represents a segmented blob (collection of pixels) from a given frame.

    A blob can represent a single animal or multiple animals during an
    occlusion or crossing.

    Parameters
    ----------

    centroid : tuple
        Tuple (float, float) with the pixels coordinates of the blob center
        of mass.
    contour : numpy array
        Array with the points that define the contour of the blob. The
        array is of the form [[[x1,y1]],[[x2,y2]],...,[[xn,yn]]].
    area : int
        Number of pixels that conform the blob.
    bounding_box_in_frame_coordinates : list
        Coordinates of the bounding box that encloses the segmented blob
        [(x, y), (x + bounding_box_width, y + bounding_box_height)]. Note
        that this bouding box is expanded some pixels with respect to the
        original bounding box obtained with OpenCV.
    bounding_box_image : numpy array or None, optional
        Bonding box image that encloses the blob, by default None.
    bounding_box_images_path : str, optional
        Path to the file where the bounding box images are stored, by
        default None.
    estimated_body_length : float, optional
        Body length of the animal estimated from the diagonal of the
        original bounding box, by default None.
    pixels : list, optional
        List of pixels that define the blob, by default None.
    number_of_animals : int, optional
        Number of animals in the video as defined by the user,
        by default None.
    frame_number : int, optional
        Frame number in the video from which the blob was segmented,
        by default None.
    frame_number_in_video_path : int, optional
        Frame number in the video path from which the blob was segmented,
        by default None.
    in_frame_index : int, optional
        Index of the blob in the frame where it was segmented,
        by default None. This index comes from OpenCV and is defined by the
        hierarchy of the countours found in the frame.
    pixels_path : str, optional
        Path to the file were the pixels of the blob are stored,
        by default None.
    video_height : int, optional
        Height of the video considering the resolution reduction factor,
        by default None.
    video_width : [type], optional
        Width of the video considering the resolution reduction factor,
            by default None.
    video_path : str, optional
        Path to the video file from which the blob was segmented,
        by default None.
    pixels_are_from_eroded_blob : bool, optional
        Flag to indicate if the pixels of the blobs come from from an
        eroded blob, by default False.
    resolution_reduction : float, optional
        Resolution reductio factor as defined by the user, by default 1.0.
    """

    def __init__(
        self,
        centroid,
        contour,
        area,
        bounding_box_in_frame_coordinates,
        bounding_box_image=None,
        bounding_box_images_path=None,
        estimated_body_length=None,
        pixels=None,
        number_of_animals=None,
        frame_number=None,
        frame_number_in_video_path=None,
        in_frame_index=None,
        pixels_path=None,
        video_height=None,
        video_width=None,
        video_path=None,
        pixels_are_from_eroded_blob=False,
        resolution_reduction=1.0,
    ):
        # Attributed from the input arguments
        self.centroid = centroid
        self.contour = contour
        self.area = area
        self.bounding_box_in_frame_coordinates = (
            bounding_box_in_frame_coordinates
        )
        self._bounding_box_image = bounding_box_image
        self.bounding_box_images_path = bounding_box_images_path
        self.estimated_body_length = estimated_body_length
        self._pixels = pixels
        self.number_of_animals = number_of_animals
        self.frame_number = frame_number
        self.frame_number_in_video_path = frame_number_in_video_path
        self.in_frame_index = in_frame_index
        self._pixels_path = pixels_path
        self.video_height = video_height
        self.video_width = video_width
        self.video_path = video_path
        self.pixels_are_from_eroded_blob = pixels_are_from_eroded_blob
        self._resolution_reduction = resolution_reduction

        # Attributes populated at different points of the tracking
        # During crossing detection
        self.identification_image_index = None
        self.next = []
        self.previous = []
        self._is_an_individual = False
        self._is_a_crossing = False
        # During fragmentation
        self._fragment_identifier = None
        self._blob_index = None
        # During the cascade of training and identification protocols
        self._used_for_training = None
        self._accumulation_step = None
        self._identity = None
        # During postprocessing and interpolation of crossings
        self.interpolated_centroids = None
        self._was_a_crossing = False
        self._identities_corrected_closing_gaps = None
        self._identity_corrected_solving_jumps = None
        self.has_eroded_pixels = False
        self._eroded_pixels = None
        # During validation
        self._user_generated_identities = None
        self._user_generated_centroids = None

    @property
    def bounding_box_image(self):
        """Image cropped from the original video that contains the blob.

        This image is used later to extract the `image_for_identification` that
        will be used to train and evaluate the crossing detector CNN and the
        identification CNN. This image can either be stored in the object
        (in RAM), or be stored in in a file (in DISK), or not be stored at all,
        in which case it is recomupted from the original video.

        Returns
        -------
        numpy array (uint8)
            Image cropped from the video containing the pixels that represent
            the blob.
        """
        if self._bounding_box_image is not None:
            return self._bounding_box_image
        elif self.bounding_box_images_path is not None and os.path.isfile(
            self.bounding_box_images_path
        ):
            with h5py.File(self.bounding_box_images_path, "r") as f:
                return f[
                    str(self.frame_number) + "-" + str(self.in_frame_index)
                ][:]
        else:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(1, self.frame_number_in_video_path)
            ret, frame = cap.read()
            bb = self.bounding_box_in_frame_coordinates
            return frame[bb[0][1] : bb[1][1], bb[0][0] : bb[1][0], 0]

    @property
    def pixels(self):
        """List of pixels that define the blob as determined from the
        segmentation parameters defined by the user.

        Note that pixels are represented as linearized coordinates of the
        frame (considering the resolution reduction factor), i.e. each pixel
        is a single integer.

        The pixels can either be stored in the object (in RAM), or be stored
        in a file (in DISK), or not be stored at all, in which case they are
        recomputed from the contour and the original frame of the video.

        Returns
        -------
        List
            List of integers indicating the linarized indices of the pixels
            that represent the blob.
        """
        if self._pixels is not None:
            return self._pixels
        elif self._pixels_path is not None and os.path.isfile(
            self._pixels_path
        ):
            with h5py.File(self._pixels_path, "r") as f:
                if not self.pixels_are_from_eroded_blob:
                    dataset_name = (
                        str(self.frame_number) + "-" + str(self.in_frame_index)
                    )
                else:
                    dataset_name = (
                        str(self.frame_number)
                        + "-"
                        + str(self.in_frame_index)
                        + "-eroded"
                    )
                return f[dataset_name][:]
        else:
            cimg = np.zeros((self.video_height, self.video_width))
            cv2.drawContours(cimg, [self.contour], -1, color=255, thickness=-1)
            pts = np.where(cimg == 255)
            pixels = np.asarray(list(zip(pts[0], pts[1])))
            pixels = np.ravel_multi_index(
                [pixels[:, 0], pixels[:, 1]],
                (self.video_height, self.video_width),
            )
            return pixels

    @property
    def eroded_pixels(self):
        """Pixels of the blob after erosion.

        The erosion is performed during the crossings interpolation process.

        The pixels can either be stored in the object (in RAM), or be stored
        in a file (in DISK), or not be stored at all, in which case they are
        computed

        Returns
        -------
        List
            List of integers indicating the linarized indices of the pixels
            that represent the blob after the erosion process.
        """
        if self._eroded_pixels is not None:
            return self._pixels
        elif self._pixels_path is not None and os.path.isfile(
            self._pixels_path
        ):
            with h5py.File(self._pixels_path, "r") as f:
                return f[
                    str(self.frame_number)
                    + "-"
                    + str(self.in_frame_index)
                    + "-eroded"
                ][:]
        else:
            # TODO: Check that if the blob is eroded the contour is updated.
            # Otherwise this pixels will be the original ones and no the
            # eroded ones.
            cimg = np.zeros((self.video_height, self.video_width))
            cv2.drawContours(cimg, [self.contour], -1, color=255, thickness=-1)
            pts = np.where(cimg == 255)
            pixels = np.asarray(list(zip(pts[0], pts[1])))
            pixels = np.ravel_multi_index(
                [pixels[:, 0], pixels[:, 1]],
                (self.video_height, self.video_width),
            )
            return pixels

    @eroded_pixels.setter
    def eroded_pixels(self, eroded_pixels):
        if self._pixels_path is not None:  # is saving in disk
            with h5py.File(self._pixels_path, "a") as f:
                dataset_name = (
                    str(self.frame_number)
                    + "-"
                    + str(self.in_frame_index)
                    + "-eroded"
                )
                if dataset_name in f:
                    del f[dataset_name]
                f.create_dataset(dataset_name, data=eroded_pixels)
        else:
            self._eroded_pixels = eroded_pixels
        self.has_eroded_pixels = True

    @property
    def fragment_identifier(self):
        """
        Integer indicating the index of the fragment that contains the blob

        See Also
        --------
        :class:`~fragment.Fragment`
        """
        return self._fragment_identifier

    @property
    def is_an_individual(self):
        """Flag indicating whether the blob represent a single animal.

        The private attribute `_is_an_individual` is assigned during the
        crossings detection process.
        """
        return self._is_an_individual

    @is_an_individual.setter
    def is_an_individual(self, value: bool):
        assert isinstance(value, bool)
        self._is_an_individual = value
        self._is_a_crossing = not value

    # TODO: consider removing this property, as it is redundant.
    @property
    def is_a_crossing(self):
        """Flag indicating whether the blob represents two or more animals
        together.

        This attribute is the negative of `is_an_individual` and is set at
        the same time as is an individual
        """
        return self._is_a_crossing

    @property
    def was_a_crossing(self):
        """Flag indicating whether the blob was created after splitting a
        crossing blob during the crossings interpolation process
        """
        return self._was_a_crossing

    # TODO: consider adding a feature in the validation GUI to add a contour.
    @property
    def is_a_generated_blob(self):
        """Flag indicating whether the blob was created by the user.

        Blobs created by the users during the validation process do not have
        a contour as only the centroid is created
        """
        return self.contour is None

    def check_for_multiple_next_or_previous(self, direction=None):
        """Flag indicating if the blob has multiple blobs in its past or future
        overlapping history

        This method is used to check whether the blob is a crossing.

        Parameters
        ----------
        direction : str
            "previous" or "next". If "previous" the past overlapping history
            will be checked in order to find out if the blob will split in the
            past.
            Symmetrically, if "next" the future overlapping history of the blob
            will be checked.

        Returns
        -------
        bool
            If True the blob splits into two or multiple overlapping blobs in
            its "past" or "future" history, depending on the parameter
            "direction".
        """
        current = getattr(self, direction)[0]

        while len(getattr(current, direction)) == 1:

            current = getattr(current, direction)[0]
            if len(getattr(current, direction)) > 1:
                return True

        return False

    def check_for_crossing_in_next_or_previous(self, direction=None):
        """Flag indicating if the blob has a crossing in its past or future
        overlapping history

        This method is used to check whether the blob is an individual.

        Parameters
        ----------
        direction : str
            "previous" or "next". If "previous" the past overlapping history
            will be checked in order to find out if the blob ends up in a
            crossing.
            Symmetrically, if "next" the future overlapping history of the blob
            will be checked.

        Returns
        -------
        bool
            If True the blob has a crossing in its "past" or "future" history,
            depending on the parameter `direction`.
        """
        opposite_direction = "next" if direction == "previous" else "previous"
        current = getattr(self, direction)[0]

        while len(getattr(current, direction)) == 1:

            current = getattr(current, direction)[0]
            if (
                len(getattr(current, opposite_direction)) > 1
                and current.is_a_crossing
            ):
                return True
        return False

    def is_a_sure_individual(self):
        """Flag indicating that the blob is a sure individual according to
        some heuristics and it can be used to train the crossing detector CNN.

        Returns
        -------
        bool
        """
        if (
            self.is_an_individual  # assigned in _apply_area_and_unicity_heuristics
            and len(self.previous) == 1
            and len(self.next) == 1
            and len(self.next[0].previous) == 1
            and len(self.previous[0].next) == 1
        ):
            has_crossing_in_past = self.check_for_crossing_in_next_or_previous(
                "previous"
            )
            has_crossing_in_future = (
                self.check_for_crossing_in_next_or_previous("next")
            )
            if has_crossing_in_past and has_crossing_in_future:
                return True
        else:
            return False

    def is_a_sure_crossing(self):
        """Flag indicating that the blob is a sure crossing according to
        some heuristics and it can be used to train the crossing detector CNN.

        Returns
        -------
        bool
        """
        if self.is_a_crossing and (
            len(self.previous) > 1 or len(self.next) > 1
        ):
            return True
        elif (
            self.is_a_crossing
            and len(self.previous) == 1
            and len(self.next) == 1
        ):
            has_multiple_previous = self.check_for_multiple_next_or_previous(
                "previous"
            )
            has_multiple_next = self.check_for_multiple_next_or_previous(
                "next"
            )
            if has_multiple_previous and has_multiple_next:
                return True
        else:
            return False

    def overlaps_with(self, other):
        """Computes whether the pixels in `self` intersect with the pixels in
        `other`

        Parameters
        ----------
        other : <Blob object>
            An instance of the class Blob

        Returns
        -------
        bool
            True if the lists of pixels of both blobs have non-empty
            intersection
        """
        overlaps = False
        intersection = np.isin(self.pixels, other.pixels, assume_unique=True)
        if any(intersection):
            overlaps = True
        return overlaps

    def now_points_to(self, other):
        """Given two consecutive blob objects updates their respective
        overlapping histories

        Parameters
        ----------
        other : <Blob object>
            An instance of the class Blob
        """
        self.next.append(other)
        other.previous.append(self)

    def squared_distance_to(self, other):
        """Returns the squared distance from the centroid of self to the
        centroid of `other`

        Parameters
        ----------
        other : <Blob object> or tuple
            An instance of the class Blob or a tuple (x,y)

        Returns
        -------
        float
            Squared distance between centroids
        """
        if isinstance(other, Blob):
            return np.sum(
                (np.asarray(self.centroid) - np.asarray(other.centroid)) ** 2
            )
        elif isinstance(other, (tuple, list, np.ndarray)):
            return np.sum((np.asarray(self.centroid) - np.asarray(other)) ** 2)

    def distance_from_countour_to(self, point):
        r"""Returns the distance between `point` and the closest
        point in the contour of the blob.

        Parameters
        ----------
        point : tuple
            (x,y)

        Returns
        -------
        float
            Smallest distance between `point` and the contour of the blob.
        """
        return np.abs(cv2.pointPolygonTest(self.contour, point, True))

    @property
    def resolution_reduction(self):
        """Resolution reduction factor defined by the user"""
        if hasattr(self, "_resolution_reduction"):
            return self._resolution_reduction
        else:
            return 1.0

    @property
    def used_for_training(self):
        """Flag indicating if the blob has been used to train the
        identification CNN
        """
        return self._used_for_training

    @property
    def accumulation_step(self):
        """Integer indicating the accumulation step in which the blob
        was assign by the cascade of training and identification protocols
        """
        return self._accumulation_step

    @property
    def is_in_a_fragment(self):
        """Boolean indicating if the blob is in a fragment, i.e. is not an
        isolated blob or is in the extreme of a fragment
        """
        # TODO: check if it also indicates that it in an extreme of a fragment
        return len(self.previous) == len(self.next) == 1

    @property
    def is_an_individual_in_a_fragment(self):
        """Boolean indicating if the blob is in an individual fragment"""
        return self.is_an_individual and self.is_in_a_fragment

    @property
    def blob_index(self):
        """Blob index at the segmentation step (comes from the find contours
        function of OpenCV)
        """
        return self._blob_index

    @blob_index.setter
    def blob_index(self, new_blob_index):
        if self.is_an_individual_in_a_fragment:
            self._blob_index = new_blob_index

    def in_a_global_fragment_core(self, blobs_in_frame):
        """Boolean indicating if a blob is in the core of a globalfragment.

        A blob in a frame is in the core of a global fragment if in
        that frame there are as many blobs as number of animals to track

        Parameters
        ----------
        blobs_in_frame : list
            List of Blob objects representing the animals segmented in the
            frame self.frame_number

        Returns
        -------
        bool
            True if the blob is in the core of a global fragment
        """
        return len(blobs_in_frame) == self.number_of_animals

    @property
    def identity(self):
        """Identity of the blob assigned during the identification process"""
        return self._identity

    @property
    def user_generated_identities(self):
        """List of identities of the blob some of which might have been give
        by a user during the validation process
        """
        return self._user_generated_identities

    @property
    def identity_corrected_solving_jumps(self):
        """Identity of the blob after correcting impossible velocity jumps"""
        return self._identity_corrected_solving_jumps

    @property
    def identities_corrected_closing_gaps(self):
        """Identity of the blob after crossings interpolation"""
        return self._identities_corrected_closing_gaps

    @property
    def assigned_identities(self):
        """Identities assigned to the blob during the tracking process"""
        if self.identities_corrected_closing_gaps is not None:
            return self.identities_corrected_closing_gaps
        elif self.identity_corrected_solving_jumps is not None:
            return [self.identity_corrected_solving_jumps]
        return [self.identity]

    @property
    def final_identities(self):
        """Identities of the blob after the tracking process and after
        potential modifications by the users during the validation procedure.
        """
        if self.user_generated_identities is not None:
            # Note that sometimes len(user_generated_identities)
            # > len(assigned_identities)
            final_identities = []
            for i, user_generated_identity in enumerate(
                self.user_generated_identities
            ):
                if user_generated_identity is None and i < len(
                    self.assigned_identities
                ):
                    final_identities.append(self.assigned_identities[i])
                elif (
                    user_generated_identity is not None
                    and user_generated_identity >= 0
                ):
                    final_identities.append(user_generated_identity)

            return final_identities
        return self.assigned_identities

    def save_image_for_identification(
        self, identification_image_size, height, width, file_path
    ):
        """Saves in disk the image that will be used to train and evaluate the
        crossing detector CNN and the identification CNN.

        This also updates the `identification_image_index` and the `episode`
        attributes. This helps to load the image from the correct `file_path`.

        Parameters
        ----------
        identification_image_size : tuple
            Tuple of integers (height, width, channels).
        height : int
            Video height considering the resolution reduction factor.
        width : int
            Video width considering the resolution reduction factor.
        file_path : str
            Path to the hdf5 file where the images will be stored.
        """
        image_for_identification = self.get_image_for_identification(
            identification_image_size,
            height,
            width,
        )

        # For RAM optimization
        with h5py.File(file_path, "a") as f:
            dset = f["identification_images"]
            i = dset.shape[0]
            dset.resize(
                (
                    i + 1,
                    image_for_identification.shape[1],
                    image_for_identification.shape[1],
                )
            )
            dset[i, ...] = image_for_identification
        self.identification_image_index = i
        self.episode = int(
            os.path.basename(file_path).split(".")[0].split("_")[-1]
        )

    def get_image_for_identification(
        self,
        identification_image_size,
        height,
        width,
    ):
        """Gets the image used to train and evaluate the crossing detector CNN
        and the identification CNN.

        Parameters
        ----------
        identification_image_size : tuple
            Dimensions of the identification image (height, widht, channels).
            Channels is always 1 as images in color are still not considered.
        height : int
            Video height considering resolution reduction factor.
        width : int
            Video width considering resolution reduction factor.

        Returns
        -------
        ndarray
            Square image with black background used to train the crossings
            detector CNN and the identifiactio CNN.
        """

        return self._get_image_for_identification(
            height,
            width,
            self.bounding_box_image,
            self.pixels,
            self.bounding_box_in_frame_coordinates,
            identification_image_size[0],
        )

    # TODO: Consider changing the name of image_for_identification
    # it is also the image for the crossing detector.
    @staticmethod
    def _get_image_for_identification(
        height,
        width,
        bounding_box_image,
        pixels,
        bounding_box_in_frame_coordinates,
        image_size,
    ):
        """It generates the image that will be used to train and evaluate the
        crossings detector CNN and the identification CNN.

        Parameters
        ----------
        height : int
            Frame height
        width : int
            Frame width
        bounding_box_image : ndarray
            Images cropped from the frame by considering the bounding box
            associated to a blob
        pixels : list
            List of pixels associated to a blob
        bounding_box_in_frame_coordinates : list
            [(x, y), (x + bounding_box_width, y + bounding_box_height)]
        image_size : int
            Size of the width and height of the square identification image

        Returns
        -------
        ndarray
            Square image with black background used to train the crossings
            detector CNN and the identifiactio CNN.

        """
        bounding_box_image = _mask_background_pixels(
            height,
            width,
            bounding_box_image,
            pixels,
            bounding_box_in_frame_coordinates,
        )
        pca = PCA()
        pxs = np.unravel_index(pixels, (height, width))
        pxs1 = np.asarray(list(zip(pxs[1], pxs[0])))
        pca.fit(pxs1)
        rot_ang = (
            np.arctan(pca.components_[0][1] / pca.components_[0][0])
            * 180
            / np.pi
            + 45
        )
        # we substract 45 so that the fish is aligned in the diagonal.
        # This way we have smaller frames
        center = (pca.mean_[0], pca.mean_[1])
        center = _transform_to_bbox_coordinates(
            center, bounding_box_in_frame_coordinates
        )
        center = np.array([int(center[0]), int(center[1])])

        # rotate
        diag = np.sqrt(
            np.sum(np.asarray(bounding_box_image.shape) ** 2)
        ).astype(int)
        diag = (diag, diag)
        M = cv2.getRotationMatrix2D(tuple(center), rot_ang, 1)
        minif_rot = cv2.warpAffine(
            bounding_box_image,
            M,
            diag,
            borderMode=cv2.BORDER_CONSTANT,
            flags=cv2.INTER_CUBIC,
        )

        crop_distance = int(image_size / 2)
        x_range = range(center[0] - crop_distance, center[0] + crop_distance)
        y_range = range(center[1] - crop_distance, center[1] + crop_distance)
        image_for_identification = minif_rot.take(
            y_range, mode="wrap", axis=0
        ).take(x_range, mode="wrap", axis=1)

        return image_for_identification

    @property
    def contour_full_resolution(self):
        """Blob contour coordinates in pixels without considering the
        resolution reduction factor, i.e. in the full resolution of the video.

        Returns
        -------
        numpy array.
            Coordinates in pixels of the points in the blob contour
        """
        if self.contour is not None:
            return (self.contour / self.resolution_reduction).astype(np.int32)
        else:
            return None

    @property
    def bounding_box_full_resolution(self):
        """Bounding box cordinates without considering the resolution reduction
         factor, i.e. in the full resolution of the video.

        Returns
        -------
        numpy array, or None
            Bounding box coordinates of the blob in full resolution of the
            video, [(x, y), (x + bounding_box_width, y + bounding_box_height)].
        """
        if self.bounding_box_in_frame_coordinates is not None:
            bounding_box_full_resolution = (
                np.asarray(self.bounding_box_in_frame_coordinates)
                / self.resolution_reduction
            ).astype(int)
            return tuple(map(tuple, bounding_box_full_resolution))
        else:
            # TODO: Check when this happens and document.
            return None

    # Centroids
    @property
    def user_generated_centroids(self):
        """List of centroids generated by the user during the validation
        processes.

        Returns
        -------
        list or None
            If the user has not generated any centroid for this blob, it
            returns None.
        """

        if hasattr(self, "_user_generated_centroids"):
            return self._user_generated_centroids
        return None

    @property
    def assigned_centroids(self):
        """Centroids assigned to the blob during the tracking process.

        It considers the default centroid of the blob at segmentation time
        or new centroids added to the blob during the interpolation of the
        crossings.

        Returns
        -------
        list
            List of pairs (x, y) indicating the position of each individual
            in the blob.
        """
        if (
            hasattr(self, "interpolated_centroids")
            and self.interpolated_centroids is not None
        ):
            assert isinstance(self.interpolated_centroids, list)
            return self.interpolated_centroids
        return [self.centroid]

    @property
    def final_centroids(self):
        """List of the animal/s centroid/s in the blob, considering the
        potential centroids that might have been aded by the user during
        the validation.

        By default the centroid will be the center of mass of the blob of
        pixels defined by the blob. It can be different if the user modified
        the default centroid during validation or generated more centroids.

        Returns
        -------
        list
            List of tuples (x, y) indicating the centroids of the blob.
        """
        if self.user_generated_centroids is not None:
            # Note that sometimes len(user_generated_centroids) >
            # len(assigned_centroids)
            final_centroids = []
            for i, user_generated_centroid in enumerate(
                self.user_generated_centroids
            ):
                # TODO: missing else
                if user_generated_centroid[0] is None and i < len(
                    self.assigned_centroids
                ):
                    final_centroids.append(self.assigned_centroids[i])
                elif (
                    user_generated_centroid[0] is not None
                    and user_generated_centroid[0] > 0
                ):
                    final_centroids.append(user_generated_centroid)
            return final_centroids
        return self.assigned_centroids

    @property
    def final_centroids_full_resolution(self):
        """Returns the centroids considering the full resolution of the frame.

          Note that after validation a blob can have more than one centroid.
          For example, this can happen in a crossing blob with missing
          centroids.

        Returns
        -------
          List of tuples (x, y), where x, and y are coordinates in the full
          resolution of the video frame.
        """
        return [
            (
                centroid[0] / self.resolution_reduction,
                centroid[1] / self.resolution_reduction,
            )
            for centroid in self.final_centroids
        ]

    # Methods used to modify the blob attributes during the validation of the
    # trajectories obtained after tracking.
    # TODO: Consider removing this from this class. Maybe move to valdiation.
    def removable_identity(self, identity_to_remove, blobs_in_frame):
        """[Validation] Checks if the identity can be removed.

        Parameters
        ----------
        identity_to_remove : int
            Identity to be removed
        blobs_in_frame : list
            List of Blob objects in the frame where the identity is going to
            be removed

        Returns
        -------
        bool
            True if the identity can be removed.
        """
        for blob in blobs_in_frame:
            if blob != self:
                if (
                    identity_to_remove in blob.final_identities
                ):  # Is duplicated in another blob
                    return True
            else:
                if (
                    blob.final_identities.count(identity_to_remove) > 1
                ):  # Is duplicated in the same blob
                    return True
        return False

    def update_centroid(self, video, old_centroid, new_centroid, identity):
        """[Validation] Updates the centroid of the blob.

        Parameters
        ----------
        video : idtrackerai.video.Video
            Instance of Video object
        old_centroid : tuple
            Centroid to be updated
        new_centroid : tuple
            Coordinates of the new centroid
        identity : int
            Identity of the centroid to be updated
        """
        logger.info("Calling update_centroid")
        video.is_centroid_updated = True
        old_centroid = (
            old_centroid[0] * self.resolution_reduction,
            old_centroid[1] * self.resolution_reduction,
        )
        new_centroid = (
            new_centroid[0] * self.resolution_reduction,
            new_centroid[1] * self.resolution_reduction,
        )

        if not (isinstance(old_centroid, tuple) and len(old_centroid) == 2):
            raise Exception("The old_centroid must be a tuple of length 2")
        if not (isinstance(new_centroid, tuple) and len(new_centroid) == 2):
            raise Exception("The new centroid must be a tuple of length 2")

        if self.user_generated_centroids is None:
            self._user_generated_centroids = [(None, None)] * len(
                self.final_centroids
            )
        if self.user_generated_identities is None:
            self._user_generated_identities = [None] * len(
                self.final_centroids
            )

        try:
            if old_centroid in self.user_generated_centroids:
                centroid_index = self.user_generated_centroids.index(
                    old_centroid
                )
                identity = self.user_generated_identities[centroid_index]
            elif old_centroid in self.assigned_centroids:
                if self.assigned_centroids.count(old_centroid) > 1:
                    centroid_index = self.assigned_identities.index(identity)
                else:
                    centroid_index = self.assigned_centroids.index(
                        old_centroid
                    )
                identity = self.assigned_identities[centroid_index]
            else:
                raise Exception(
                    "There is no centroid with the values of old_centroid"
                )
        except ValueError:
            raise Exception(
                "There is no centroid with the values of old_centroid"
            )

        self.user_generated_centroids[centroid_index] = new_centroid
        self.user_generated_identities[centroid_index] = identity

        video.is_centroid_updated = True

    def delete_centroid(
        self,
        video,
        identity,
        centroid,
        blobs_in_frame,
        apply_resolution_reduction=True,
    ):
        """[Validation] Deletes a centroid of the blob.

        Parameters
        ----------
        video : idtrackerai.video.Video
            Instance of Video object
        identity : int
            Identity of the centroid to be deleted
        centroid : tuple
            Centroid to be deleted from the blob
        blobs_in_frame : List
            List of the blobs in the frame
        apply_resolution_reduction : bool, optional
            Whether to consider the resolution reduction factor when
            adding the centroid, by default True. Note that the video is showed
            as full resolution in the validation GUI, but all centroids of
            the blobs consider the resolution reduction factor.

        Raises
        ------
        Exception
            If the centroid is not a tuple of length 2
        Exception
            If the identity is unique in the frame
        Exception
            If it is the last centroid of the blob
        """
        logger.info("Calling delete_centroid")
        if not (isinstance(centroid, tuple) and len(centroid) == 2):
            raise Exception("The centroid must be a tuple of length 2")

        if not self.removable_identity(identity, blobs_in_frame):
            raise Exception(
                "The centroid cannot be remove beucase it belongs to a"
                "unique identity."
                "Only centroids of duplicated identities can be deleted"
            )

        if len(self.final_centroids) == 1:
            raise Exception(
                "The centroid cannot be removed because if the last "
                "centroid of the blob"
            )

        if apply_resolution_reduction:
            centroid = (
                centroid[0] * self.resolution_reduction,
                centroid[1] * self.resolution_reduction,
            )

        if (
            self.user_generated_centroids is None
        ):  # removing a centroid from a crossing
            self._user_generated_centroids = [(None, None)] * len(
                self.final_centroids
            )
        if self.user_generated_identities is None:
            self._user_generated_identities = [None] * len(
                self.final_identities
            )

        try:
            if centroid in self.user_generated_centroids:
                centroid_index = self.user_generated_centroids.index(centroid)
            elif centroid in self.assigned_centroids:
                if self.assigned_centroids.count(centroid) > 1:
                    centroid_index = self.assigned_identities.index(identity)
                else:
                    centroid_index = self.assigned_centroids.index(centroid)
            else:
                raise Exception(
                    "There is no centroid with the values of centroid"
                )
        except ValueError:
            raise Exception("There is no centroid with the values of centroid")

        self._user_generated_centroids[centroid_index] = (-1, -1)
        self._user_generated_identities[centroid_index] = -1
        video.is_centroid_updated = True

    def add_centroid(
        self, video, centroid, identity, apply_resolution_reduction=True
    ):
        """[Validation] Adds a centroid with a given identity to the blob.

        This method is used in the validation GUI. It is useful to add
        centroids for crossing blobs that are missing some centroids, or to
        individual blobs that should have been classified as crossings and
        are also missing some centroids.


        Parameters
        ----------
        video : idtrackerai.video.Video
            Instance of Video object
        centroid : tuple
            Centroid to be added to the blob
        identity : int
            Identity of the centroid to be added
        apply_resolution_reduction : bool, optional
            Whether to consider the resolution reduction factor when
            adding the centroid, by default True. Note that the video is showed
            as full resolution in the validation GUI, but all centroids of
            the blobs consider the resolution reduction factor.

        Raises
        ------
        Exception
            If the centroid is not a tuple of length 2
        Exception
            If the identity is not an integer between 1 and number of animals
        Exception
            If there is already another centroid with the same identity
        """
        logger.info("Calling add_centroid")
        if not (isinstance(centroid, tuple) and len(centroid) == 2):
            raise Exception("The centroid must be a tuple of length 2")
        if not (
            isinstance(identity, int)
            and identity > 0
            and identity <= self.number_of_animals
        ):
            raise Exception(
                "The identity must be an integer between 1 and the number of "
                "animals in the video"
            )
        if identity in self.final_identities:
            raise Exception(
                "The identity of the centroid to be created already exist in "
                "this blob"
            )

        if apply_resolution_reduction:
            centroid = (
                centroid[0] * self.resolution_reduction,
                centroid[1] * self.resolution_reduction,
            )

        if self.user_generated_centroids is None:
            self._user_generated_centroids = [(None, None)] * len(
                self.final_centroids
            )
        if self.user_generated_identities is None:
            self._user_generated_identities = [None] * len(
                self.final_identities
            )

        self._user_generated_centroids.append(centroid)
        self._user_generated_identities.append(identity)

        video.is_centroid_updated = True

    def update_identity(self, old_identity, new_identity, centroid):
        """[Validation] Updates the identity of the blob.

        This method is used during the validation GUI.
        It populates the private attributes `_user_generated_identities`
        and `_user_generated_centroids`.

        Parameters
        ----------
        new_identity : int
            new value for the identity of the blob
        old_identity : int
            old value of the identity of the blob. It must be specified when the
            blob has multiple identities already assigned.
        centroid : tuple
            centroid which identity must be updated.
        """
        if not (
            isinstance(new_identity, int)
            and new_identity >= 0
            and new_identity <= self.number_of_animals
        ):
            raise Exception(
                "The new identity must be an integer between 0 and the number "
                "of animals in the video. Blobs with 0 identity will be ommited "
                "for the generation of the trajectories"
            )

        # We prepare to also modify the centroid
        if self.user_generated_centroids is None:
            self._user_generated_centroids = [(None, None)] * len(
                self.final_centroids
            )
        if self.user_generated_identities is None:
            self._user_generated_identities = [None] * len(
                self.final_identities
            )

        # TODO: review this piece of code, not sure I need the try/except
        try:
            if centroid in self.user_generated_centroids:
                centroid_index = self.user_generated_centroids.index(centroid)
            elif centroid in self.assigned_centroids:
                if self.assigned_centroids.count(centroid) > 1:
                    centroid_index = self.assigned_identities.index(
                        old_identity
                    )
                else:
                    centroid_index = self.assigned_centroids.index(centroid)
            else:
                raise Exception(
                    "There is no centroid with the values of centroid"
                )
        except ValueError:
            raise Exception("There is no centroid with the values of centroid")

        self._user_generated_identities[centroid_index] = new_identity
        self._user_generated_centroids[centroid_index] = (
            centroid[0],
            centroid[1],
        )

    def propagate_identity(self, old_identity, new_identity, centroid):
        """[Validation] Propagates the new identity to next and previous blobs.

        This method called in the validation GUI when the used updates the
        identity of a given blob.

        Parameters
        ----------
        old_identity : int
            Previous identity of the blob
        new_identity : int
            New ientity of the blob
        centroid : tuple
            [description]
        """
        count_past_corrections = 1  # to take into account the modification
        # already done in the current frame
        count_future_corrections = 0

        current = self
        current_centroid = np.asarray(centroid)

        while (
            len(current.next) == 1
            and current.next[0].fragment_identifier == self.fragment_identifier
        ):
            # There is only one next blob and we are in the same fragment
            if len(current.next[0].final_centroids) > 1:
                # The next blob has multiple centroids, i.e. a crossing.

                # Find the index of the centroid that correspond to the
                # identity that we want to modify
                index_same_identities = np.where(
                    np.asarray(current.next[0].final_identities)
                    == old_identity
                )[0]
                if index_same_identities.size == 1:
                    # there is only one centroid with the old identity
                    next_centroid = current.next[0].final_centroids[
                        index_same_identities[0]
                    ]
                else:
                    # there are several centroids in the blob with the same
                    # identity
                    next_centroids = np.asarray(
                        current.next[0].final_centroids
                    )
                    index_centroid = np.argmin(
                        np.sqrt(
                            np.sum((current_centroid - next_centroids) ** 2)
                        )
                    )
                    next_centroid = current.next[0].final_centroids[
                        index_centroid
                    ]
            else:
                # The next blob has a single centroid, i.e. an individual.
                next_centroid = current.next[0].final_centroids[0]
            current.next[0].update_identity(
                old_identity, new_identity, next_centroid
            )
            current = current.next[0]
            current_centroid = np.asarray(next_centroid)
            count_future_corrections += 1

        current = self

        while (
            len(current.previous) == 1
            and current.previous[0].fragment_identifier
            == self.fragment_identifier
        ):
            # There is only one previous blob and we are in the same fragment
            if len(current.previous[0].final_centroids) > 1:
                # There are multiple centroids, i.e. a crossing.
                index_same_identities = np.where(
                    np.asarray(current.previous[0].final_identities)
                    == old_identity
                )[0]
                if index_same_identities.size == 1:
                    # there is only one centroid with the old identity
                    previous_centroid = current.previous[0].final_centroids[
                        index_same_identities[0]
                    ]
                else:
                    # there are several centroids in the blob with the same
                    # identity
                    previous_centroids = np.asarray(
                        current.previous[0].final_centroids
                    )
                    index_centroid = np.argmin(
                        np.sqrt(
                            np.sum(
                                (current_centroid - previous_centroids) ** 2
                            )
                        )
                    )
                    previous_centroid = current.previous[0].final_centroids[
                        index_centroid
                    ]
            else:
                # There is a single centroid, i.e. and individual blob
                previous_centroid = current.previous[0].final_centroids[0]
            current.previous[0].update_identity(
                old_identity, new_identity, previous_centroid
            )
            current = current.previous[0]
            current_centroid = np.asarray(previous_centroid)
            count_past_corrections += 1

    @property
    def summary(self):
        """[Validation] Returns a summary string for some blob attributes.

        Returns
        -------
        str
            Summary description of the blob
        """
        blob_name = f"{self}\n"
        used_for_training = f"used for training: {self.used_for_training}\n"
        fragment_id = f"fragment id: {self.fragment_identifier}\n"
        previous_blob = f"previous blob(s): {self.previous}\n"
        next_blob = f"next blob(s): {self.next}\n"
        sure_individual_crossing = (
            f"sure individual-crossing: "
            f"{self.is_a_sure_individual()}-"
            f"{self.is_a_sure_crossing()}\n"
        )
        individual_crossing = (
            f"individual-crossing: "
            f"{self.is_an_individual}-"
            f"{self.is_a_crossing}\n"
        )
        was_a_crossing = f"was_a_crossing: {self.was_a_crossing}\n"
        id = f"identity: {self.identity}\n"
        id_correcting_jumps = (
            f"identity correcting jumps "
            f"{self.identity_corrected_solving_jumps}\n"
        )
        correcting_gaps_id = (
            f"id correcting gaps: {self.identities_corrected_closing_gaps}\n"
        )
        assigned_identities = (
            f"assigned identities: {self.assigned_identities}\n"
        )
        assigned_centroids = f"assigned centroids: {self.assigned_centroids}\n"
        user_identities = (
            f"user identities: {self.user_generated_identities}\n"
        )
        user_centroids = f"user centroids: {self.user_generated_centroids}\n"
        final_identities = f"final identities: {self.final_identities}\n"
        final_centroids = f"final centroids: {self.final_centroids}\n"

        summary_str = (
            blob_name
            + used_for_training
            + fragment_id
            + previous_blob
            + next_blob
            + sure_individual_crossing
            + individual_crossing
            + was_a_crossing
            + id
            + id_correcting_jumps
            + correcting_gaps_id
            + assigned_identities
            + assigned_centroids
            + user_identities
            + user_centroids
            + final_identities
            + final_centroids
        )
        return summary_str

    def draw(
        self, frame, colors_lst=None, selected_id=None, is_selected=False
    ):
        """[Validation] Draw the blob in a given frame of the video.

        Parameters
        ----------
        frame : numpy.array
            Image where the blob should be draw.
        colors_lst : [type], optional
            List of colors used to draw the blobs, by default None
        selected_id : [type], optional
            Identity of the selected blob., by default None
        is_selected : bool, optional
            Flag indicated if the blob has been selected by the user,
            by default False
        """

        contour = self.contour_full_resolution
        bounding_box = self.bounding_box_full_resolution

        for i, (identity, centroid) in enumerate(
            zip(self.final_identities, self.final_centroids_full_resolution)
        ):

            pos = int(round(centroid[0], 0)), int(round(centroid[1], 0))

            if colors_lst:
                color = (
                    colors_lst[identity]
                    if identity is not None
                    else colors_lst[0]
                )
            else:
                color = (0, 0, 255)

            if contour is not None:
                if not is_selected:
                    cv2.polylines(
                        frame, np.array([contour]), True, (0, 255, 0), 1
                    )
                else:
                    cv2.polylines(
                        frame, np.array([contour]), True, (0, 255, 0), 2
                    )

            # cv2.circle(frame, pos, 8, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, pos, 6, color, -1, lineType=cv2.LINE_AA)

            if identity is not None:

                if identity == selected_id:
                    cv2.circle(
                        frame, pos, 10, (0, 0, 255), 2, lineType=cv2.LINE_AA
                    )
                idroot = ""
                if (
                    self.user_generated_identities is not None
                    and identity in self.user_generated_identities
                ) and (
                    self.user_generated_centroids is not None
                    and centroid in self.user_generated_centroids
                ):
                    idroot = "u-"
                elif (
                    self.identities_corrected_closing_gaps is not None
                    and not self.is_an_individual
                ):
                    idroot = "c-"

                idstr = idroot + str(identity)
                text_size = cv2.getTextSize(
                    idstr, cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness=2
                )
                text_width = text_size[0][0]
                str_pos = pos[0] - text_width // 2, pos[1] - 12

                # cv2.putText(frame, idstr, str_pos, cv2.FONT_HERSHEY_SIMPLEX,
                # 1.0, (0, 0, 0), thickness=3,
                #             lineType=cv2.LINE_AA)
                cv2.putText(
                    frame,
                    idstr,
                    str_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )

                if idroot == "c-" and bounding_box is not None:
                    rect_color = (
                        self.rect_color
                        if hasattr(self, "rect_color")
                        else (255, 0, 0)
                    )
                    cv2.rectangle(
                        frame, bounding_box[0], bounding_box[1], rect_color, 2
                    )
            elif bounding_box is not None:
                rect_color = (
                    self.rect_color
                    if hasattr(self, "rect_color")
                    else (255, 0, 0)
                )
                cv2.rectangle(
                    frame, bounding_box[0], bounding_box[1], rect_color, 2
                )

                idstr = "0"
                text_size = cv2.getTextSize(
                    idstr, cv2.FONT_HERSHEY_SIMPLEX, 1.0, thickness=2
                )
                text_width = text_size[0][0]
                str_pos = pos[0] - text_width // 2, pos[1] - 12
                cv2.putText(
                    frame,
                    idstr,
                    str_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 0),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )


def _mask_background_pixels(
    height,
    width,
    bounding_box_image,
    pixels,
    bounding_box_in_frame_coordinates,
):
    """Masks to black the pixels in bounding_box_images that belong to the
    background.

    Note that in this step the blob of pixels defined by `pixels` according
    to the segmentation parameters is dilated to get more information from the
    border of the animal.

    Parameters
    ----------
    height : int
        Frame height
    width : int
        Frame width
    bounding_box_image : ndarray
        Images cropped from the frame by considering the bounding box
        associated to a blob
    pixels : list
        List of pixels associated to a blob
    bounding_box_in_frame_coordinates : list
        [(x, y), (x + bounding_box_width, y + bounding_box_height)]

    Returns
    -------
    ndarray
        Image with black background pixels
    """
    pxs = np.array(np.unravel_index(pixels, (height, width))).T
    pxs = np.array(
        [
            pxs[:, 0] - bounding_box_in_frame_coordinates[0][1],
            pxs[:, 1] - bounding_box_in_frame_coordinates[0][0],
        ]
    )
    temp_image = np.zeros_like(bounding_box_image).astype("uint8")
    temp_image[pxs[0, :], pxs[1, :]] = 255

    temp_image = cv2.dilate(
        temp_image, np.ones((3, 3)).astype("uint8"), iterations=1
    )

    rows, columns = np.where(temp_image == 255)
    dilated_pixels = np.array([rows, columns])
    temp_image[
        dilated_pixels[0, :], dilated_pixels[1, :]
    ] = bounding_box_image[dilated_pixels[0, :], dilated_pixels[1, :]]

    return temp_image


def _transform_to_bbox_coordinates(point, bounding_box):
    """Transforms a point in full-frame coordinates to boinding_box
    coordinates.

    Parameters
    ----------
    point : tuple
        point (px, py) in full-frame coordinates
    boundingBox : list
        bounding box coordinates following the convention
        [(x, y), (x + bounding_box_width, y + bounding_box_height)]

    Returns
    -------
    tuple
        point (px', py') in bouning_box_coordinates

    """
    return tuple(
        np.asarray(point)
        - np.asarray([bounding_box[0][0], bounding_box[0][1]])
    )
