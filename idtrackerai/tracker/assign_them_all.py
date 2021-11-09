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

import cv2
import numpy as np
from confapp import conf
from scipy.spatial.distance import cdist
from tqdm import tqdm

from idtrackerai.tracker.compute_velocity_model import (
    compute_model_velocity,
)
from idtrackerai.tracker.erosion import (
    compute_erosion_disk,
    get_eroded_blobs,
)

logger = logging.getLogger("__main__.assign_them_all")

""" assign them all """


def set_individual_with_identity_0_as_crossings(list_of_blobs_no_gaps):
    for blobs_in_frame in list_of_blobs_no_gaps.blobs_in_video:
        for blob in blobs_in_frame:
            if (
                blob.is_an_individual
                and len(blob.assigned_identities) == 1
                and blob.assigned_identities[0] == 0
            ):
                blob._is_an_individual = False
                blob._is_a_crossing = True
                blob._identity = None
                blob._identity_corrected_solving_jumps = None


def find_the_gap_interval(
    blobs_in_video, possible_identities, gap_start, list_of_occluded_identities
):
    # logger.debug('Finding gap interval')
    there_are_missing_identities = True
    frame_number = gap_start + 1
    if frame_number < len(blobs_in_video):

        while (
            there_are_missing_identities
            and frame_number > 0
            and frame_number < len(blobs_in_video)
        ):
            blobs_in_frame = blobs_in_video[frame_number]
            occluded_identities_in_frame = list_of_occluded_identities[
                frame_number
            ]
            missing_identities = get_missing_identities_from_blobs_in_frame(
                possible_identities,
                blobs_in_frame,
                occluded_identities_in_frame,
            )
            if (
                len(missing_identities) == 0
                or frame_number == len(blobs_in_video) - 1
            ):
                there_are_missing_identities = False
            else:
                frame_number += 1
            gap_end = frame_number
    else:
        gap_end = gap_start
    # logger.debug('Finished finding gap interval')
    return (gap_start, gap_end)


def get_blob_by_identity(blobs_in_frame, identity):
    for blob in blobs_in_frame:
        if identity in blob.final_identities:
            return [blob]
    return None


def get_candidate_blobs_by_overlapping(blob_to_test, eroded_blobs_in_frame):
    # logger.debug('Getting candidate blobs by overlapping')
    overlapping_blobs = [
        blob
        for blob in eroded_blobs_in_frame
        if blob_to_test.overlaps_with(blob)
    ]
    # logger.debug('Finished getting candidate blobs by overlapping')
    return (
        overlapping_blobs
        if len(overlapping_blobs) > 0
        else eroded_blobs_in_frame
    )


def get_missing_identities_from_blobs_in_frame(
    possible_identities, blobs_in_frame, occluded_identities_in_frame
):
    identities_in_frame = []
    for blob in blobs_in_frame:
        identities_in_frame.extend(blob.final_identities)
    return (set(possible_identities) - set(identities_in_frame)) - set(
        occluded_identities_in_frame
    )


def get_candidate_centroid(
    individual_gap_interval,
    previous_blob_to_the_gap,
    next_blob_to_the_gap,
    identity,
    border="",
    inner_frame_number=None,
):
    # logger.debug('Getting candidate centroids')
    blobs_for_interpolation = [previous_blob_to_the_gap, next_blob_to_the_gap]
    centroids_to_interpolate = [
        blob_for_interpolation.final_centroids[
            blob_for_interpolation.final_identities.index(identity)
        ]
        for blob_for_interpolation in blobs_for_interpolation
    ]
    centroids_to_interpolate = np.asarray(list(zip(*centroids_to_interpolate)))
    argsort_x = np.argsort(centroids_to_interpolate[0])
    centroids_to_interpolate[0] = centroids_to_interpolate[0][argsort_x]
    centroids_to_interpolate[1] = centroids_to_interpolate[1][argsort_x]
    number_of_points = (
        individual_gap_interval[1] - individual_gap_interval[0] + 1
    )
    x_interp = np.linspace(
        centroids_to_interpolate[0][0],
        centroids_to_interpolate[0][1],
        number_of_points + 1,
    )
    y_interp = np.interp(
        x_interp, centroids_to_interpolate[0], centroids_to_interpolate[1]
    )
    if border == "start" and np.all(argsort_x == np.asarray([0, 1])):
        # logger.debug('Finished getting candidate centroids')
        return list(zip(x_interp, y_interp))[1]
    elif border == "start" and np.all(argsort_x == np.asarray([1, 0])):
        # logger.debug('Finished getting candidate centroids')
        return list(zip(x_interp, y_interp))[-2]
    elif border == "end" and np.all(argsort_x == np.asarray([0, 1])):
        # logger.debug('Finished getting candidate centroids')
        return list(zip(x_interp, y_interp))[-2]
    elif border == "end" and np.all(argsort_x == np.asarray([1, 0])):
        # logger.debug('Finished getting candidate centroids')
        return list(zip(x_interp, y_interp))[1]
    else:
        raise ValueError(
            "border must be start or end: %s was given instead" % border
        )


def find_the_individual_gap_interval(
    blobs_in_video,
    possible_identities,
    identity,
    a_frame_in_the_gap,
    list_of_occluded_identities,
):
    # logger.debug('Finding the individual gap interval')
    # find gap start
    identity_is_missing = True
    gap_start = a_frame_in_the_gap
    frame_number = gap_start

    # logger.debug('To the past while loop')
    while (
        identity_is_missing
        and frame_number > 0
        and frame_number < len(blobs_in_video)
    ):
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[
            frame_number
        ]
        missing_identities = get_missing_identities_from_blobs_in_frame(
            possible_identities, blobs_in_frame, occluded_identities_in_frame
        )
        if identity not in missing_identities:
            gap_start = frame_number + 1
            identity_is_missing = False
        else:
            frame_number -= 1

    # find gap end
    identity_is_missing = True
    frame_number = a_frame_in_the_gap
    gap_end = a_frame_in_the_gap

    # logger.debug('To the future while loop')
    while (
        identity_is_missing
        and frame_number > 0
        and frame_number < len(blobs_in_video)
    ):
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[
            frame_number
        ]
        missing_identities = get_missing_identities_from_blobs_in_frame(
            possible_identities, blobs_in_frame, occluded_identities_in_frame
        )
        if identity not in missing_identities:
            gap_end = frame_number
            identity_is_missing = False
        else:
            gap_end += 1
            frame_number = gap_end
    # logger.debug('Finished finding the individual gap interval')
    return (gap_start, gap_end)


def get_previous_and_next_blob_wrt_gap(
    blobs_in_video,
    possible_identities,
    identity,
    frame_number,
    list_of_occluded_identities,
):
    # logger.debug('Finding previons and next blobs to the gap of this identity')
    individual_gap_interval = find_the_individual_gap_interval(
        blobs_in_video,
        possible_identities,
        identity,
        frame_number,
        list_of_occluded_identities,
    )
    if individual_gap_interval[0] != 0:
        previous_blob_to_the_gap = get_blob_by_identity(
            blobs_in_video[individual_gap_interval[0] - 1], identity
        )
    else:
        previous_blob_to_the_gap = None
    if individual_gap_interval[1] != len(blobs_in_video):
        next_blob_to_the_gap = get_blob_by_identity(
            blobs_in_video[individual_gap_interval[1]], identity
        )
    else:
        next_blob_to_the_gap = None
    if (
        previous_blob_to_the_gap is not None
        and len(previous_blob_to_the_gap) == 1
        and previous_blob_to_the_gap[0] is not None
    ):
        previous_blob_to_the_gap = previous_blob_to_the_gap[0]
    else:
        previous_blob_to_the_gap = None
    if (
        next_blob_to_the_gap is not None
        and len(next_blob_to_the_gap) == 1
        and next_blob_to_the_gap[0] is not None
    ):
        next_blob_to_the_gap = next_blob_to_the_gap[0]
    else:
        next_blob_to_the_gap = None
    # logger.debug('Finished finding previons and next blobs to the gap of this identity')
    return (
        individual_gap_interval,
        previous_blob_to_the_gap,
        next_blob_to_the_gap,
    )


def get_closest_contour_point_to(contour, candidate_centroid):
    return tuple(
        contour[np.argmin(cdist([candidate_centroid], np.squeeze(contour)))][0]
    )


def get_nearest_eroded_blob_to_candidate_centroid(
    eroded_blobs, candidate_centroid, identity=None, inner_frame_number=None
):
    eroded_blob_index = np.argmin(
        [
            blob.distance_from_countour_to(candidate_centroid)
            for blob in eroded_blobs
        ]
    )
    return eroded_blobs[eroded_blob_index]


def nearest_candidate_blob_is_near_enough(
    video, candidate_blob, candidate_centroid, blob_in_border_frame
):
    points = [candidate_centroid, blob_in_border_frame.centroid]
    distances = np.asarray(
        [
            np.sqrt(candidate_blob.squared_distance_to(point))
            for point in points
        ]
    )
    return np.any(distances < video.velocity_threshold)


def eroded_blob_overlaps_with_blob_in_border_frame(
    eroded_blob, blob_in_border_frame
):
    return eroded_blob.overlaps_with(blob_in_border_frame)


def centroid_is_inside_of_any_eroded_blob(
    candidate_eroded_blobs, candidate_centroid
):
    # logger.debug('Checking whether the centroids is inside of a blob')
    candidate_centroid = tuple(
        [
            int(centroid_coordinate)
            for centroid_coordinate in candidate_centroid
        ]
    )
    # logger.debug('Finished whether the centroids is inside of a blob')
    return [
        blob
        for blob in candidate_eroded_blobs
        if cv2.pointPolygonTest(blob.contour, candidate_centroid, False) >= 0
    ]


def evaluate_candidate_blobs_and_centroid(
    video,
    candidate_eroded_blobs,
    candidate_centroid,
    blob_in_border_frame,
    blobs_in_frame=None,
    inner_frame_number=None,
    identity=None,
):
    # logger.debug('Evaluating candidate blobs and centroids')
    blob_containing_candidate_centroid = centroid_is_inside_of_any_eroded_blob(
        candidate_eroded_blobs, candidate_centroid
    )
    if blob_containing_candidate_centroid:
        # logger.debug('Finished evaluating candidate blobs and centroids: '
        #              'the candidate centroid is in an eroded blob')
        return blob_containing_candidate_centroid[0], candidate_centroid
    elif len(candidate_eroded_blobs) > 0:
        nearest_blob = get_nearest_eroded_blob_to_candidate_centroid(
            candidate_eroded_blobs,
            candidate_centroid,
            identity,
            inner_frame_number,
        )
        new_centroid = get_closest_contour_point_to(
            nearest_blob.contour, candidate_centroid
        )
        if nearest_candidate_blob_is_near_enough(
            video, nearest_blob, candidate_centroid, blob_in_border_frame
        ) or eroded_blob_overlaps_with_blob_in_border_frame(
            nearest_blob, blob_in_border_frame
        ):
            # logger.debug('Finished evaluating candidate blobs and centroids: '
            #              'the candidate centroid is near to a candidate blob')
            return nearest_blob, new_centroid
        else:
            # logger.debug('Finished evaluating candidate blobs and centroids: '
            #              'the candidate centrois is far from a candidate blob')
            return None, None
    else:
        # logger.debug('Finished evaluating candidate blobs and centroids: '
        #              'there where no candidate blobs')
        return None, None


def get_candidate_tuples_with_centroids_in_original_blob(
    original_blob, candidate_tuples_to_close_gap
):
    candidate_tuples_with_centroids_in_original_blob = [
        candidate_tuple
        for candidate_tuple in candidate_tuples_to_close_gap
        if cv2.pointPolygonTest(
            original_blob.contour,
            tuple([int(c) for c in candidate_tuple[1]]),
            False,
        )
        >= 0
    ]
    return candidate_tuples_with_centroids_in_original_blob


def assign_identity_to_new_blobs(
    video,
    fragments,
    blobs_in_video,
    possible_identities,
    original_inner_blobs_in_frame,
    candidate_tuples_to_close_gap,
    list_of_occluded_identities,
):
    # logger.debug('Assigning identity to new blobs')
    new_original_blobs = []

    for i, original_blob in enumerate(original_inner_blobs_in_frame):
        # logger.debug('Checking original blob')
        candidate_tuples_with_centroids_in_original_blob = (
            get_candidate_tuples_with_centroids_in_original_blob(
                original_blob, candidate_tuples_to_close_gap
            )
        )
        if (
            len(candidate_tuples_with_centroids_in_original_blob) == 1
        ):  # the gap is a single individual blob
            # logger.debug('Only a candidate tuple for this original blob')
            identity = candidate_tuples_with_centroids_in_original_blob[0][2]
            centroid = candidate_tuples_with_centroids_in_original_blob[0][1]
            if (
                original_blob.is_an_individual
                and len(original_blob.final_identities) == 1
                and original_blob.final_identities[0] == 0
            ):
                original_blob._identities_corrected_closing_gaps = [identity]
                [
                    setattr(
                        blob, "_identities_corrected_closing_gaps", [identity]
                    )
                    for blobs_in_frame in blobs_in_video
                    for blob in blobs_in_frame
                    if blob.fragment_identifier
                    == original_blob.fragment_identifier
                ]
            elif original_blob.is_an_individual:
                list_of_occluded_identities[original_blob.frame_number].append(
                    identity
                )
            elif original_blob.is_a_crossing:
                if original_blob.identities_corrected_closing_gaps is not None:
                    identity = (
                        original_blob.identities_corrected_closing_gaps
                        + [identity]
                    )
                    centroid = original_blob.interpolated_centroids + [
                        centroid
                    ]
                else:
                    identity = [identity]
                    centroid = [centroid]
                frame_number = original_blob.frame_number
                new_blob = candidate_tuples_with_centroids_in_original_blob[0][
                    0
                ]
                new_blob.frame_number = frame_number
                new_blob._identities_corrected_closing_gaps = identity
                new_blob.interpolated_centroids = centroid
                original_blob = new_blob

            new_original_blobs.append(original_blob)
        elif (
            len(candidate_tuples_with_centroids_in_original_blob) > 1
            and original_blob.is_a_crossing
        ):  # Note that the original blobs that were unidentified (identity 0)
            # are set to zero before starting the main while loop
            # logger.debug('Many candidate tuples for this original blob, '
            #              'and the original blob is a crossing')
            candidate_eroded_blobs = list(
                zip(*candidate_tuples_with_centroids_in_original_blob)
            )[0]
            candidate_eroded_blobs_centroids = list(
                zip(*candidate_tuples_with_centroids_in_original_blob)
            )[1]
            candidate_eroded_blobs_identities = list(
                zip(*candidate_tuples_with_centroids_in_original_blob)
            )[2]
            if len(set(candidate_eroded_blobs)) == 1:  # crossing not split
                original_blob.interpolated_centroids = [
                    candidate_eroded_blob_centroid
                    for candidate_eroded_blob_centroid in candidate_eroded_blobs_centroids
                ]
                original_blob._identities_corrected_closing_gaps = [
                    candidate_eroded_blob_identity
                    for candidate_eroded_blob_identity in candidate_eroded_blobs_identities
                ]
                original_blob.eroded_pixels = candidate_eroded_blobs[0].pixels
                new_original_blobs.append(original_blob)

            elif len(set(candidate_eroded_blobs)) > 1:  # crossing split
                list_of_new_blobs_in_next_frames = []
                count_eroded_blobs = {
                    eroded_blob: candidate_eroded_blobs.count(eroded_blob)
                    for eroded_blob in candidate_eroded_blobs
                }
                for j, (eroded_blob, centroid, identity) in enumerate(
                    candidate_tuples_with_centroids_in_original_blob
                ):
                    if (
                        count_eroded_blobs[eroded_blob] == 1
                    ):  # split blob, single individual
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.centroid = centroid
                        eroded_blob._identities_corrected_closing_gaps = [
                            identity
                        ]
                        eroded_blob._is_an_individual = True
                        eroded_blob._was_a_crossing = True
                        new_original_blobs.append(eroded_blob)
                    elif count_eroded_blobs[eroded_blob] > 1:
                        if not hasattr(eroded_blob, "interpolated_centroids"):
                            eroded_blob.interpolated_centroids = []
                            eroded_blob._identities_corrected_closing_gaps = []
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.interpolated_centroids.append(centroid)
                        eroded_blob._identities_corrected_closing_gaps.append(
                            identity
                        )
                        eroded_blob._is_a_crossing = True
                        new_original_blobs.append(eroded_blob)

        new_original_blobs.append(original_blob)

    new_original_blobs = list(set(new_original_blobs))
    blobs_in_video[original_blob.frame_number] = new_original_blobs
    # logger.debug('Finished assigning identity to new blobs')
    return blobs_in_video, list_of_occluded_identities


def get_forward_backward_list_of_frames(gap_interval):
    """input:
    gap_interval: array of tuple [start_frame_number, end_frame_number]
    output:
    [f1, fn, f2, fn-1, ...] for f1 = start_frame_number and
                                fn = end_frame_number"""
    # logger.debug('Got forward-backward list of frames')
    gap_range = range(gap_interval[0], gap_interval[1])
    gap_length = len(gap_range)
    return np.insert(gap_range[::-1], np.arange(gap_length), gap_range)[
        :gap_length
    ]


def interpolate_trajectories_during_gaps(
    video,
    list_of_blobs,
    list_of_fragments,
    list_of_occluded_identities,
    possible_identities,
    erosion_counter,
):
    # logger.debug('In interpolate_trajectories_during_gaps')
    blobs_in_video = list_of_blobs.blobs_in_video
    for frame_number, (blobs_in_frame, occluded_identities_in_frame) in tqdm(
        enumerate(zip(blobs_in_video, list_of_occluded_identities)),
        desc="closing gaps",
    ):
        if frame_number != 0:
            # logger.debug('-Main frame number %i' %frame_number)
            # logger.debug('Getting missing identities')
            missing_identities = get_missing_identities_from_blobs_in_frame(
                possible_identities,
                blobs_in_frame,
                occluded_identities_in_frame,
            )
            if len(missing_identities) > 0 and len(blobs_in_frame) >= 1:
                gap_interval = find_the_gap_interval(
                    blobs_in_video,
                    possible_identities,
                    frame_number,
                    list_of_occluded_identities,
                )
                forward_backward_list_of_frames = (
                    get_forward_backward_list_of_frames(gap_interval)
                )
                # logger.debug('--There are missing identities in this main '
                #              'frame: gap interval %s ' %(gap_interval,))
                for index, inner_frame_number in enumerate(
                    forward_backward_list_of_frames
                ):
                    # logger.debug('---Length '
                    #              'forward_backward_list_of_frames '
                    #              '%i' %len(forward_backward_list_of_frames))
                    # logger.debug('---Gap interval: interval '
                    #              '%s ' %(gap_interval,))
                    # logger.debug('---Inner frame number '
                    #              '%i' %inner_frame_number )
                    inner_occluded_identities_in_frame = (
                        list_of_occluded_identities[inner_frame_number]
                    )
                    inner_blobs_in_frame = blobs_in_video[inner_frame_number]
                    if len(inner_blobs_in_frame) != 0:
                        # logger.debug('----There are blobs in the inner frame')
                        if erosion_counter != 0:
                            eroded_blobs_in_frame = get_eroded_blobs(
                                video, inner_blobs_in_frame, inner_frame_number
                            )  # list of eroded blobs!
                            if len(eroded_blobs_in_frame) == 0:
                                eroded_blobs_in_frame = inner_blobs_in_frame
                        else:
                            eroded_blobs_in_frame = inner_blobs_in_frame
                        # logger.debug('Getting missing identities')
                        inner_missing_identities = (
                            get_missing_identities_from_blobs_in_frame(
                                possible_identities,
                                inner_blobs_in_frame,
                                inner_occluded_identities_in_frame,
                            )
                        )
                        candidate_tuples_to_close_gap = []
                        for identity in inner_missing_identities:
                            # logger.debug('-----Solving identity %i' %identity)
                            (
                                individual_gap_interval,
                                previous_blob_to_the_gap,
                                next_blob_to_the_gap,
                            ) = get_previous_and_next_blob_wrt_gap(
                                blobs_in_video,
                                possible_identities,
                                identity,
                                inner_frame_number,
                                list_of_occluded_identities,
                            )
                            # logger.debug('individual_gap_interval: '
                            #              '%s' %(individual_gap_interval,))
                            if (
                                previous_blob_to_the_gap is not None
                                and next_blob_to_the_gap is not None
                            ):
                                # logger.debug('------The previous and next '
                                #              'blobs are not None')
                                border = "start" if index % 2 == 0 else "end"
                                candidate_centroid = get_candidate_centroid(
                                    individual_gap_interval,
                                    previous_blob_to_the_gap,
                                    next_blob_to_the_gap,
                                    identity,
                                    border=border,
                                    inner_frame_number=inner_frame_number,
                                )
                                if border == "start":
                                    blob_in_border_frame = (
                                        previous_blob_to_the_gap
                                    )
                                elif border == "end":
                                    blob_in_border_frame = next_blob_to_the_gap
                                candidate_eroded_blobs_by_overlapping = (
                                    get_candidate_blobs_by_overlapping(
                                        blob_in_border_frame,
                                        eroded_blobs_in_frame,
                                    )
                                )
                                candidate_eroded_blobs_by_inclusion_of_centroid = centroid_is_inside_of_any_eroded_blob(
                                    eroded_blobs_in_frame, candidate_centroid
                                )
                                candidate_eroded_blobs = (
                                    candidate_eroded_blobs_by_overlapping
                                    + candidate_eroded_blobs_by_inclusion_of_centroid
                                )
                                (
                                    candidate_blob_to_close_gap,
                                    centroid,
                                ) = evaluate_candidate_blobs_and_centroid(
                                    video,
                                    candidate_eroded_blobs,
                                    candidate_centroid,
                                    blob_in_border_frame,
                                    blobs_in_frame=inner_blobs_in_frame,
                                    inner_frame_number=inner_frame_number,
                                    identity=identity,
                                )
                                if candidate_blob_to_close_gap is not None:
                                    # logger.debug('------There is a tuple '
                                    #              '(blob, centroid, identity) '
                                    #              'to close the gap in this '
                                    #              'inner frame)')
                                    candidate_tuples_to_close_gap.append(
                                        (
                                            candidate_blob_to_close_gap,
                                            centroid,
                                            identity,
                                        )
                                    )
                                else:
                                    # logger.debug('------There are no candidate '
                                    #              'blobs and/or centroids: it '
                                    #              'must be occluded or it jumped')
                                    list_of_occluded_identities[
                                        inner_frame_number
                                    ].append(identity)
                            # this manages the case in which identities are
                            # missing in the first frame or disappear
                            # without appearing anymore,
                            else:
                                # and evntual occlusions (an identified blob
                                # does not appear in the previous and/or the
                                # next frame)
                                # logger.debug('------There is not next or not '
                                #              'previous blob to this inner gap:'
                                #              ' it must be occluded')
                                # if previous_blob_to_the_gap is None:
                                #     logger.debug('previous_blob_to_the_gap '
                                #                  'is None')
                                # else:
                                #     logger.debug('previous_blob_to_the_gap '
                                #                  'exists')
                                # if next_blob_to_the_gap:
                                #     logger.debug('next_blob_to_the_gap is None')
                                # else:
                                #     logger.debug('next_blob_to_the_gap exists')
                                [
                                    list_of_occluded_identities[i].append(
                                        identity
                                    )
                                    for i in range(
                                        individual_gap_interval[0],
                                        individual_gap_interval[1],
                                    )
                                ]

                        # logger.debug('-----Assinning identities to candidate '
                        #              'tuples (blob, centroid, identity)')
                        (
                            blobs_in_video,
                            list_of_occluded_identities,
                        ) = assign_identity_to_new_blobs(
                            video,
                            list_of_fragments.fragments,
                            blobs_in_video,
                            possible_identities,
                            inner_blobs_in_frame,
                            candidate_tuples_to_close_gap,
                            list_of_occluded_identities,
                        )
                # else:
                # logger.debug('----No blobs in this inner frame')
            # else:
            # logger.debug('--No missing identities in this frame')
        # else:
        # logger.debug('-We do not check the first frame')
    return blobs_in_video, list_of_occluded_identities


def get_number_of_non_split_crossing(blobs_in_video):
    return len(
        [
            blob
            for blobs_in_frame in blobs_in_video
            for blob in blobs_in_frame
            if blob.is_a_crossing
        ]
    )


def reset_blobs_in_video_before_erosion_iteration(blobs_in_video):
    """Resets the identity of crossings and individual with multiple identities
    before starting a loop of interpolation

    Parameters
    ----------
    blobs_in_video : list of lists of `Blob` objects
    """
    # logger.debug('Reseting blobs to start erosion iteration')
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                blob._identity = None
            elif blob.is_an_individual and len(blob.final_identities) > 1:
                blob._identities_corrected_closing_gaps = None


def closing_gap_stopping_criteria(
    blobs_in_video, previous_number_of_non_split_crossings_blobs
):
    current_number_of_non_split_crossings = get_number_of_non_split_crossing(
        blobs_in_video
    )
    return (
        current_number_of_non_split_crossings,
        previous_number_of_non_split_crossings_blobs
        > current_number_of_non_split_crossings,
    )


def clean_individual_blob_before_saving(blobs_in_video):
    """Clean inidividual blobs whose identity is a list (it cannot be, hence an
    occluded identity has been assigned to an individual blob).
    """
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_an_individual and len(blob.final_identities) > 1:
                blob._identities_corrected_closing_gaps = None

    return blobs_in_video


def close_trajectories_gaps(video, list_of_blobs, list_of_fragments):
    """This is the main function to close the gaps where animals have not been
    identified (labelled with identity 0), are crossing with another animals or
    are occluded or not segmented.

    Parameters
    ----------
    video : <Video object>
        Object containing all the parameters of the video.
    list_of_blobs : <ListOfBlobs object>
        Object with the collection of blobs found during segmentation with associated
        methods. See :class:`list_of_blobs.ListOfBlobs`
    list_of_fragments : <ListOfFragments object>
        Collection of individual and crossing fragments with associated methods.
        See :class:`list_of_fragments.ListOfFragments`

    Returns
    -------
    list_of_blobs : <ListOfBlobs object>
        ListOfBlobs object with the updated blobs and identities that close gaps

    See Also
    --------
    :func:`set_individual_with_identity_0_as_crossings`
    :func:`compute_erosion_disk`
    :func:`compute_model_velocity`
    :func:`reset_blobs_in_video_before_erosion_iteration`
    :func:`interpolate_trajectories_during_gaps`
    :func:`closing_gap_stopping_criteria`
    :func:`clean_individual_blob_before_saving`

    """
    logger.debug("********************************")
    logger.debug("Starting close_trajectories_gaps")
    set_individual_with_identity_0_as_crossings(list_of_blobs)
    continue_erosion_protocol = True
    previous_number_of_non_split_crossings_blobs = sum(
        [
            fragment.number_of_images
            for fragment in list_of_fragments.fragments
            if fragment.is_a_crossing
        ]
    )
    if not hasattr(video, "_erosion_kernel_size"):
        video._erosion_kernel_size = compute_erosion_disk(
            video, list_of_blobs.blobs_in_video
        )
        video.save()
    if not hasattr(video, "velocity_threshold"):
        video.velocity_threshold = compute_model_velocity(
            list_of_fragments.fragments,
            video.user_defined_parameters["number_of_animals"],
            percentile=conf.VEL_PERCENTILE,
        )
    possible_identities = range(
        1, video.user_defined_parameters["number_of_animals"] + 1
    )
    erosion_counter = 0
    list_of_occluded_identities = [
        [] for i in range(len(list_of_blobs.blobs_in_video))
    ]

    while continue_erosion_protocol or erosion_counter == 1:
        reset_blobs_in_video_before_erosion_iteration(
            list_of_blobs.blobs_in_video
        )
        (
            list_of_blobs.blobs_in_video,
            list_of_occluded_identities,
        ) = interpolate_trajectories_during_gaps(
            video,
            list_of_blobs,
            list_of_fragments,
            list_of_occluded_identities,
            possible_identities,
            erosion_counter,
        )
        (
            current_number_of_non_split_crossings,
            continue_erosion_protocol,
        ) = closing_gap_stopping_criteria(
            list_of_blobs.blobs_in_video,
            previous_number_of_non_split_crossings_blobs,
        )
        previous_number_of_non_split_crossings_blobs = (
            current_number_of_non_split_crossings
        )
        erosion_counter += 1

    if not video.is_centroid_updated:
        list_of_blobs.blobs_in_video = clean_individual_blob_before_saving(
            list_of_blobs.blobs_in_video
        )
    return list_of_blobs
