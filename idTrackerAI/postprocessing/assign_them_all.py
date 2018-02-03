from __future__ import absolute_import, print_function, division
import os
import sys
sys.path.append('../utils')
sys.path.append('../')
sys.path.append('../preprocessing')

from tqdm import tqdm
import collections
import numpy as np
import cv2
from pprint import pprint
from scipy.spatial.distance import cdist

from video_utils import blobExtractor
from list_of_fragments import ListOfFragments
from blob import Blob
from list_of_blobs import ListOfBlobs

''' erosion '''
def compute_erosion_disk(video, blobs_in_video):
    return np.ceil(np.nanmedian([compute_min_frame_distance_transform(video, blobs_in_frame)
                    for blobs_in_frame in blobs_in_video if len(blobs_in_frame) > 0])).astype(np.int)

def compute_min_frame_distance_transform(video, blobs_in_frame):
    max_distance_transform = [compute_max_distance_transform(video, blob)
                    for blob in blobs_in_frame
                    if blob.is_an_individual]
    return np.min(max_distance_transform) if len(max_distance_transform) > 0 else np.nan

def generate_temp_image(video, pixels, bounding_box_in_frame_coordinates):
    pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
    pxs = np.array([pxs[:, 0] - bounding_box_in_frame_coordinates[0][1],
                    pxs[:, 1] - bounding_box_in_frame_coordinates[0][0]])
    temp_image = np.zeros((bounding_box_in_frame_coordinates[1][1] -
                            bounding_box_in_frame_coordinates[0][1],
                            bounding_box_in_frame_coordinates[1][0] -
                            bounding_box_in_frame_coordinates[0][0])).astype('uint8')
    temp_image[pxs[0,:], pxs[1,:]] = 255
    return temp_image

def compute_max_distance_transform(video, blob):
    temp_image = generate_temp_image(video, blob.pixels, blob.bounding_box_in_frame_coordinates)
    return np.max(cv2.distanceTransform(temp_image, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE))

def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image,kernel,iterations = 1)

def get_eroded_blobs(video, blobs_in_frame):
    segmented_frame = np.zeros((video.height, video.width)).astype('uint8')

    for blob in blobs_in_frame:
        pixels = blob.eroded_pixels if hasattr(blob,'eroded_pixels') else blob.pixels
        pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
        segmented_frame[pxs[:,0], pxs[:,1]] = 255

    segmented_eroded_frame = erode(segmented_frame, video.erosion_kernel_size)
    boundingBoxes, _, centroids, _, pixels_all, contours, _ = blobExtractor(segmented_eroded_frame, segmented_eroded_frame, 0, np.inf)
    return [Blob(centroid, contour, None, bounding_box, pixels = pixels)
                for centroid, contour, pixels, bounding_box in zip(centroids, contours, pixels_all, boundingBoxes)]

''' assign them all '''

def set_individual_with_identity_0_as_crossings(list_of_blobs_no_gaps):
    [(setattr(blob, '_is_an_individual', False),
        setattr(blob, '_is_a_crossing', True),
        setattr(blob, '_identity', None),
        setattr(blob, '_identity_corrected_solving_duplications', None))
        for blobs_in_frame in list_of_blobs_no_gaps.blobs_in_video
        for blob in blobs_in_frame
        if blob.assigned_identity == 0]

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def find_the_gap_interval(blobs_in_video, possible_identities, gap_start, list_of_occluded_identities):
    there_are_missing_identities = True
    frame_number = gap_start + 1
    if frame_number < len(blobs_in_video):

        while there_are_missing_identities and frame_number < len(blobs_in_video):
            blobs_in_frame = blobs_in_video[frame_number]
            occluded_identities_in_frame = list_of_occluded_identities[frame_number]
            missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame, occluded_identities_in_frame)
            if len(missing_identities) == 0 or frame_number == len(blobs_in_video)-1:
                there_are_missing_identities = False
            else:
                frame_number += 1
            gap_end = frame_number
    else:
        gap_end = gap_start
    return (gap_start, gap_end)

def get_blob_by_identity(blobs_in_frame, identity):
    for blob in blobs_in_frame:
        if blob.assigned_identity == identity:
            return [blob]
        elif (hasattr(blob, 'identity_corrected_closing_gaps')
            and isinstance(blob.identity_corrected_closing_gaps, list) and
            identity in blob.identity_corrected_closing_gaps):
            return [blob]
        elif (hasattr(blob, 'identity_corrected_closing_gaps')
            and isinstance(blob.identity_corrected_closing_gaps, int) and
            identity == blob.identity_corrected_closing_gaps):
            return [blob]
    return None

def get_candidate_blobs_by_overlapping(blob_to_test, eroded_blobs_in_frame):
    overlapping_blobs = [blob for blob in eroded_blobs_in_frame
                    if blob_to_test.overlaps_with(blob)]

    return overlapping_blobs if len(overlapping_blobs) > 0 else eroded_blobs_in_frame

def get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame, occluded_identities_in_frame):
    identities_in_frame = []
    for blob in blobs_in_frame:
        if isinstance(blob.assigned_identity, int):
            identities_in_frame.append(blob.assigned_identity)
        elif isinstance(blob.assigned_identity, list):
            identities_in_frame.extend(blob.assigned_identity)
    return (set(possible_identities) - set(identities_in_frame)) - set(occluded_identities_in_frame)

def get_candidate_centroid(individual_gap_interval, previous_blob_to_the_gap, next_blob_to_the_gap, identity, border = '', inner_frame_number = None):
    blobs_for_interpolation = [previous_blob_to_the_gap, next_blob_to_the_gap]
    centroids_to_interpolate = [blob_for_interpolation.centroid
                                if blob_for_interpolation.is_an_individual else
                                blob_for_interpolation.interpolated_centroids[blob_for_interpolation.identity_corrected_closing_gaps.index(identity)]
                                for blob_for_interpolation in blobs_for_interpolation]
    centroids_to_interpolate = np.asarray(zip(*centroids_to_interpolate))
    argsort_x = np.argsort(centroids_to_interpolate[0])
    centroids_to_interpolate[0] = centroids_to_interpolate[0][argsort_x]
    centroids_to_interpolate[1] = centroids_to_interpolate[1][argsort_x]
    number_of_points = individual_gap_interval[1] - individual_gap_interval[0] + 1
    x_interp = np.linspace(centroids_to_interpolate[0][0], centroids_to_interpolate[0][1], number_of_points + 1)
    y_interp = np.interp(x_interp, centroids_to_interpolate[0], centroids_to_interpolate[1])
    if border == 'start' and np.all(argsort_x == np.asarray([0, 1])):
        return zip(x_interp, y_interp)[1]
    elif border == 'start' and np.all(argsort_x == np.asarray([1, 0])):
            return zip(x_interp, y_interp)[-2]
    elif border == 'end' and np.all(argsort_x == np.asarray([0, 1])):
            return zip(x_interp, y_interp)[-2]
    elif border == 'end' and np.all(argsort_x == np.asarray([1, 0])):
        return zip(x_interp, y_interp)[1]
    else:
        raise ValueError('border must be start or end: %s was given instead' %border)

def find_the_individual_gap_interval(blobs_in_video, possible_identities, identity, a_frame_in_the_gap, list_of_occluded_identities):
    # find gap start
    identity_is_missing = True
    gap_start = a_frame_in_the_gap
    frame_number = gap_start

    while identity_is_missing and frame_number > 0 and frame_number < len(blobs_in_video) :
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[frame_number]
        missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame, occluded_identities_in_frame)
        if identity not in missing_identities:
            gap_start = frame_number + 1
            identity_is_missing = False
        else:
            frame_number -= 1

    # find gap end
    identity_is_missing = True
    frame_number = a_frame_in_the_gap
    gap_end = a_frame_in_the_gap

    while identity_is_missing and frame_number > 0 and frame_number < len(blobs_in_video):
        blobs_in_frame = blobs_in_video[frame_number]
        occluded_identities_in_frame = list_of_occluded_identities[frame_number]
        missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame, occluded_identities_in_frame)
        if identity not in missing_identities:
            gap_end = frame_number
            identity_is_missing = False
        else:
            gap_end += 1
            frame_number = gap_end

    return (gap_start, gap_end)

def get_previous_and_next_blob_wrt_gap(blobs_in_video, possible_identities, identity, frame_number, list_of_occluded_identities):
    individual_gap_interval = find_the_individual_gap_interval(blobs_in_video,
                                        possible_identities,
                                        identity,
                                        frame_number,
                                        list_of_occluded_identities)
    if individual_gap_interval[0] != 0:
        previous_blob_to_the_gap = get_blob_by_identity(blobs_in_video[individual_gap_interval[0]-1], identity)
    else:
        previous_blob_to_the_gap = None
    if individual_gap_interval[1] != len(blobs_in_video):
        next_blob_to_the_gap = get_blob_by_identity(blobs_in_video[individual_gap_interval[1]], identity)
    else:
        next_blob_to_the_gap = None
    if previous_blob_to_the_gap is not None and len(previous_blob_to_the_gap) == 1 and previous_blob_to_the_gap[0] is not None:
        previous_blob_to_the_gap = previous_blob_to_the_gap[0]
    elif previous_blob_to_the_gap is None:
        return individual_gap_interval, None, None
    if  next_blob_to_the_gap is not None and len(next_blob_to_the_gap) == 1 and next_blob_to_the_gap[0] is not None:
        next_blob_to_the_gap = next_blob_to_the_gap[0]
    elif next_blob_to_the_gap is None:
        return individual_gap_interval, None, None

    return individual_gap_interval, previous_blob_to_the_gap, next_blob_to_the_gap

def get_closest_contour_point_to(contour, candidate_centroid):
    return tuple(contour[np.argmin(cdist([candidate_centroid], np.squeeze(contour)))][0])

def get_internal_point_to_blob(pixels):
    return tuple(contour[np.argmin(cdist([candidate_centroid], np.squeeze(contour)))][0])

def plot_blob_and_centroid(video, blob, centroid):
    segmented_frame = np.zeros((video.height, video.width)).astype('uint8')

    pixels = blob.eroded_pixels if hasattr(blob,'eroded_pixels') else blob.pixels
    pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
    segmented_frame[pxs[:,0], pxs[:,1]] = 255

    segmented_eroded_frame = erode(segmented_frame, video.erosion_kernel_size)
    segmented_eroded_frame = cv2.cvtColor(segmented_eroded_frame, cv2.COLOR_GRAY2RGB)
    segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_GRAY2RGB)
    cv2.circle(segmented_eroded_frame, centroid, 1, (0, 255, 0), -1)
    cv2.circle(segmented_frame, centroid, 1, (0, 255, 0), -1)
    cv2.imshow("segmented_frame %i " %blob.frame_number, segmented_frame)
    cv2.imshow("segmented_eroded_frame %i " %blob.frame_number, segmented_eroded_frame)
    cv2.waitKey()

def get_nearest_eroded_blob_to_candidate_centroid(eroded_blobs, candidate_centroid, identity, inner_frame_number):
    eroded_blob_index = np.argmin([blob.distance_from_countour_to(candidate_centroid) for blob in eroded_blobs])
    return eroded_blobs[eroded_blob_index]

def nearest_candidate_blob_is_near_enough(video, candidate_blob, candidate_centroid, blob_in_border_frame):
    points = [candidate_centroid, blob_in_border_frame.centroid]
    distances = np.asarray([np.sqrt(candidate_blob.squared_distance_to(point)) for point in points])
    return np.any(distances < video.velocity_threshold)

def eroded_blob_overlaps_with_blob_in_border_frame(eroded_blob, blob_in_border_frame):
    return eroded_blob.overlaps_with(blob_in_border_frame)

def centroid_is_inside_of_any_eroded_blob(candidate_eroded_blobs, candidate_centroid):
    candidate_centroid = tuple([int(centroid_coordinate) for centroid_coordinate in candidate_centroid])
    return [blob for blob in candidate_eroded_blobs
            if cv2.pointPolygonTest(blob.contour, candidate_centroid, False) >= 0]

def evaluate_candidate_blobs_and_centroid(video, candidate_eroded_blobs, candidate_centroid, blob_in_border_frame, blobs_in_frame = None, inner_frame_number = None, identity = None):
    blob_containing_candidate_centroid = centroid_is_inside_of_any_eroded_blob(candidate_eroded_blobs, candidate_centroid)
    if blob_containing_candidate_centroid:
        return blob_containing_candidate_centroid[0], candidate_centroid
    else:
        nearest_blob = get_nearest_eroded_blob_to_candidate_centroid(candidate_eroded_blobs, candidate_centroid, identity, inner_frame_number)
        new_centroid = get_closest_contour_point_to(nearest_blob.contour, candidate_centroid)
        if nearest_candidate_blob_is_near_enough(video, nearest_blob, candidate_centroid, blob_in_border_frame) or \
            eroded_blob_overlaps_with_blob_in_border_frame(nearest_blob, blob_in_border_frame):
            return nearest_blob, new_centroid
        else:
            return None, None

def get_blobs_with_centroid_in_original_blob(original_blob, candidate_tuples_to_close_gap):
    candidate_tuples_with_centroids_in_original_blob = [candidate_tuple for candidate_tuple in candidate_tuples_to_close_gap
                                                        if cv2.pointPolygonTest(original_blob.contour,
                                                        tuple([int(c) for c in candidate_tuple[1]]), False) >= 0]
    return candidate_tuples_with_centroids_in_original_blob

def assign_identity_to_new_blobs(video, fragments, blobs_in_video, possible_identities, original_blobs, candidate_tuples_to_close_gap, list_of_occluded_identities):
    new_original_blobs = []

    for i, original_blob in enumerate(original_blobs):
        candidate_tuples_with_centroids_in_original_blob = get_blobs_with_centroid_in_original_blob(original_blob, candidate_tuples_to_close_gap)
        if len(candidate_tuples_with_centroids_in_original_blob) == 1: # the gap is a single individual blob
            identity = candidate_tuples_with_centroids_in_original_blob[0][2]
            centroid = candidate_tuples_with_centroids_in_original_blob[0][1]
            if original_blob.assigned_identity == 0 and original_blob.is_an_individual:
                original_blob._identity_corrected_closing_gaps = identity
                [setattr(blob,'_identity_corrected_closing_gaps', identity)
                    for blobs_in_frame in blobs_in_video for blob in blobs_in_frame
                    if blob.fragment_identifier == original_blob.fragment_identifier]
            elif original_blob.is_an_individual:
                list_of_occluded_identities[original_blob.frame_number].append(identity)
            elif original_blob.is_a_crossing:
                if original_blob.assigned_identity is not None:
                    if isinstance(original_blob.assigned_identity, list):
                        identity = original_blob.assigned_identity + [identity]
                        centroid = original_blob.interpolated_centroids + [centroid]
                    else:
                        identity = [original_blob.assigned_identity, identity]
                        centroid = [original_blob.centroid, centroid]
                else:
                    identity = [identity]
                    centroid = [centroid]
                frame_number = original_blob.frame_number
                new_blob = candidate_tuples_with_centroids_in_original_blob[0][0]
                new_blob.frame_number = frame_number
                new_blob._identity_corrected_closing_gaps = identity
                new_blob.interpolated_centroids = centroid
                original_blob = new_blob

            new_original_blobs.append(original_blob)
        elif len(candidate_tuples_with_centroids_in_original_blob) > 1 and original_blob.is_a_crossing:
            candidate_eroded_blobs = zip(*candidate_tuples_with_centroids_in_original_blob)[0]
            candidate_eroded_blobs_centroids = zip(*candidate_tuples_with_centroids_in_original_blob)[1]
            candidate_eroded_blobs_identities = zip(*candidate_tuples_with_centroids_in_original_blob)[2]
            if len(set(candidate_eroded_blobs)) == 1: # crossing not split
                original_blob.interpolated_centroids = [candidate_eroded_blob_centroid
                                                                    for candidate_eroded_blob_centroid in candidate_eroded_blobs_centroids]
                original_blob._identity_corrected_closing_gaps = [candidate_eroded_blob_identity
                                                                    for candidate_eroded_blob_identity in candidate_eroded_blobs_identities]
                original_blob.eroded_pixels = candidate_eroded_blobs[0].pixels
                new_original_blobs.append(original_blob)

            elif len(set(candidate_eroded_blobs)) > 1: # crossing split
                list_of_new_blobs_in_next_frames = []
                count_eroded_blobs = {eroded_blob: candidate_eroded_blobs.count(eroded_blob)
                                        for eroded_blob in candidate_eroded_blobs}
                for j, (eroded_blob, centroid, identity) in enumerate(candidate_tuples_with_centroids_in_original_blob):
                    if count_eroded_blobs[eroded_blob] == 1: # split blob, single individual
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.centroid = centroid
                        eroded_blob._identity_corrected_closing_gaps = identity
                        eroded_blob._is_an_individual = True
                        eroded_blob._was_a_crossing = True
                        new_original_blobs.append(eroded_blob)
                    elif count_eroded_blobs[eroded_blob] > 1:
                        if not hasattr(eroded_blob, 'interpolated_centroids'):
                            eroded_blob.interpolated_centroids = []
                            eroded_blob._identity_corrected_closing_gaps = []
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.interpolated_centroids.append(centroid)
                        eroded_blob._identity_corrected_closing_gaps.append(identity)
                        eroded_blob._is_a_crossing = True
                        new_original_blobs.append(eroded_blob)
        else:
            new_original_blobs.append(original_blob)

    new_original_blobs = list(set(new_original_blobs))
    blobs_in_video[original_blob.frame_number] = new_original_blobs
    return blobs_in_video, list_of_occluded_identities

def get_forward_backward_list_of_frames(gap_interval):
    """input:
    gap_interval: array of tuple [start_frame_number, end_frame_number]
    output:
    [f1, fn, f2, fn-1, ...] for f1 = start_frame_number and
                                fn = end_frame_number"""
    gap_range = range(gap_interval[0],gap_interval[1])
    gap_length = len(gap_range)
    return np.insert(gap_range[::-1], np.arange(gap_length), gap_range)[:gap_length]

def interpolate_trajectories_during_gaps(video, list_of_blobs, list_of_fragments, list_of_occluded_identities, possible_identities, erosion_counter):
    blobs_in_video = list_of_blobs.blobs_in_video
    for frame_number, (blobs_in_frame, occluded_identities_in_frame) in enumerate(tqdm(zip(blobs_in_video, list_of_occluded_identities), desc = "closing gaps")):

        if frame_number != 0:
            missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame, occluded_identities_in_frame)
            if len(missing_identities) > 0 and len(blobs_in_frame) >= 1:
                gap_interval = find_the_gap_interval(blobs_in_video, possible_identities, frame_number, list_of_occluded_identities)
                forward_backward_list_of_frames = get_forward_backward_list_of_frames(gap_interval)

                for index, inner_frame_number in enumerate(forward_backward_list_of_frames):
                    inner_occluded_identities_in_frame = list_of_occluded_identities[inner_frame_number]
                    inner_blobs_in_frame = blobs_in_video[inner_frame_number]
                    if erosion_counter != 0:
                        eroded_blobs_in_frame = get_eroded_blobs(video, inner_blobs_in_frame) #list of eroded blobs!
                        if len(eroded_blobs_in_frame) == 0:
                            eroded_blobs_in_frame = inner_blobs_in_frame
                    else:
                        eroded_blobs_in_frame = inner_blobs_in_frame
                    candidate_tuples_to_close_gap = []
                    inner_missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities,
                                                                                        inner_blobs_in_frame,
                                                                                        inner_occluded_identities_in_frame)

                    for identity in inner_missing_identities:
                        individual_gap_interval,\
                        previous_blob_to_the_gap,\
                        next_blob_to_the_gap = get_previous_and_next_blob_wrt_gap(blobs_in_video,
                                                                                    possible_identities,
                                                                                    identity,
                                                                                    inner_frame_number,
                                                                                    list_of_occluded_identities)
                        if previous_blob_to_the_gap is not None and next_blob_to_the_gap is not None:
                            border = 'start' if index % 2 == 0 else 'end'
                            candidate_centroid = get_candidate_centroid(individual_gap_interval,
                                                                        previous_blob_to_the_gap,
                                                                        next_blob_to_the_gap,
                                                                        identity,
                                                                        border = border,
                                                                        inner_frame_number = inner_frame_number)
                            if border == 'start':
                                blob_in_border_frame = previous_blob_to_the_gap
                            elif border == 'end':
                                blob_in_border_frame = next_blob_to_the_gap
                            candidate_eroded_blobs_by_overlapping = get_candidate_blobs_by_overlapping(blob_in_border_frame,
                                                                                                eroded_blobs_in_frame)
                            candidate_eroded_blobs_by_inclusion_of_centroid = centroid_is_inside_of_any_eroded_blob(eroded_blobs_in_frame, candidate_centroid)
                            candidate_eroded_blobs = candidate_eroded_blobs_by_overlapping + candidate_eroded_blobs_by_inclusion_of_centroid
                            candidate_blob_to_close_gap, centroid = evaluate_candidate_blobs_and_centroid(video,
                                                                                                            candidate_eroded_blobs,
                                                                                                            candidate_centroid,
                                                                                                            blob_in_border_frame,
                                                                                                            blobs_in_frame = inner_blobs_in_frame,
                                                                                                            inner_frame_number = inner_frame_number, identity = identity)
                            if candidate_blob_to_close_gap is not None:
                                candidate_tuples_to_close_gap.append((candidate_blob_to_close_gap, centroid, identity))
                            else:
                                list_of_occluded_identities[inner_frame_number].append(identity)
                        else: # this manages the case in which identities are missing in the first frame or disappear without appearing anymore,
                            # and evntual occlusions (an identified blob does not appear in the previous and/or the next frame)
                            list_of_occluded_identities[inner_frame_number].append(identity)

                    blobs_in_video, list_of_occluded_identities = assign_identity_to_new_blobs(video, list_of_fragments.fragments,
                                                                blobs_in_video, possible_identities,
                                                                inner_blobs_in_frame, candidate_tuples_to_close_gap,
                                                                list_of_occluded_identities)
    return blobs_in_video, list_of_occluded_identities

def get_number_of_non_split_crossing(blobs_in_video):
    return len([blob for blobs_in_frame in blobs_in_video
                for blob in blobs_in_frame
                if blob.is_a_crossing])

def reset_blobs_in_video_before_erosion_iteration(blobs_in_video):
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_crossing:
                blob._identity = None
            elif blob.is_an_individual and isinstance(blob.identity_corrected_closing_gaps, list):
                blob._identity_corrected_closing_gaps = None

def closing_gap_stopping_criteria(blobs_in_video, previous_number_of_non_split_crossings_blobs):
    current_number_of_non_split_crossings = get_number_of_non_split_crossing(blobs_in_video)
    return current_number_of_non_split_crossings, previous_number_of_non_split_crossings_blobs > current_number_of_non_split_crossings

def clean_individual_blob_before_saving(blobs_in_video):
    """Clean inidividual blobs whose identity is a list (it cannot be, hence an
    occluded identity has been assigned to an individual blob).
    """
    for blobs_in_frame in blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_an_individual and isinstance(blob.assigned_identity, list):
                if blob.identity_corrected_solving_duplication is not None:
                    blob._identity_corrected_closing_gaps = blob.identity_corrected_solving_duplication
                elif blob.identity is not None and blob.identity != 0:
                    blob._identity_corrected_closing_gaps = blob.identity
                elif blob.identity == 0:
                    blob._identity = 0

    return blobs_in_video

def close_trajectories_gaps(video, list_of_blobs, list_of_fragments):
    set_individual_with_identity_0_as_crossings(list_of_blobs)
    continue_erosion_protocol = True
    previous_number_of_non_split_crossings_blobs = sum([fragment.number_of_images
                                                    for fragment in list_of_fragments.fragments
                                                    if fragment.is_a_crossing])
    if not hasattr(video, '_erosion_kernel_size'):
        video._erosion_kernel_size = compute_erosion_disk(video, list_of_blobs.blobs_in_video)
        video.save()
    possible_identities = range(1, video.number_of_animals + 1)
    erosion_counter = 0
    list_of_occluded_identities = [[] for i in range(len(list_of_blobs.blobs_in_video))]

    while continue_erosion_protocol or erosion_counter == 1:
        reset_blobs_in_video_before_erosion_iteration(list_of_blobs.blobs_in_video)
        list_of_blobs.blobs_in_video, \
        list_of_occluded_identities = interpolate_trajectories_during_gaps(video,
                                                                            list_of_blobs,
                                                                            list_of_fragments,
                                                                            list_of_occluded_identities,
                                                                            possible_identities,
                                                                            erosion_counter)
        current_number_of_non_split_crossings, continue_erosion_protocol = closing_gap_stopping_criteria(list_of_blobs.blobs_in_video,
                                                                                previous_number_of_non_split_crossings_blobs)
        previous_number_of_non_split_crossings_blobs = current_number_of_non_split_crossings
        erosion_counter += 1

    list_of_blobs.blobs_in_video = clean_individual_blob_before_saving(list_of_blobs.blobs_in_video)
    return list_of_blobs

if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_n/video_object.npy').item()
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    if len(list_of_blobs.blobs_in_video[-1]) == 0:
        list_of_blobs.blobs_in_video = list_of_blobs.blobs_in_video[:-1]
    list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments, video.fragment_identifier_to_index)
    list_of_blobs = close_trajectories_gaps(video, list_of_blobs)

    video.blobs_no_gaps_path = os.path.join(os.path.split(video.blobs_path)[0], 'blobs_collection_no_gaps.npy')
    video.save()
    list_of_blobs.save(video, path_to_save = video.blobs_no_gaps_path, number_of_chunks = video.number_of_frames)
