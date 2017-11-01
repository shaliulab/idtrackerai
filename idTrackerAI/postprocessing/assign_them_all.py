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

def compute_erosion_disk(video, blobs_in_video):
    return np.ceil(np.median([compute_min_frame_distance_transform(video, blobs_in_frame)
                    for blobs_in_frame in blobs_in_video if len(blobs_in_frame) > 0])).astype(np.int)

def compute_min_frame_distance_transform(video, blobs_in_frame):
    return np.min([compute_max_distance_transform(video, blob)
                    for blob in blobs_in_frame
                    if blob.is_an_individual])

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

def find_the_individual_gap_interval(blobs_in_video, possible_identities, identity, gap_start):
    it_is_still_missing = True
    frame_number = gap_start + 1

    while it_is_still_missing:
        blobs_in_frame = blobs_in_video[frame_number]
        missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame)
        if identity not in missing_identities:
            gap_end = frame_number
            it_is_still_missing = False
        else:
            frame_number += 1

    return (gap_start, gap_end)

def find_the_gap_interval(blobs_in_video, possible_identities, gap_start):
    it_is_still_missing = True
    frame_number = gap_start + 1

    while it_is_still_missing:
        blobs_in_frame = blobs_in_video[frame_number]
        missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame)
        if len(missing_identities) == 0:
            gap_end = frame_number
            it_is_still_missing = False
        else:
            frame_number += 1
    print("gap_interval ", (gap_start, gap_end))
    return (gap_start, gap_end)

def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image,kernel,iterations = 1)

def get_blob_by_identity(blobs_in_frame, identity):
    # print(blobs_in_frame)
    # print(blobs_in_frame[0].frame_number)
    for blob in blobs_in_frame:
        # if hasattr(blob, 'identity'):
        #     print(blob.final_identity, blob.identity, blob.identity_corrected_solving_duplication, blob.user_generated_identity)
        # if hasattr(blob, 'identity_corrected_closing_gaps'): print(blob.identity_corrected_closing_gaps)
        if blob.final_identity == identity:
            # print("pass1")
            return [blob]
        elif (hasattr(blob, 'identity_corrected_closing_gaps')
            and isinstance(blob.identity_corrected_closing_gaps, list) and
            identity in blob.identity_corrected_closing_gaps):
            # print("pass2")
            return [blob]
        elif (hasattr(blob, 'identity_corrected_closing_gaps')
            and isinstance(blob.identity_corrected_closing_gaps, int) and
            identity == blob.identity_corrected_closing_gaps):
            # print("pass3")
            return [blob]
        # else:
            # print("other")

def get_candidate_blobs(blob_to_test, eroded_blobs_in_frame):
    print("getting candidate blobs")
    overlapping_blobs = [blob for blob in eroded_blobs_in_frame
                    if blob_to_test.overlaps_with(blob)]

    return overlapping_blobs if len(overlapping_blobs) > 0 else eroded_blobs_in_frame

def get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame):
    identities_in_frame = []
    for blob in blobs_in_frame:
        if isinstance(blob.final_identity, int):
            identities_in_frame.append(blob.final_identity)
        elif isinstance(blob.final_identity, list):
            identities_in_frame.extend(blob.final_identity)
    return set(possible_identities) - set(identities_in_frame)

def get_candidate_centroid(individual_gap_interval, previous_blob_to_the_gap, next_blob_to_the_gap, identity):
    blobs_for_interpolation = [previous_blob_to_the_gap, next_blob_to_the_gap]
    centroids_to_interpolate = [blob_for_interpolation.centroid
                                if blob_for_interpolation.is_an_individual else
                                blob_for_interpolation.interpolated_centroids[blob_for_interpolation.identity_corrected_closing_gaps.index(identity)]
                                for blob_for_interpolation in blobs_for_interpolation]
    centroids_to_interpolate = zip(*centroids_to_interpolate)
    number_of_points = individual_gap_interval[1] - individual_gap_interval[0] + 1
    x_interp = np.linspace(centroids_to_interpolate[0][0], centroids_to_interpolate[0][1], number_of_points + 2)
    y_interp = np.interp(x_interp, centroids_to_interpolate[0], centroids_to_interpolate[1])
    print("candidate centroid ", zip(x_interp, y_interp)[1])
    return zip(x_interp, y_interp)[1]

def get_previous_and_next_blob_wrt_gap(blobs_in_video, possible_identities, identity, frame_number):
    individual_gap_interval = find_the_individual_gap_interval(blobs_in_video,
                                        possible_identities,
                                        identity,
                                        frame_number)
    print("individual_gap_interval ", individual_gap_interval)
    previous_blob_to_the_gap = get_blob_by_identity(blobs_in_video[frame_number - 1], identity)
    next_blob_to_the_gap = get_blob_by_identity(blobs_in_video[individual_gap_interval[1]], identity)
    if len(previous_blob_to_the_gap) == 1:
        print("the individual gap has a previous")
        previous_blob_to_the_gap = previous_blob_to_the_gap[0]
    else:
        print(previous_blob_to_the_gap)
        print("first frame!!!!")

    if len(next_blob_to_the_gap) == 1:
        print("the individual gap has a next")
        next_blob_to_the_gap = next_blob_to_the_gap[0]
    else:
        print("last frame!!!!")
    return individual_gap_interval, previous_blob_to_the_gap, next_blob_to_the_gap

def get_eroded_blobs(video, blobs_in_frame):
    print("eroding all blobs in frame")
    segmented_frame = np.zeros((video.height, video.width)).astype('uint8')

    for blob in blobs_in_frame:
        pixels = blob.eroded_pixels if hasattr(blob,'eroded_pixels') else blob.pixels
        pxs = np.array(np.unravel_index(pixels,(video.height, video.width))).T
        segmented_frame[pxs[:,0], pxs[:,1]] = 255

    segmented_eroded_frame = erode(segmented_frame, video.erosion_kernel_size)
    # if blob.frame_number in [39, 40, 41, 42]:
    #     cv2.imshow("segmented_frame %i " %blob.frame_number, segmented_frame)
    #     cv2.imshow("segmented_eroded_frame %i " %blob.frame_number, segmented_eroded_frame)
    #     cv2.waitKey()
    _, _, centroids, _, pixels_all, contours, _ = blobExtractor(segmented_eroded_frame, segmented_eroded_frame, 0, np.inf)
    return [Blob(centroid, contour, None, None, pixels = pixels)
                for centroid, contour, pixels in zip(centroids, contours, pixels_all)]

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
    # if blob.frame_number in [39, 40, 41, 42]:
    cv2.imshow("segmented_frame %i " %blob.frame_number, segmented_frame)
    cv2.imshow("segmented_eroded_frame %i " %blob.frame_number, segmented_eroded_frame)
    cv2.waitKey()

def evaluate_candidate_blobs_and_centroid(video, candidate_eroded_blobs, candidate_centroid, blob_in_border_frame, blobs_in_frame = None):
    print("evaluating candidate blobs and candidate centroid")
    blob_containing_candidate_centroid = centroid_is_inside_of_any_eroded_blob(candidate_eroded_blobs, candidate_centroid)
    if blob_containing_candidate_centroid:
        print(" the centroid is inside of an eroded blob")
        return blob_containing_candidate_centroid[0], candidate_centroid
    else:
        print(" the centroid is not inside of an eroded blob, finding nearest_blob")
        nearest_blob = get_nearest_eroded_blob_to_candidate_centroid(candidate_eroded_blobs, candidate_centroid)
        if nearest_candidate_blob_is_near_enough(video, nearest_blob, candidate_centroid, blob_in_border_frame) or \
            eroded_blob_overlaps_with_blob_in_border_frame(eroded_blob, blob_in_border_frame):
            if cv2.pointPolygonTest(nearest_blob.contour, candidate_centroid, False) >= 0:
                print("  new candidate centroid ", nearest_blob.contour)
                return nearest_blob, candidate_centroid
            else:
                candidate_centroid = tuple([int(centroid_coordinate) for centroid_coordinate in candidate_centroid])
                new_centroid = get_closest_contour_point_to(nearest_blob.contour, candidate_centroid)
                print("  new candidate centroid moving ", new_centroid)
                return nearest_blob, new_centroid

        else:
            raise ValueError("uups no candidate blobs")

def centroid_is_inside_of_any_eroded_blob(candidate_eroded_blobs, candidate_centroid):
    candidate_centroid = tuple([int(centroid_coordinate) for centroid_coordinate in candidate_centroid])
    return [blob for blob in candidate_eroded_blobs
            if cv2.pointPolygonTest(blob.contour, candidate_centroid, False) >= 0]

def get_nearest_eroded_blob_to_candidate_centroid(eroded_blobs, candidate_centroid):
    eroded_blob_index = np.argmin([blob.distance_from_countour_to(candidate_centroid) for blob in eroded_blobs])
    return eroded_blobs[eroded_blob_index]

def nearest_candidate_blob_is_near_enough(video, candidate_blob, candidate_centroid, blob_in_border_frame):
    points = [candidate_centroid, blob_in_border_frame.centroid]
    distances = np.asarray([np.sqrt(candidate_blob.squared_distance_to(point)) for point in points])
    return np.any(distances < video.velocity_threshold)

def eroded_blob_overlaps_with_blob_in_border_frame(eroded_blob, blob_in_border_frame):
    return eroded_blob.overlaps_with(blob_in_border_frame)

def get_blobs_with_centroid_in_original_blob(original_blob, candidate_tuples_to_close_gap):
    candidate_tuples_with_centroids_in_original_blob = [candidate_tuple for candidate_tuple in candidate_tuples_to_close_gap
                                                            if cv2.pointPolygonTest(original_blob.contour,
                                                                                    tuple([int(c) for c in candidate_tuple[1]]), False) >= 0]
    return candidate_tuples_with_centroids_in_original_blob

def assign_identity_to_new_blobs(video, fragments, blobs_in_video, possible_identities, original_blobs, candidate_tuples_to_close_gap):
    new_original_blobs = []
    print('\n')
    for i, original_blob in enumerate(original_blobs):
        print("len original_blobs ", len(original_blobs))
        print("-original blob ", i,
                " fragment_identifier ", original_blob.fragment_identifier,
                " final_identity ", original_blob.final_identity ,
                " is_an_individual ", original_blob.is_an_individual,
                " is_a_crossing ", original_blob.is_a_crossing)
        candidate_tuples_with_centroids_in_original_blob = get_blobs_with_centroid_in_original_blob(original_blob, candidate_tuples_to_close_gap)
        if len(candidate_tuples_with_centroids_in_original_blob) == 1: # the gap is a single individual blob
            print("this original blob only has an eroded blob inside")
            identity = candidate_tuples_with_centroids_in_original_blob[0][2]
            centroid = candidate_tuples_with_centroids_in_original_blob[0][1]
            if original_blob.final_identity == 0 and original_blob.is_an_individual:
                print("is and individual with identity 0 and we propagate")
                original_blob._identity_corrected_closing_gaps = identity
                [setattr(blob,'_identity_corrected_closing_gaps', identity)
                    for blobs_in_frame in blobs_in_video for blob in blobs_in_frame
                    if blob.fragment_identifier == original_blob.fragment_identifier]
            elif original_blob.is_an_individual:
                print("is probably a failure of the model area")
                identity = [original_blob.final_identity, identity]
                centroid = [original_blob.centroid, centroid]
                original_blob._identity_corrected_closing_gaps = identity
                original_blob.interpolated_centroids = centroid
                [setattr(blob,'interpolated_centroids', centroid)
                    for blobs_in_frame in blobs_in_video for blob in blobs_in_frame
                    if blob.fragment_identifier == original_blob.fragment_identifier]
                [setattr(blob,'_identity_corrected_closing_gaps', identity)
                    for blobs_in_frame in blobs_in_video for blob in blobs_in_frame
                    if blob.fragment_identifier == original_blob.fragment_identifier]
            elif original_blob.is_a_crossing:
                print("is probably a failure of the model area")
                if original_blob.final_identity is not None:
                    identity = [original_blob.final_identity, identity]
                    centroid = [original_blob.centroid, centroid]
                else:
                    identity = [identity]
                    centroid = [centroid]
                frame_number = original_blob.frame_number
                original_blob = candidate_tuples_with_centroids_in_original_blob[0][0]
                original_blob.frame_number = frame_number
                original_blob._identity_corrected_closing_gaps = identity
                original_blob.interpolated_centroids = centroid

            new_original_blobs.append(original_blob)
        elif len(candidate_tuples_with_centroids_in_original_blob) > 1: # the gap is a crossing
            print("this original blob has more than one eroded blob inside")
            candidate_eroded_blobs = zip(*candidate_tuples_with_centroids_in_original_blob)[0]
            candidate_eroded_blobs_centroids = zip(*candidate_tuples_with_centroids_in_original_blob)[1]
            candidate_eroded_blobs_identities = zip(*candidate_tuples_with_centroids_in_original_blob)[2]
            if len(set(candidate_eroded_blobs)) == 1: # crossing not split
                print(" the eroded blobs inside of the original blob are the same, crossing NOT splitted")
                original_blob.interpolated_centroids = [candidate_eroded_blob_centroid
                                                                    for candidate_eroded_blob_centroid in candidate_eroded_blobs_centroids]
                original_blob._identity_corrected_closing_gaps = [candidate_eroded_blob_identity
                                                                    for candidate_eroded_blob_identity in candidate_eroded_blobs_identities]
                print("    centroids in crossing ", original_blob.interpolated_centroids)
                print("    identities in crossing ", original_blob._identity_corrected_closing_gaps)
                original_blob.eroded_pixels = candidate_eroded_blobs[0].pixels
                new_original_blobs.append(original_blob)

            elif len(set(candidate_eroded_blobs)) > 1: # crossing split
                list_of_new_blobs_in_next_frames = []
                print(" the eroded blobs inside of the original blob are NOT the same, crossing splitted")
                count_eroded_blobs = {eroded_blob: candidate_eroded_blobs.count(eroded_blob)
                                        for eroded_blob in candidate_eroded_blobs}
                print("    there are %i eroded blobs inside the crossing " %len(count_eroded_blobs))
                for j, (eroded_blob, centroid, identity) in enumerate(candidate_tuples_with_centroids_in_original_blob):
                    print("    -eroded blob ", j)
                    if count_eroded_blobs[eroded_blob] == 1: # split blob, single individual
                        print("     the eroded blob is unique so it must be an individual")
                        eroded_blob.frame_number = original_blob.frame_number
                        eroded_blob.centroid = centroid
                        eroded_blob._identity_corrected_closing_gaps = identity
                        eroded_blob._is_an_individual = True
                        new_original_blobs.append(eroded_blob)
                    elif count_eroded_blobs[eroded_blob] > 1:
                        print("     the eroded blob is repeated so it must be an crossing")
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
    return blobs_in_video

def interpolate_trajectories_during_gaps(video, list_of_blobs):
    if len(list_of_blobs.blobs_in_video[-1]) == 0:
        list_of_blobs.blobs_in_video = list_of_blobs.blobs_in_video[:-1]
    possible_identities = range(1, video.number_of_animals + 1)
    blobs_in_video = list_of_blobs.blobs_in_video
    video._erosion_kernel_size = compute_erosion_disk(video, blobs_in_video)

    for frame_number, blobs_in_frame in enumerate(tqdm(blobs_in_video, desc = "closing gaps")):
        if frame_number != 0:
            print("***frame_number %i" %frame_number)
            missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame)
            if len(missing_identities) > 0:
                print("\nThere are missing identities: ", missing_identities)
                gap_interval = find_the_gap_interval(blobs_in_video, possible_identities, frame_number)
                eroded_blobs_in_frame = get_eroded_blobs(video, blobs_in_frame) #list of eroded blobs!
                candidate_tuples_to_close_gap = []

                for identity in missing_identities:
                    print("**identity ", identity)
                    individual_gap_interval, previous_blob_to_the_gap, next_blob_to_the_gap = get_previous_and_next_blob_wrt_gap(blobs_in_video,
                                                                                                                                possible_identities,
                                                                                                                                identity,
                                                                                                                                frame_number)
                    candidate_centroid = get_candidate_centroid(individual_gap_interval,
                                                                previous_blob_to_the_gap,
                                                                next_blob_to_the_gap,
                                                                identity)
                    candidate_eroded_blobs = get_candidate_blobs(previous_blob_to_the_gap, eroded_blobs_in_frame)
                    candidate_blob_to_close_gap, centroid = evaluate_candidate_blobs_and_centroid(video,
                                                                                                    candidate_eroded_blobs,
                                                                                                    candidate_centroid,
                                                                                                    previous_blob_to_the_gap,
                                                                                                    blobs_in_frame = blobs_in_frame)
                    candidate_tuples_to_close_gap.append((candidate_blob_to_close_gap, centroid, identity))

                print('\n-- Assigning final identity to the %i candidate blobs to close the gap' %len(candidate_tuples_to_close_gap))
                print("the candidate blobs are ", [b[0] for b in candidate_tuples_to_close_gap])
                print("the centroids of the candidate blobs are ", [b[1] for b in candidate_tuples_to_close_gap])
                print("the identities of the candidate blobs are ", [b[2] for b in candidate_tuples_to_close_gap])
                blobs_in_video = assign_identity_to_new_blobs(video, list_of_fragments.fragments,
                                            blobs_in_video, possible_identities,
                                            blobs_in_frame, candidate_tuples_to_close_gap)
                print("\n")

    return blobs_in_video


if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/conflict8Short/session_3/video_object.npy').item()
    list_of_fragments = ListOfFragments.load(video.fragments_path)
    list_of_blobs = ListOfBlobs.load(video.blobs_path)
    list_of_blobs.update_from_list_of_fragments(list_of_fragments.fragments)
    blobs_in_video = interpolate_trajectories_during_gaps(video, list_of_blobs)
    list_of_blobs.blobs_in_video = blobs_in_video
    blobs_no_gaps_path = os.path.join(os.path.split(video.blobs_path)[0], 'blobs_collection_no_gaps.npy')
    video.blobs_no_gaps_path = blobs_no_gaps_path
    video.save()
    list_of_blobs.save(path_to_save = blobs_no_gaps_path, number_of_chunks = video.number_of_frames)































# def propagate_identity_of_individual_eroded_blob(video, blobs_in_video, possible_identities, eroded_blob, candidate_eroded_blobs):
#     identity = eroded_blob.identity_corrected_closing_gaps
#     current_eroded_blob = eroded_blob
#     frame_number = eroded_blob.frame_number + 1
#     eroded_blob_is_in_an_individual_fragment = True
#
#     print("     propagating the identity %i to the next frame for the single individual eroded blob" %identity)
#     list_of_new_blobs_in_frame = []
#     while eroded_blob_is_in_an_individual_fragment:
#         print("      frame ", frame_number)
#         blobs_in_frame = blobs_in_video[frame_number]
#         missing_identities = get_missing_identities_from_blobs_in_frame(possible_identities, blobs_in_frame)
#         print("       missing_identities ", missing_identities)
#         new_blobs_in_frame = []
#         if identity in missing_identities:
#             print("       the identity is still missing")
#             eroded_blobs_in_frame = get_eroded_blobs(video, blobs_in_frame)
#             next_frame_candidate_eroded_blobs = get_candidate_blobs(current_eroded_blob, eroded_blobs_in_frame)
#             if len(next_frame_candidate_eroded_blobs) == 1:
#                 print("       single candidate eroded blob")
#                 current_frame_eroded_blobs_overlapping_with_next_candidate = get_candidate_blobs(next_frame_candidate_eroded_blobs[0],
#                                                                                                     candidate_eroded_blobs)
#                 if len(current_frame_eroded_blobs_overlapping_with_next_candidate) == 1:
#                     print("       the overlapping with the next frame is unique")
#                     #### we want to create a new list of blobs_in_frame for every step in the while
#                     # as far as we split at least one blob. We will update the blobs_in_video afterwards
#                     # when we finish the outer for loop of the original_blobs
#                     current_eroded_blob = next_frame_candidate_eroded_blobs[0]
#                     current_eroded_blob._identity_corrected_closing_gaps = identity
#                     current_eroded_blob.frame_number = frame_number
#                     new_blobs_in_frame.append(current_eroded_blob)
#                     frame_number += 1
#                 else:
#                     print("       the overlapping with the next frame is NOT unique")
#                     eroded_blob_is_in_an_individual_fragment = False
#             else:
#                 list_of_new_blobs_in_frame.append(None)
#                 print("       multiple candidate eroded blob")
#                 eroded_blob_is_in_an_individual_fragment = False
#         else:
#             print("       the identity is NOT missing")
#             list_of_new_blobs_in_frame.append(None)
#             eroded_blob_is_in_an_individual_fragment = False
#
#     return list_of_new_blobs_in_frame


#
# def propagate(eroded_blobs, blobs_in_video, frame_number, possible_identities):
#     frame_number += 1
#     missing_identities = get_missing_identities_from_blobs_in_frame(blobs_in_video[frame_number],
#                                                                     possible_identities)
#     identities = set([blob.identity_corrected_closing_gaps
#                     for blob in eroded_blobs
#                     if blob.is_an_individual])
#     identities_to_propagate = missing_identities & identities
#
#     while len(identities_to_propagate) > 0:
#         blobs_in_frame = blobs_in_video[frame_number]
#         eroded_blobs_in_current_frame = get_eroded_blobs(video, blobs_in_frame)
#         blobs_to_be_removed = []
#         blobs_to_be_added = []
#
#         for identity_to_propagate in identities_to_propagate:
#
#             current_eroded_blob = [blob for blob in eroded_blobs
#                                     if blob.identity_corrected_closing_gaps == identity_to_propagate]
#             next_frame_candidate_eroded_blobs = get_candidate_blobs(current_eroded_blob, eroded_blobs_in_current_frame)
#             if len(next_frame_candidate_eroded_blobs) == 1:
#                 print("       single candidate eroded blob")
#                 current_frame_eroded_blobs_overlapping_with_next_candidate = get_candidate_blobs(next_frame_candidate_eroded_blobs[0],
#                                                                                                     candidate_eroded_blobs)
#                 if len(current_frame_eroded_blobs_overlapping_with_next_candidate) == 1:
#                     print("       the overlapping with the next frame is unique")
#                     current_eroded_blob = next_frame_candidate_eroded_blobs[0]
#                     current_eroded_blob._identity_corrected_closing_gaps = identity
#                     current_eroded_blob.frame_number = frame_number
#                     current_eroded_blob.is_an_individual = True
#                     original_blob_in_current_frame = get_candidate_blobs(current_eroded_blob, blobs_in_frame)
#                     new_blobs = get_candidate_blobs(original_blob_in_current_frame[0], eroded_blobs_in_current_frame)
#                     blobs_to_be_added.extend(new_blobs)
#                     blobs_to_be_removed.extend(original_blob_in_current_frame)
#                 else:
#                     print("       the overlapping with the next frame is NOT unique")
#                     eroded_blob_is_in_an_individual_fragment = False
#             else:
#                 list_of_new_blobs_in_frame.append(None)
#                 print("       multiple candidate eroded blob")
#                 eroded_blob_is_in_an_individual_fragment = False
#
#         frame_number += 1
