from __future__ import absolute_import, print_function, division
import collections
import numpy as np
import cv2
from pprint import pprint

def get_crossing_fragments(fragments):
    return [fragment for fragment in fragments if fragment.is_a_crossing
            and not fragment.is_a_ghost_crossing]

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_non_admissible_identities_in_crossing(crossing_fragment):
    return set([fragment.final_identity for fragment in crossing_fragment.coexisting_individual_fragments])

def get_fragments_in_direction(video, fragments, crossing_fragment, direction):
    return [fragments[video.fragment_identifier_to_index[fragment_identifier]]
                for fragment_identifier in getattr(crossing_fragment, direction + '_blobs_fragment_identifier')]

def get_identities_in_direction(video, fragments, crossing_fragment, direction):
    fragments_in_direction = get_fragments_in_direction(video, fragments, crossing_fragment, direction = direction)
    return [fragment.final_identity for fragment in fragments_in_direction]

def get_identities_in_crossing(video, fragments, crossing_fragment, direction):
    setattr(crossing_fragment,
            direction +'_identities',
            get_identities_in_direction(video, fragments, crossing_fragment, direction))

def get_final_identity_crossing_fragment(crossing_fragment):
    previous_ids = set(flatten(crossing_fragment.previous_identities))
    if None in previous_ids: previous_ids.remove(None)
    next_ids = set(flatten(crossing_fragment.next_identities))
    if None in next_ids: next_ids.remove(None)
    return list((previous_ids | next_ids) - get_non_admissible_identities_in_crossing(crossing_fragment))

def set_identities_in_crossings(video, fragments):
    crossing_fragments = get_crossing_fragments(fragments)
    [get_identities_in_crossing(video, fragments, crossing_fragment, 'previous')
        for crossing_fragment in crossing_fragments]
    [get_identities_in_crossing(video, fragments, crossing_fragment, 'next')
        for crossing_fragment in crossing_fragments[::-1]]
    [setattr(crossing_fragment, '_identity', get_final_identity_crossing_fragment(crossing_fragment))
        for crossing_fragment in crossing_fragments]
    return crossing_fragments

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

def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image,kernel,iterations = 1)

def erode_crossing_images(crossing_fragment, erosion_kernel_size):
    eroded_images = []

    for pixels, bb in zip(crossing_fragment.pixels, crossing_fragment.bounding_box_in_frame_coordinates):
        temp_image = generate_temp_image(video, pixels, bb)
        eroded_image = erode(temp_image, erosion_kernel_size)
        contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    return eroded_images

def get_pixels_from_bw_image(image):
    return np.where(image == 255)[0]

def get_border_fragment_with_given_identity(fragments, identity):
    return [fragment for fragment in fragments if fragment.final_identity == identity]

def get_border_fragment_pixels_wrt_identity(fragments, identity):
    return get_border_fragment_with_given_identity(fragments, identity)[0].pixels


def assign_crossing_by_individual(identity, crossing_fragment):
    fragments_past = get_fragments_in_direction(video,
                                    fragments,
                                    crossing_fragment,
                                    direction = 'previous')
    fragments_future = get_fragments_in_direction(video,
                                    fragments,
                                    crossing_fragment,
                                    direction = 'next')
    pixels_past = get_border_fragment_pixels_wrt_identity(fragments_past, identity)[-1]
    pixels_future = get_border_fragment_pixels_wrt_identity(fragments_future, identity)[0]





def assign_identity_crossing(crossing_fragment, erosion_kernel_size):
    if len(crossing_fragment.final_identity) == 1:
        crossing_fragment._identity =  crossing_fragment.identity[0]
        crossing_fragment._is_a_crossing = False
        crossing_fragment._is_an_individual = True
    elif len(crossing_fragment.final_identity) > 1:
        eroded_images = erode_crossing_images(crossing_fragment,
                                                erosion_kernel_size)

        for identity in crossing_fragment.final_identity:
            assign_crossing_by_individual(identity, crossing_fragment)

if __name__ == "__main__":
    crossing_fragments = set_identities_in_crossings(video, list_of_fragments.fragments)
    erosion_kernel_size = compute_erosion_disk(video, list_of_blobs.blobs_in_video)
