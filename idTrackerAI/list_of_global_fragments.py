from __future__ import absolute_import, division, print_function
import os
import random
import logging

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from globalfragment import GlobalFragment

logger = logging.getLogger("__main__.list_of_global_fragments")

class ListOfGlobalFragments(object):
    def __init__(self, video, global_fragments):
        self.video = video
        self.global_fragments = global_fragments
        self.number_of_global_fragments = len(self.global_fragments)

    def reset(self, roll_back_to = None):
        [global_fragment.reset(roll_back_to) for global_fragment in self.global_fragments]

    def order_by_distance_travelled(self):
        self.global_fragments = sorted(self.global_fragments, key = lambda x: x.minimum_distance_travelled, reverse = True)

    def order_by_distance_to_the_first_global_fragment(self):
        self.order_by_distance_travelled()
        self.global_fragments = sorted(self.global_fragments,
                                        key = lambda x: np.abs(x.index_beginning_of_fragment - self.video.first_frame_first_global_fragment),
                                        reverse = False)

    def compute_number_of_unique_images(self):
        individual_fragments_used = []
        number_of_images = 0

        for global_fragment in self.global_fragments:

            for fragment in global_fragment.individual_fragments:

                if fragment.identifier not in individual_fragments_used:
                    number_of_images += fragment.number_of_images
                    individual_fragments_used.append(fragment.identifier)

        self.number_of_images

    def compute_maximum_number_of_images(self):
        self.maximum_number_of_images = np.max([global_fragment.get_total_number_of_images() for global_fragment in self.global_fragments])

    def filter_by_minimum_number_of_frames(self, minimum_number_of_frames = 3):
        self.global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if np.min(global_fragment.number_of_images_per_individual_fragment) >= minimum_number_of_frames]
        self.number_of_global_fragments = len(self.global_fragments)

    def get_data_plot(self):
        number_of_images_in_shortest_individual_fragment = []
        number_of_images_in_longest_individual_fragment = []
        number_of_images_per_individual_fragment_in_global_fragment = []
        median_number_of_images = []
        minimum_distance_travelled = []
        for global_fragment in self.global_fragments:
            number_of_images_in_shortest_individual_fragment.append(min(global_fragment.number_of_images_per_individual_fragment))
            number_of_images_in_longest_individual_fragment.append(max(global_fragment.number_of_images_per_individual_fragment))
            number_of_images_per_individual_fragment_in_global_fragment.append(global_fragment.number_of_images_per_individual_fragment)
            median_number_of_images.append(np.median(global_fragment.number_of_images_per_individual_fragment))
            minimum_distance_travelled.append(min(global_fragment.distance_travelled_per_individual_fragment))

        return number_of_images_in_shortest_individual_fragment,\
                number_of_images_in_longest_individual_fragment,\
                number_of_images_per_individual_fragment_in_global_fragment,\
                median_number_of_images,\
                minimum_distance_travelled

    def delete_fragments_from_global_fragments(self):
        [setattr(global_fragment,'individual_fragments',None) for global_fragment in self.global_fragments]

    def relink_fragments_to_global_fragments(self, fragments):
        [global_fragment.get_individual_fragments_of_global_fragment(fragments) for global_fragment in self.global_fragments]

    def save(self):
        logger.info("saving list of global fragments at %s" %self.video.global_fragments_path)
        self.delete_fragments_from_global_fragments()
        np.save(self.video.global_fragments_path,self)

    @classmethod
    def load(self, path_to_load, fragments):
        logger.info("loading list of global fragments from %s" %path_to_load)
        list_of_global_fragments = np.load(path_to_load).item()
        list_of_global_fragments.relink_fragments_to_global_fragments(fragments)
        return list_of_global_fragments

def detect_beginnings(boolean_array):
    """ detects the frame where the core of a global fragment starts.
    A core of a global fragment is the part of the global fragment where all the
    individuals are visible, i.e. the number of animals in the frame equals the
    number of animals in the video
    :boolean_array: array with True where the number of animals in the frame equals
    the number of animals in the video
    """
    return [i for i in range(0,len(boolean_array)) if (boolean_array[i] and not boolean_array[i-1])]

def check_global_fragments(blobs_in_video, num_animals):
    """Returns an array with True iff:
    * each blob has a unique blob intersecting in the past and future
    * number of blobs equals num_animals
    """
    def all_blobs_in_a_fragment(blobs_in_frame):
        return all([blob.is_in_a_fragment for blob in blobs_in_frame])

    return [all_blobs_in_a_fragment(blobs_in_frame) and len(blobs_in_frame) == num_animals for blobs_in_frame in blobs_in_video]

def create_list_of_global_fragments(blobs_in_video, fragments, num_animals):
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video, fragments, i, num_animals) for i in indices_beginning_of_fragment]

def get_images_and_labels_from_global_fragment(list_of_fragments, global_fragment, individual_fragments_identifiers_already_used = []):
    images = []
    labels = []
    lengths = []
    individual_fragments_identifiers = []

    for fragment in global_fragment.individual_fragments:
        if fragment.identifier not in individual_fragments_identifiers_already_used :
            images.extend(fragment.images)
            labels.extend([fragment.temporary_id] * fragment.number_of_images)
            lengths.append(fragment.number_of_images)
            individual_fragments_identifiers.append(fragment.identifier)

    return images, labels, lengths, individual_fragments_identifiers

def get_images_and_labels_from_global_fragments(list_of_fragments, global_fragments, individual_fragments_identifiers_already_used = []):
    logger.info("Getting images from global fragments")
    images = []
    labels = []
    lengths = []
    candidate_individual_fragments_identifiers = []
    individual_fragments_identifiers_already_used = list(individual_fragments_identifiers_already_used)

    for global_fragment in global_fragments:
        images_global_fragment, \
        labels_global_fragment, \
        lengths_global_fragment, \
        individual_fragments_identifiers = get_images_and_labels_from_global_fragment(list_of_fragments, global_fragment,
                                                                                        individual_fragments_identifiers_already_used)
        if len(images_global_fragment) != 0:
            images.append(images_global_fragment)
            labels.append(labels_global_fragment)
            lengths.extend(lengths_global_fragment)
            candidate_individual_fragments_identifiers.extend(individual_fragments_identifiers)
            individual_fragments_identifiers_already_used.extend(individual_fragments_identifiers)
    if len(images) != 0:
        return np.concatenate(images, axis = 0), np.concatenate(labels, axis = 0), candidate_individual_fragments_identifiers, np.cumsum(lengths)[:-1]
    else:
        return None, None, candidate_individual_fragments_identifiers, None
