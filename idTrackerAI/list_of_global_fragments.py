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

    @staticmethod
    def give_me_frequencies_first_fragment_accumulated(i, number_of_animals, fragment):
        frequencies = np.zeros(number_of_animals)
        frequencies[i] = fragment.number_of_images
        return frequencies

    def set_first_global_fragment_for_accumulation(self, accumulation_trial):
        self.order_by_distance_travelled()
        self.first_global_fragment_for_accumulation = self.global_fragments[accumulation_trial]
        [(setattr(fragment, '_acceptable_for_training', True),
            setattr(fragment, '_temporary_id', i),
            setattr(fragment, '_frequencies', self.give_me_frequencies_first_fragment_accumulated(i, self.video.number_of_animals, fragment)),
            setattr(fragment, '_is_certain', True),
            setattr(fragment, '_certainty', 1.),
            setattr(fragment, '_P1_vector', fragment.compute_P1_from_frequencies(fragment.frequencies)))
            for i, fragment in enumerate(self.first_global_fragment_for_accumulation.individual_fragments)]
        self.video._first_frame_first_global_fragment = self.first_global_fragment_for_accumulation.index_beginning_of_fragment
        self.video.save()

    def order_by_distance_to_the_first_global_fragment_for_accumulation(self):
        self.global_fragments = sorted(self.global_fragments,
                                        key = lambda x: np.abs(x.index_beginning_of_fragment - self.video.first_frame_first_global_fragment),
                                        reverse = False)

    def compute_maximum_number_of_images(self):
        self.maximum_number_of_images = max([global_fragment.get_total_number_of_images() for global_fragment in self.global_fragments])

    def filter_candidates_global_fragments_for_accumulation(self):
        self.non_accumulable_global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if not global_fragment.candidate_for_accumulation]
        self.global_fragments = [global_fragment for global_fragment in self.global_fragments
                    if global_fragment.candidate_for_accumulation]
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

    def save(self, fragments):
        logger.info("saving list of global fragments at %s" %self.video.global_fragments_path)
        self.delete_fragments_from_global_fragments()
        np.save(self.video.global_fragments_path,self)
        # After saving the list of globa fragments the individual fragments are deleted and we need to relink them again
        self.relink_fragments_to_global_fragments(fragments)

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
        # return all([blob.is_in_a_fragment for blob in blobs_in_frame])
        return all([blob.is_an_individual for blob in blobs_in_frame])

    return [all_blobs_in_a_fragment(blobs_in_frame) and len(blobs_in_frame) == num_animals for blobs_in_frame in blobs_in_video]

def create_list_of_global_fragments(blobs_in_video, fragments, num_animals):
    global_fragments_boolean_array = check_global_fragments(blobs_in_video, num_animals)
    indices_beginning_of_fragment = detect_beginnings(global_fragments_boolean_array)
    return [GlobalFragment(blobs_in_video, fragments, i, num_animals) for i in indices_beginning_of_fragment]
