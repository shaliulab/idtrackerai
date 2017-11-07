from __future__ import absolute_import, division, print_function
import os
import sys
import random
import logging

sys.path.append('./utils')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from tqdm import tqdm

from fragment import Fragment
from py_utils import set_attributes_of_object_to_value, append_values_to_lists

logger = logging.getLogger("__main__.list_of_fragments")

class ListOfFragments(object):
    def __init__(self, video, fragments):
        self.video = video
        self.fragments = fragments
        self.number_of_fragments = len(self.fragments)

    def get_fragment_identifier_to_index_list(self):
        fragments_identifiers = [fragment.identifier for fragment in self.fragments]
        fragment_identifier_to_index = np.arange(len(fragments_identifiers))
        fragments_identifiers_argsort = np.argsort(fragments_identifiers)
        return fragment_identifier_to_index[fragments_identifiers_argsort]

    def reset(self, roll_back_to = None):
        [fragment.reset(roll_back_to) for fragment in self.fragments]

    def get_images_from_fragments_to_assign(self):
        return np.concatenate([np.asarray(fragment.images) for fragment in self.fragments
                                if not fragment.used_for_training and fragment.is_an_individual], axis = 0)

    def compute_number_of_unique_images_used_for_pretraining(self):
        return sum([fragment.number_of_images for fragment in self.fragments if fragment.used_for_pretraining])

    def compute_number_of_unique_images_used_for_training(self):
        return sum([fragment.number_of_images for fragment in self.fragments if fragment.used_for_training])

    def compute_total_number_of_images_in_global_fragments(self):
        self.number_of_images_in_global_fragments = sum([fragment.number_of_images for fragment in self.fragments if fragment.is_in_a_global_fragment])

    def compute_ratio_of_images_used_for_pretraining(self):
        return self.compute_number_of_unique_images_used_for_pretraining() / self.number_of_images_in_global_fragments

    def compute_ratio_of_images_used_for_training(self):
        return self.compute_number_of_unique_images_used_for_training() / self.number_of_images_in_global_fragments


    def get_data_plot(self):
        number_of_images_in_individual_fragments = []
        distance_travelled_individual_fragments = []
        number_of_images_in_crossing_fragments = []
        for fragment in self.fragments:
            if fragment.is_an_individual:
                number_of_images_in_individual_fragments.append(fragment.number_of_images)
                distance_travelled_individual_fragments.append(fragment.distance_travelled)
            elif fragment.is_a_crossing:
                number_of_images_in_crossing_fragments.append(fragment.number_of_images)
        return np.asarray(number_of_images_in_individual_fragments),\
                np.asarray(distance_travelled_individual_fragments),\
                number_of_images_in_crossing_fragments

    def update_from_list_of_blobs(self, blobs_in_video):
        [setattr(self.fragments[self.video.fragment_identifier_to_index[blob.fragment_identifier]], '_user_generated_identity', blob.user_generated_identity)
            for blobs_in_frame in blobs_in_video for blob in blobs_in_frame if blob.user_generated_identity is not None ]

    def get_ordered_list_of_fragments(self, scope = None):
        if scope == 'to_the_past':
            fragments_subset = [fragment for fragment in self.fragments if fragment.start_end[1] <= self.video.first_frame_first_global_fragment]
            fragments_subset.sort(key=lambda x: x.start_end[1], reverse=True)
        elif scope == 'to_the_future':
            fragments_subset = [fragment for fragment in self.fragments if fragment.start_end[0] >= self.video.first_frame_first_global_fragment]
            fragments_subset.sort(key=lambda x: x.start_end[0], reverse=False)
        return fragments_subset

    def save(self):
        logger.info("saving list of fragments at %s" %self.video.fragments_path)
        [setattr(fragment, 'coexisting_individual_fragments', None) for fragment in self.fragments]
        np.save(self.video.fragments_path,self)
        [fragment.get_coexisting_individual_fragments_indices(self.fragments) for fragment in self.fragments]

    @classmethod
    def load(self, path_to_load):
        logger.info("loading list of fragments from %s" %path_to_load)
        list_of_fragments = np.load(path_to_load).item()
        [fragment.get_coexisting_individual_fragments_indices(list_of_fragments.fragments) for fragment in list_of_fragments.fragments]
        return list_of_fragments

    def create_light_list(self, attributes = None):
        if attributes == None:
            attributes_to_discard = ['images',
                            'pixels',
                            'coexisting_individual_fragments']
        return [{attribute: getattr(fragment, attribute) for attribute in fragment.__dict__.keys() if attribute not in attributes_to_discard}
                    for fragment in self.fragments]

    def save_light_list(self, accumulation_folder):
        np.save(os.path.join(accumulation_folder, 'light_list_of_fragments.npy'), self.create_light_list())

    def load_light_list(self, accumulation_folder):
        list_of_dictionaries = np.load(os.path.join(accumulation_folder, 'light_list_of_fragments.npy'))
        self.update_fragments_dictionary(list_of_dictionaries)

    def update_fragments_dictionary(self, list_of_dictionaries):
        assert len(list_of_dictionaries) == len(self.fragments)
        [fragment.__dict__.update(dictionary) for fragment, dictionary in zip(self.fragments, list_of_dictionaries)]

    def get_new_images_and_labels_for_training(self):
        images = []
        labels = []
        for fragment in self.fragments:
            if fragment.acceptable_for_training and not fragment.used_for_training:
                assert fragment.is_an_individual
                images.append(fragment.images)
                labels.extend([fragment.temporary_id] * fragment.number_of_images)
        if len(images) != 0:
            return np.concatenate(images, axis = 0), np.asarray(labels)
        else:
            return None, None

    def get_accumulable_individual_fragments_identifiers(self, list_of_global_fragments):
        self.accumulable_individual_fragments = set([identifier for global_fragment in list_of_global_fragments.global_fragments
                                                                        for identifier in global_fragment.individual_fragments_identifiers])

    def get_not_accumulable_individual_fragments_identifiers(self, list_of_global_fragments):
        '''list of individual fragments in global fragments that cannot be accumulated because are too small
        '''
        self.not_accumulable_individual_fragments = set([identifier for global_fragment in list_of_global_fragments.non_accumulable_global_fragments
                                                        for identifier in global_fragment.individual_fragments_identifiers]) - self.accumulable_individual_fragments

    def set_fragments_as_accumulable_or_not_accumulable(self):
        for fragment in self.fragments:
            if fragment.identifier in self.accumulable_individual_fragments:
                setattr(fragment, '_accumulable', True)
            elif fragment.identifier in self.not_accumulable_individual_fragments:
                setattr(fragment, '_accumulable', False)
            else:
                setattr(fragment, '_accumulable', None)

    def get_stats(self, list_of_global_fragments):
        # number of fragments per class
        self.number_of_crossing_fragments = sum([fragment.is_a_crossing for fragment in self.fragments])
        self.number_of_individual_fragments = sum([fragment.is_an_individual for fragment in self.fragments])
        self.number_of_individual_fragments_not_in_a_global_fragment = sum([not fragment.is_in_a_global_fragment
                                                                        and fragment.is_an_individual
                                                                        for fragment in self.fragments])
        self.number_of_accumulable_individual_fragments = len(self.accumulable_individual_fragments)
        self.number_of_not_accumulable_individual_fragments = len(self.not_accumulable_individual_fragments)
        fragments_not_accumualted = set([fragment.identifier for fragment in self.fragments if not fragment.used_for_training])
        self.number_of_not_accumulated_individual_fragments = len(self.accumulable_individual_fragments & fragments_not_accumualted)
        self.number_of_globally_accumulated_individual_fragments = sum([fragment.accumulated_globally
                                                                    and fragment.is_an_individual for fragment in self.fragments])
        self.number_of_partially_accumulated_individual_fragments = sum([fragment.accumulated_partially
                                                                    and fragment.is_an_individual for fragment in self.fragments])

        # number of blobs per class
        self.number_of_blobs = sum([fragment.number_of_images for fragment in self.fragments])
        self.number_of_crossing_blobs = sum([fragment.is_a_crossing * fragment.number_of_images for fragment in self.fragments])
        self.number_of_individual_blobs = sum([fragment.is_an_individual * fragment.number_of_images for fragment in self.fragments])
        self.number_of_individual_blobs_not_in_a_global_fragment = sum([(not fragment.is_in_a_global_fragment
                                                                        and fragment.is_an_individual) * fragment.number_of_images
                                                                        for fragment in self.fragments])
        self.number_of_accumulable_individual_blobs = sum([fragment.accumulable * fragment.number_of_images for fragment in self.fragments
                                                                if fragment.accumulable is not None])
        self.number_of_not_accumulable_individual_blobs = sum([(not fragment.accumulable) * fragment.number_of_images for fragment in self.fragments
                                                                if fragment.accumulable is not None])
        fragments_not_accumualted = self.accumulable_individual_fragments & set([fragment.identifier for fragment in self.fragments if not fragment.used_for_training])
        self.number_of_not_accumulated_individual_blobs = sum([fragment.number_of_images for fragment in self.fragments if fragment.identifier in fragments_not_accumualted])
        self.number_of_globally_accumulated_individual_blobs = sum([(fragment.accumulated_globally
                                                                    and fragment.is_an_individual) * fragment.number_of_images for fragment in self.fragments if fragment.accumulated_globally is not None])
        self.number_of_partially_accumulated_individual_blobs = sum([(fragment.accumulated_partially
                                                                    and fragment.is_an_individual) * fragment.number_of_images for fragment in self.fragments if fragment.accumulated_partially is not None])


        logger.info('number_of_fragments %i' %self.number_of_fragments)
        logger.info('number_of_crossing_fragments %i' %self.number_of_crossing_fragments)
        logger.info('number_of_individual_fragments %i ' %self.number_of_individual_fragments)
        logger.info('number_of_individual_fragments_not_in_a_global_fragment %i' %self.number_of_individual_fragments_not_in_a_global_fragment)
        logger.info('number_of_accumulable_individual_fragments %i' %self.number_of_accumulable_individual_fragments)
        logger.info('number_of_not_accumulable_individual_fragments %i' %self.number_of_not_accumulable_individual_fragments)
        logger.info('number_of_not_accumulated_individual_fragments %i' %self.number_of_not_accumulated_individual_fragments)
        logger.info('number_of_globally_accumulated_individual_fragments %i' %self.number_of_globally_accumulated_individual_fragments)
        logger.info('number_of_partially_accumulated_individual_fragments %i' %self.number_of_partially_accumulated_individual_fragments)

        attributes_to_return = ['number_of_fragments', 'number_of_crossing_fragments', 'number_of_individual_fragments',
                                    'number_of_individual_fragments_not_in_a_global_fragment',
                                    'number_of_accumulable_individual_fragments',
                                    'number_of_not_accumulable_individual_fragments',
                                    'number_of_accumualted_individual_fragments',
                                    'number_of_globally_accumulated_individual_fragments',
                                    'number_of_partially_accumulated_individual_fragments',
                                    'number_of_blobs', 'number_of_crossing_blobs', 'number_of_individual_blobs',
                                    'number_of_individual_blobs_not_in_a_global_fragment',
                                    'number_of_accumulable_individual_blobs',
                                    'number_of_not_accumulable_individual_blobs',
                                    'number_of_accumualted_individual_blobs',
                                    'number_of_globally_accumulated_individual_blobs',
                                    'number_of_partially_accumulated_individual_blobs']
        return {key: getattr(self, key) for key in self.__dict__ if key in attributes_to_return}

    def plot_stats(self):
        plt.ion()
        fig, ax = plt.subplots(1,1)
        sns.set_style("ticks")
        colors = ['grey', 'y', 'y', 'r', 'g', 'g']
        hatches = ['', '', '/', '', '', '//']
        labels = ['crossings', 'not in GF', 'not accumulable', 'not accumulated', 'globally accumulated', 'partially accumulated']
        sizes = np.asarray([self.number_of_crossing_blobs,
                            self.number_of_individual_blobs_not_in_a_global_fragment,
                            self.number_of_not_accumulable_individual_blobs,
                            self.number_of_not_accumulated_individual_blobs,
                            self.number_of_globally_accumulated_individual_blobs,
                            self.number_of_partially_accumulated_individual_blobs]) / self.number_of_blobs * 100
        labels_with_percentage = [' %.2f' %size + r'$\%$ - ' + label for label, size in zip(labels, sizes)]
        explode = (0, 0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        patches_pie = ax.pie(sizes, colors = colors, explode=explode, autopct='%1.1f%%',
                    shadow=False, startangle=90, pctdistance=1.1)
        # patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for hatch, patch in zip(hatches, patches_pie[0]):
            patch.set_hatch(hatch)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.legend(patches_pie, labels=labels_with_percentage, loc=3, title = 'Blob class percentage')

        fig.savefig(os.path.join(self.video.preprocessing_folder,'fragments_summary_1.pdf'), transparent=True)


        import matplotlib.patches as patches
        def get_class(fragment):
            if fragment.is_a_crossing:
                return 0 #crossing
            elif fragment.is_an_individual and (fragment.accumulated_globally or fragment.accumulated_partially):
                return 1 #assigned during accumulation
            elif fragment.identity != 0:
                return 2 #assigned after accumulation
            else:
                return 3 #not assigned

        labels = ['crossings', 'assigned during accumulation', 'assigned after accumulation', 'not assigned']
        colors = ['k', 'g', 'y', 'r']
        fig, ax = plt.subplots(1,1)
        sns.set_style("ticks")
        for fragment in self.fragments:
            if fragment.is_an_individual:
                blob_index = fragment.blob_hierarchy_in_starting_frame
                type = get_class(fragment)
                (start, end) = fragment.start_end
                ax.add_patch(
                    patches.Rectangle(
                        (start, blob_index - 0.5),   # (x,y)
                        end - start,  # width
                        1.,          # height
                        fill=True,
                        edgecolor=None,
                        facecolor=colors[type],
                        alpha = 1.
                    )
                )

        ax.axis('tight')
        ax.set_xlabel('Frame number')
        ax.set_ylabel('Blob index')
        ax.set_yticks(range(0,self.video.number_of_animals,4))
        ax.set_yticklabels(range(1,self.video.number_of_animals+1,4))
        ax.set_xlim([0., self.video.number_of_frames])
        ax.set_ylim([-.5, .5 + self.video.number_of_animals - 1])
        fig.savefig(os.path.join(self.video.preprocessing_folder,'fragments_summary_2.pdf'), transparent=True)
        plt.show()

def create_list_of_fragments(blobs_in_video, number_of_animals):
    attributes_to_set = ['_image_for_identification', 'bounding_box_image', 'bounding_box_in_frame_coordinates'
                                        '_area', '_next', '_previous',]
    fragments = []
    used_fragment_identifiers = set()

    for blobs_in_frame in tqdm(blobs_in_video, desc = 'creating list of fragments'):
        for blob in blobs_in_frame:
            current_fragment_identifier = blob.fragment_identifier
            if current_fragment_identifier not in used_fragment_identifiers:
                images = [blob.image_for_identification]
                centroids = [blob.centroid]
                areas = [blob.area]
                pixels = [blob.pixels]
                start = blob.frame_number
                current = blob

                while len(current.next) > 0 and current.next[0].fragment_identifier == current_fragment_identifier:
                    current = current.next[0]
                    images, centroids, areas, pixels = append_values_to_lists([current.image_for_identification,
                                                                current.centroid,
                                                                current.area,
                                                                current.pixels],
                                                                [images,
                                                                centroids,
                                                                areas,
                                                                pixels])

                end = current.frame_number
                fragment = Fragment(current_fragment_identifier,
                                    (start, end + 1), # it is not inclusive to follow Python convention
                                    blob.blob_index,
                                    images,
                                    centroids,
                                    areas,
                                    pixels,
                                    blob.is_an_individual,
                                    blob.is_a_crossing,
                                    blob.is_a_jump,
                                    blob.is_a_jumping_fragment,
                                    blob.is_a_ghost_crossing,
                                    number_of_animals,
                                    user_generated_identity = blob.user_generated_identity)
                if fragment.is_a_ghost_crossing:
                    fragment.next_blobs_fragment_identifier = [next_blob.fragment_identifier for next_blob in blob.next if len(blob.next) > 0]
                    fragment.previous_blobs_fragment_identifier = [previous_blob.fragment_identifier for previous_blob in blob.previous if len(blob.previous) > 0]
                used_fragment_identifiers.add(current_fragment_identifier)
                fragments.append(fragment)

            set_attributes_of_object_to_value(blob, attributes_to_set, value = None)
    print("getting coexisting individual fragments indices")
    [fragment.get_coexisting_individual_fragments_indices(fragments) for fragment in fragments]
    print("coexisting fragments done")
    return fragments
