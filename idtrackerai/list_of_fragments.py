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
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (R-F.,F. and B.,M. contributed equally to this work.)
 

from __future__ import absolute_import, division, print_function
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from idtrackerai.fragment import Fragment
from idtrackerai.utils.py_utils import  set_attributes_of_object_to_value, append_values_to_lists
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.list_of_fragments")

class ListOfFragments(object):
    """ Collects all the instances of the class :class:`~fragment.Fragment`
    generated from the blobs extracted from the video during segmentation
    (see :mod:`~segmentation`) after having assigned to each Blob instance
    a fragment identifier by using the method
    :meth:`~list_of_blobs.compute_fragment_identifier_and_blob_index`

    Attributes
    ----------

    fragments : list
        list of instances of the class :class:`~fragment.Fragment`
    number_of_fragments : int
        number of fragments computed by the method
        :meth:`~list_of_blobs.compute_fragment_identifier_and_blob_index`
    """
    def __init__(self, fragments):
        self.fragments = fragments
        self.number_of_fragments = len(self.fragments)

    def get_fragment_identifier_to_index_list(self):
        """Creates a mapping between the attribute :attr:`fragments` and
        their identifiers build from the :class:`~list_of_blobs.ListOfBlobs`

        Returns
        -------
        list
            Mapping from the collection of fragments to the list of fragment
            identifiers

        """
        fragments_identifiers = [fragment.identifier for fragment in self.fragments]
        fragment_identifier_to_index = np.arange(len(fragments_identifiers))
        fragments_identifiers_argsort = np.argsort(fragments_identifiers)
        return fragment_identifier_to_index[fragments_identifiers_argsort]

    def reset(self, roll_back_to = None):
        """Resets all the fragment by using the method
        :meth:`~fragment.Fragment.roll_back_to`
        """
        logger.warning("Reseting list_of_fragments")
        [fragment.reset(roll_back_to) for fragment in self.fragments]
        logger.warning("Done")

    def get_images_from_fragments_to_assign(self):
        """Take all the fragments that have not been used to train the idCNN
        and that are associated with an individual, and concatenates their
        images in order to feed them to the idCNN and get predictions

        Returns
        -------
        ndarray
            [number_of_images, height, width, number_of_channels]
        """
        return np.concatenate([np.asarray(fragment.images) for fragment in self.fragments
                                if not fragment.used_for_training and fragment.is_an_individual], axis = 0)

    def compute_number_of_unique_images_used_for_pretraining(self):
        """Returns the number of images used for pretraining
        (without repetitions)

        Returns
        -------
        int
            Number of images used in pretraining
        """
        return sum([fragment.number_of_images for fragment in self.fragments if fragment.used_for_pretraining])

    def compute_number_of_unique_images_used_for_training(self):
        """Returns the number of images used for training
        (without repetitions)

        Returns
        -------
        int
            Number of images used in training
        """
        return sum([fragment.number_of_images for fragment in self.fragments if fragment.used_for_training])

    def compute_total_number_of_images_in_global_fragments(self):
        """Sets the number of images available in global fragments (without repetitions)
        """
        self.number_of_images_in_global_fragments = sum([fragment.number_of_images for fragment in self.fragments if fragment.is_in_a_global_fragment])

    def compute_ratio_of_images_used_for_pretraining(self):
        """Returns the ratio of images used for pretraining over the number of
        available images

        Returns
        -------
        float
            Ratio of images used for pretraining
        """
        return self.compute_number_of_unique_images_used_for_pretraining() / self.number_of_images_in_global_fragments

    def compute_ratio_of_images_used_for_training(self):
        """Returns the ratio of images used for training over the number of
        available images

        Returns
        -------
        float
            Ratio of images used for training
        """
        return self.compute_number_of_unique_images_used_for_training() / self.number_of_images_in_global_fragments

    def compute_P2_vectors(self):
        """Computes the P2_vector associated to every individual fragment. See
        :meth:`~fragment.Fragment.compute_P2_vector`
        """
        [fragment.compute_P2_vector() for fragment in self.fragments if fragment.is_an_individual]

    def get_number_of_unidentified_individual_fragments(self):
        """Returns the number of individual fragments that have not been
        identified during the fingerprint protocols cascade

        Returns
        -------
        int
            number of non-identified individual fragments
        """
        return len([fragment for fragment in self.fragments if fragment.is_an_individual and not fragment.used_for_training])

    def get_next_fragment_to_identify(self):
        """Returns the next fragment to be identified after the fingerprint
        protocols cascade by sorting according to the certainty computed with
        P2. See :attr:`~fragment.Fragment.certainty_P2`

        Returns
        -------
        <Fragment object>
            an instance of the class :class:`~fragment.Fragment`

        """
        fragments = [fragment for fragment in self.fragments if fragment.assigned_identity is None and fragment.is_an_individual]
        fragments.sort(key=lambda x: x.certainty_P2, reverse=True)
        return fragments[0]

    def get_data_plot(self):
        """Gathers the data to plot the individual fragments' statistics of the
        video.

        Returns
        -------
        ndarray
            array of shape [number_of_individual_fragments, 1]. Number of
            images in individual fragments
        ndarray
            array of shape [number_of_individual_fragments, 1]. Distance
            travelled in every individual fragment
        int
            Number of images in crossing fragments
        """
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

    def update_from_list_of_blobs(self, blobs_in_video, fragment_identifier_to_index):
        """Updates an instance of ListOfFragments by considering an instance of
        ListOfBlobs (see :class:`~list_of_blobs.ListOfBlobs`)

        Parameters
        ----------
        blobs_in_video : list
            list of the blob objects (see class :class:`~blob.Blob`) generated
            from the blobs segmented in the video
        fragment_identifier_to_index : list
            Mapping from the collection of fragments to the list of fragment
            identifiers
        """
        [setattr(self.fragments[fragment_identifier_to_index[blob.fragment_identifier]], '_user_generated_identity', blob.user_generated_identity)
            for blobs_in_frame in blobs_in_video for blob in blobs_in_frame if blob.user_generated_identity is not None ]

    def get_ordered_list_of_fragments(self, scope = None, first_frame_first_global_fragment = None):
        """Sorts the fragments starting from the frame number
        `first_frame_first_global_fragment`. According to `scope` the sorting
        is done either "to the future" of "to the past" with respect to
        `first_frame_first_global_fragment`

        Parameters
        ----------
        scope : str
            either "to_the_past" or "to_the_future"
        first_frame_first_global_fragment : int
            frame number corresponding to the first frame in which all the
            individual fragments coexist in the first global fragment using
            in an iteration of the fingerprint protocol cascade

        Returns
        -------
        list
            list of sorted fragments

        """
        if scope == 'to_the_past':
            fragments_subset = [fragment for fragment in self.fragments if fragment.start_end[1] <= first_frame_first_global_fragment]
            fragments_subset.sort(key=lambda x: x.start_end[1], reverse=True)
        elif scope == 'to_the_future':
            fragments_subset = [fragment for fragment in self.fragments if fragment.start_end[0] >= first_frame_first_global_fragment]
            fragments_subset.sort(key=lambda x: x.start_end[0], reverse=False)
        return fragments_subset

    def save(self, fragments_path):
        """saves an instance of ListOfFragments in the path specified by
        `fragments_path`
        """
        logger.info("saving list of fragments at %s" %fragments_path)
        [setattr(fragment, 'coexisting_individual_fragments', None) for fragment in self.fragments]
        np.save(fragments_path,self)
        [fragment.get_coexisting_individual_fragments_indices(self.fragments) for fragment in self.fragments]

    @classmethod
    def load(cls, path_to_load):
        """Loads a previously saved (see :meth:`load`) from the path
        `path_to_load`
        """
        logger.info("loading list of fragments from %s" %path_to_load)
        list_of_fragments = np.load(path_to_load).item()
        [fragment.get_coexisting_individual_fragments_indices(list_of_fragments.fragments) for fragment in list_of_fragments.fragments]
        return list_of_fragments

    def create_light_list(self, attributes = None):
        """Creates a light version of an instance of ListOfFragments by storing
        only the attributes listed in `attributes` in a list of dictionaries

        Parameters
        ----------
        attributes : list
            list of attributes to be stored

        Returns
        -------
        list
            list of dictionaries organised per fragment with keys the
            attributes listed in `attributes`
        """
        if attributes == None:
            attributes_to_discard = ['images',
                            'pixels',
                            'coexisting_individual_fragments']
        return [{attribute: getattr(fragment, attribute) for attribute in fragment.__dict__.keys() if attribute not in attributes_to_discard}
                    for fragment in self.fragments]

    def save_light_list(self, accumulation_folder):
        """Saves a list of dictionaries created with the method
        :meth:`create_light_list` in the folder `accumulation_folder`
        """
        np.save(os.path.join(accumulation_folder, 'light_list_of_fragments.npy'), self.create_light_list())

    def load_light_list(self, accumulation_folder):
        """Loads a list of dictionaries created with the method
        :meth:`create_light_list` and saved with :meth:`save_light_list` from
        the folder `accumulation_folder`
        """
        list_of_dictionaries = np.load(os.path.join(accumulation_folder, 'light_list_of_fragments.npy'))
        self.update_fragments_dictionary(list_of_dictionaries)

    def update_fragments_dictionary(self, list_of_dictionaries):
        """Update fragment objects (see :class:`~fragment.Fragment`) by
        considering a list of dictionaries
        """
        assert len(list_of_dictionaries) == len(self.fragments)
        [fragment.__dict__.update(dictionary) for fragment, dictionary in zip(self.fragments, list_of_dictionaries)]

    def get_new_images_and_labels_for_training(self):
        """Extract images and creates labels from every individual fragment
        that has not been used to train the network during the fingerprint
        protocols cascade

        Returns
        -------
        list
            images
        list
            labels
        """
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
        """Gets the unique identifiers associated to individual fragments that
        can be accumulated

        Parameters
        ----------
        list_of_global_fragments : <ListOfGlobalFragments object>
            Object collecting the global fragment objects (instances of the
            class :class:`~global_fragment.GlobalFragment`) detected in the
            entire video

        """
        self.accumulable_individual_fragments = set([identifier for global_fragment in list_of_global_fragments.global_fragments
                                                                        for identifier in global_fragment.individual_fragments_identifiers])

    def get_not_accumulable_individual_fragments_identifiers(self, list_of_global_fragments):
        """Gets the unique identifiers associated to individual fragments that
        cannot be accumulated

        Parameters
        ----------
        list_of_global_fragments : <ListOfGlobalFragments object>
            Object collecting the global fragment objects (instances of the
            class :class:`~global_fragment.GlobalFragment`) detected in the
            entire video

        """
        self.not_accumulable_individual_fragments = set([identifier for global_fragment in list_of_global_fragments.non_accumulable_global_fragments
                                                        for identifier in global_fragment.individual_fragments_identifiers]) - self.accumulable_individual_fragments

    def set_fragments_as_accumulable_or_not_accumulable(self):
        """Set the attribute :attr:`~fragment.accumulable`
        """
        for fragment in self.fragments:
            if fragment.identifier in self.accumulable_individual_fragments:
                setattr(fragment, '_accumulable', True)
            elif fragment.identifier in self.not_accumulable_individual_fragments:
                setattr(fragment, '_accumulable', False)
            else:
                setattr(fragment, '_accumulable', None)

    def get_stats(self, list_of_global_fragments):
        """Collects the following statistics from both fragments and global
        fragments:

        *number_of_fragments
        *number_of_crossing_fragments
        *number_of_individual_fragments
        *number_of_individual_fragments_not_in_a_global_fragment
        *number_of_accumulable_individual_fragments
        *number_of_not_accumulable_individual_fragments
        *number_of_accumualted_individual_fragments
        *number_of_globally_accumulated_individual_fragments
        *number_of_partially_accumulated_individual_fragments
        *number_of_blobs
        *number_of_crossing_blobs
        *number_of_individual_blobs
        *number_of_individual_blobs_not_in_a_global_fragment
        *number_of_accumulable_individual_blobs
        *number_of_not_accumulable_individual_blobs
        *number_of_accumualted_individual_blobs
        *number_of_globally_accumulated_individual_blobs
        *number_of_partially_accumulated_individual_blobs

        Parameters
        ----------
        list_of_global_fragments : <ListOfGlobalFragments object>
            Object collecting the global fragment objects (instances of the
            class :class:`~global_fragment.GlobalFragment`) detected in the
            entire video

        Returns
        -------
        dict
            dictionary of statistics

        """
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
        attributes_to_return = ['number_of_fragments',
            'number_of_crossing_fragments', 'number_of_individual_fragments',
            'number_of_individual_fragments_not_in_a_global_fragment',
            'number_of_accumulable_individual_fragments',
            'number_of_not_accumulable_individual_fragments',
            'number_of_accumualted_individual_fragments',
            'number_of_globally_accumulated_individual_fragments',
            'number_of_partially_accumulated_individual_fragments',
            'number_of_blobs', 'number_of_crossing_blobs',
            'number_of_individual_blobs',
            'number_of_individual_blobs_not_in_a_global_fragment',
            'number_of_accumulable_individual_blobs',
            'number_of_not_accumulable_individual_blobs',
            'number_of_accumualted_individual_blobs',
            'number_of_globally_accumulated_individual_blobs',
            'number_of_partially_accumulated_individual_blobs']
        return {key: getattr(self, key) for key in self.__dict__ if key in attributes_to_return}

    def plot_stats(self, video):
        """Plots the statistics obtained through :meth:`get_stats`

        Parameters
        ----------
        video : <Video object>
            See :class:`~video.Video`
        """
        from matplotlib import pyplot as plt
        import matplotlib.lines as mlines
        import seaborn as sns
        plt.ion()
        fig, ax = plt.subplots(1,1)
        sns.set_style("ticks")
        colors = ['grey', 'y', 'y', 'r', 'g', 'g']
        hatches = ['', '', '/', '', '', '//']
        labels = ['crossings', 'not in GF', 'not accumulable',
                    'not accumulated', 'globally accumulated',
                    'partially accumulated']
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

        fig.savefig(os.path.join(video.preprocessing_folder,'fragments_summary_1.pdf'), transparent=True)

        import matplotlib.patches as patches
        def get_fragment_identification_type(fragment):
            """Returns the an identifier of the process in which fragment has
            been identified in the algorithm:
            *0: The fragment is a crossing, hence identified only during
            post-processing
            *1: The (individual) fragment has been identified during the
            fingerprint protocols cascade
            *2: The (individual) fragment has been identified after the
            protocols cascade
            *3: The (individual) fragment has not been assigned

            Parameters
            ----------
            fragment : <Fragment object>
                an instance of the class :class:`~fragment.Fragment`

            Returns
            -------
            int
                identifier of the identification state of the fragment

            """
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
                type = get_fragment_identification_type(fragment)
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
        ax.set_yticks(range(0,video.number_of_animals,4))
        ax.set_yticklabels(range(1,video.number_of_animals+1,4))
        ax.set_xlim([0., video.number_of_frames])
        ax.set_ylim([-.5, .5 + video.number_of_animals - 1])
        fig.savefig(os.path.join(video.preprocessing_folder,'fragments_summary_2.pdf'), transparent=True)
        plt.show()

def create_list_of_fragments(blobs_in_video, number_of_animals):
    """Generate a list of instances of :class:`~fragment.Fragment` collecting
    all the fragments in the video.

    Parameters
    ----------
    blobs_in_video : list
        list of the blob objects (see class :class:`~blob.Blob`) generated
        from the blobs segmented in the video
    number_of_animals : int
        Number of animals to track

    Returns
    -------
    list
        list of instances of :class:`~fragment.Fragment`

    """
    attributes_to_set = ['_image_for_identification',
                        'bounding_box_image', '_next', '_previous']
    fragments = []
    used_fragment_identifiers = set()

    for blobs_in_frame in tqdm(blobs_in_video, desc = 'creating list of fragments'):
        for blob in blobs_in_frame:
            current_fragment_identifier = blob.fragment_identifier
            if current_fragment_identifier not in used_fragment_identifiers:
                images = [blob.image_for_identification] if blob.is_an_individual else [blob.bounding_box_image]
                bounding_boxes = [blob.bounding_box_in_frame_coordinates] if blob.is_a_crossing else []
                centroids = [blob.centroid]
                areas = [blob.area]
                pixels = [blob.pixels]
                start = blob.frame_number
                current = blob

                while len(current.next) > 0 and current.next[0].fragment_identifier == current_fragment_identifier:
                    current = current.next[0]
                    bounding_box_in_frame_coordinates = [current.bounding_box_in_frame_coordinates] if current.is_a_crossing else []
                    images, bounding_boxes, centroids, areas, pixels = append_values_to_lists([current.image_for_identification,
                                                                bounding_box_in_frame_coordinates,
                                                                current.centroid,
                                                                current.area,
                                                                current.pixels],
                                                                [images,
                                                                bounding_boxes,
                                                                centroids,
                                                                areas,
                                                                pixels])

                end = current.frame_number
                fragment = Fragment(current_fragment_identifier,
                                    (start, end + 1), # it is not inclusive to follow Python convention
                                    blob.blob_index,
                                    images,
                                    bounding_boxes,
                                    centroids,
                                    areas,
                                    pixels,
                                    blob.is_an_individual,
                                    blob.is_a_crossing,
                                    number_of_animals,
                                    user_generated_identity = blob.user_generated_identity)
                used_fragment_identifiers.add(current_fragment_identifier)
                fragments.append(fragment)

            set_attributes_of_object_to_value(blob, attributes_to_set, value = None)
    logger.info("getting coexisting individual fragments indices")
    [fragment.get_coexisting_individual_fragments_indices(fragments) for fragment in fragments]
    return fragments