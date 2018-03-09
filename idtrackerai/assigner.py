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
# (2018). idtracker.ai: Tracking unmarked individuals in large collectives. (R-F.,F. and B.,M. contributed equally to this work.)
 

from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from idtrackerai.network.identification_model.network_params import NetworkParams
from idtrackerai.network.identification_model.get_data import DataSet
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.network.identification_model.get_predictions import GetPrediction
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.assigner")

"""
Identification of individual fragments given the predictions generate by the idCNN
"""

def assign(net, video, images, print_flag):
    """Gathers the predictions relative to the images contained in `images`.
    Such predictions are returned as attributes of `assigner`.

    Parameters
    ----------
    net : <ConvNetwork object>
        Convolutional neural network object created according to net.params
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving
        and loading
    images : ndarray
        array of images
    print_flag : bool
        If True additional information gathered while getting the predictions
        are displayed in the terminal

    Returns
    -------
    <GetPrediction object>
        The assigner object has as main attributes the list of predictions
        associated to `images` and the the corresponding softmax vectors

    See Also
    --------
    GetPrediction
    """
    logger.info("assigning identities to images...")
    images = np.expand_dims(np.asarray(images), axis = 3)
    logger.info("generating data set. Images shape %s" %str(images.shape))
    data = DataSet(net.params.number_of_animals, images)
    logger.info("getting predictions")
    assigner = GetPrediction(data)
    assigner.get_predictions_softmax(net.predict)
    logger.info("done")
    return assigner

def compute_identification_statistics_for_non_accumulated_fragments(fragments, assigner, number_of_animals = None):
    """Given the predictions associated to the images in each (individual)
    fragment in the list fragments if computes the statistics necessary for the
    identification of fragment.

    Parameters
    ----------
    fragments : list
        List of individual fragment objects
    assigner : <GetPrediction object>
        The assigner object has as main attributes the list of predictions
        associated to `images` and the the corresponding softmax vectors
    number_of_animals : int
        number of animals to be tracked
    """
    counter = 0
    for fragment in fragments:
        if not fragment.used_for_training and fragment.is_an_individual:
            next_counter_value = counter + fragment.number_of_images
            predictions = assigner.predictions[counter : next_counter_value]
            softmax_probs = assigner.softmax_probs[counter : next_counter_value]
            fragment.compute_identification_statistics(predictions, softmax_probs, number_of_animals = number_of_animals)
            counter = next_counter_value

def assign_identity(list_of_fragments):
    """Identifies the individual fragments recursively, based on the value of
    P2

    Parameters
    ----------
    list_of_fragments : <ListOfFragments object>
        collection of the individual fragments and associated methods
    """
    list_of_fragments.compute_P2_vectors()
    number_of_unidentified_individual_fragments = list_of_fragments.get_number_of_unidentified_individual_fragments()

    while number_of_unidentified_individual_fragments != 0:
        fragment = list_of_fragments.get_next_fragment_to_identify()
        fragment.assign_identity()
        number_of_unidentified_individual_fragments -= 1

def assigner(list_of_fragments, video, net):
    """This is the main function of this method: given a list_of_fragments it
    puts in place the routine to identify, if possible, each of the individual
    fragments. The starting point for the identification is given by the
    predictions produced by the ConvNetwork net passed as input. The organisation
    of the images in individual fragments is then used to assign more accurately.

    Parameters
    ----------
    list_of_fragments : <ListOfFragments object>
        collection of the individual fragments and associated methods
    video : <Video object>
        Object collecting all the parameters of the video and paths for saving and loading
    net : <ConvNetwork object>
        Convolutional neural network object created according to net.params

    See Also
    --------
    ListOfFragments.reset(roll_back_to = 'accumulation')
    ListOfFragments.get_images_from_fragments_to_assign
    assign
    compute_identification_statistics_for_non_accumulated_fragments

    """
    logger.info("Assigning identities to non-accumulated individual fragments")
    logger.debug("Resetting list of fragments for assignment")
    list_of_fragments.reset(roll_back_to = 'accumulation')
    logger.info("Getting images")
    images = list_of_fragments.get_images_from_fragments_to_assign()
    logger.debug("Images shape before assignment %s" %str(images.shape))
    logger.info("Getting predictions")
    assigner = assign(net, video, images, print_flag = True)
    logger.debug("Number of generated predictions: %s" %str(len(assigner._predictions)))
    logger.debug("Predictions range: %s" %str(np.unique(assigner._predictions)))
    compute_identification_statistics_for_non_accumulated_fragments(list_of_fragments.fragments, assigner)
    logger.info("Assigning identities")
    assign_identity(list_of_fragments)
