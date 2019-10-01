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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import sys

from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf

from confapp import conf

from idtrackerai.network.cnn_architectures import cnn_model_crossing_detector
from idtrackerai.network.crossings_detector_model.get_crossings_data_set import CrossingDataset
from idtrackerai.network.crossings_detector_model.network_params_crossings import NetworkParams_crossings
from idtrackerai.network.crossings_detector_model.crossings_detector_model import ConvNetwork_crossings
from idtrackerai.network.crossings_detector_model.train_crossings_detector import TrainDeepCrossing
from idtrackerai.network.crossings_detector_model.get_predictions_crossings import GetPredictionCrossigns

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.crossing_detector")

# FIXME: This function returns either a TrainDeepCrossing object or a ListOfBlobs. Not good practice.

def get_train_validation_and_toassign_blobs(list_of_blobs, ratio_validation=.1):

    training_blobs = {'individuals': [], 'crossings': []}
    validation_blobs = {}
    toassign_blobs = []
    for blobs_in_frame in list_of_blobs.blobs_in_video:
        for blob in blobs_in_frame:
            if blob.is_a_sure_individual() or blob.in_a_global_fragment_core(blobs_in_frame):
                training_blobs['individuals'].append(blob)
            elif blob.is_a_sure_crossing():
                training_blobs['crossings'].append(blob)
            elif (blob.is_an_individual and not blob.in_a_global_fragment_core(blobs_in_frame) and not blob.is_a_sure_individual())\
                or (blob.is_a_crossing and not blob.is_a_sure_crossing()):
                toassign_blobs.append(blob)

    n_blobs_crossings = len(training_blobs['crossings'])
    n_blobs_individuals = len(training_blobs['individuals'])
    logger.debug("number of individual blobs (before cut): {}".format(n_blobs_individuals))
    logger.debug("number of crossing blobs: {}".format(n_blobs_crossings))

    # Shuffle and make crossings and individuals even
    np.random.shuffle(training_blobs['individuals'])
    np.random.shuffle(training_blobs['crossings'])
    if n_blobs_individuals > n_blobs_crossings:
        training_blobs['individuals'] = training_blobs['individuals'][:n_blobs_crossings]
    n_blobs_validation = int(n_blobs_crossings * ratio_validation)

    # split training and validation
    validation_blobs['individuals'] = training_blobs['individuals'][:n_blobs_validation]
    validation_blobs['crossings'] = training_blobs['crossings'][:n_blobs_validation]
    training_blobs['individuals'] = training_blobs['individuals'][n_blobs_validation:]
    training_blobs['crossings'] = training_blobs['crossings'][n_blobs_validation:]

    logger.info("{} individual blobs and {} crossing blobs for training".format(n_blobs_crossings, n_blobs_crossings))
    logger.info("{} individual blobs and {} crossing blobs for validation".format(n_blobs_validation, n_blobs_validation))
    logger.info("{} blobs to test".format(len(toassign_blobs)))

    return training_blobs, validation_blobs, toassign_blobs


def initialize_identification_images_file(video):
    image_shape = video.identification_image_size[0]
    with h5py.File(video.identification_images_file_path, 'w') as f:
        f.create_dataset("identification_images", ((0, image_shape, image_shape)),
                         chunks=(1, image_shape, image_shape),
                         maxshape=(video.number_of_animals * video.number_of_frames * 5,
                                   image_shape, image_shape))
        f.attrs['number_of_animals'] = video.number_of_animals
        f.attrs['video_path'] = video.video_path

def detect_crossings(list_of_blobs,
                     video,
                     model_area,
                     use_network=True,
                     return_store_objects=False,
                     plot_flag=False):
    """Classify all blobs in the video as bing crossings or individuals.

    Parameters
    ----------
    list_of_blobs : <ListOfBlobs object>
        Collection of the Blob objects extracted from the video
    video :  <Video object>
        Object containing all the parameters of the video.
    model_area : function
        Model of the area of a single individual
    use_network : bool
        If True the Deep Crossing Detector is used to distinguish between
        individuals and crossings images. Otherwise only the model area is applied
    return_store_objects : bool
        If True the instantiations of the class :class:`.Store_Accuracy_and_Loss`
        are returned by the function
    plot_flag : bool
        If True a figure representing the values of the loss function, accuracy
        and accuracy per class for both the training and validation set.

    Returns
    -------

    trainer or list_of_blobs : TrainDeepCrossing or ListOfBlobs()
    """

    if video.number_of_animals > 1:
        logger.info("Discriminating blobs representing individuals from blobs associated to crossings")
        list_of_blobs.apply_model_area_to_video(video, model_area, video.identification_image_size[0],
                                                video.number_of_animals)
        initialize_identification_images_file(video)
        logger.info("Computing identification images")
        list_of_blobs.set_images_for_identification(video)

        if use_network:
            tf.reset_default_graph()
            video.create_crossings_detector_folder()
            logger.info("Get list of blobs for training, validation and test")
            train_blobs, val_blobs, toassign_blobs = get_train_validation_and_toassign_blobs(list_of_blobs)

            if len(train_blobs['crossings']) > conf.MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR:
                video._there_are_crossings = True
                logger.info("There are enough crossings to train the crossing detector")

                logger.info("Creating training CrossingDataset")
                training_set = CrossingDataset(train_blobs,
                                               video,
                                               scope='training')
                training_set.get_data()

                logger.info("Creating validation CrossingDataset")
                validation_set = CrossingDataset(val_blobs,
                                                 video,
                                                 scope='validation')
                validation_set.get_data()

                logger.info("Start crossing detector training")

                crossings_detector_network_params = \
                    NetworkParams_crossings(number_of_classes=2,
                                            learning_rate=conf.LEARNING_RATE_DCD,
                                            architecture=cnn_model_crossing_detector,
                                            keep_prob=conf.KEEP_PROB_IDCNN_PRETRAINING,
                                            save_folder=video.crossings_detector_folder,
                                            image_size=video.identification_image_size)

                net = ConvNetwork_crossings(crossings_detector_network_params)
                trainer = TrainDeepCrossing(net, training_set, validation_set,
                                            num_epochs=95,
                                            plot_flag=plot_flag,
                                            return_store_objects=return_store_objects)

                logger.info("Crossing detector training finished")

                if not trainer.model_diverged:
                    del validation_set
                    del training_set
                    logger.info("Freeing memory. Validation and training crossings sets deleted")
                    test_set = CrossingDataset(toassign_blobs,
                                               video,
                                               scope='test')
                    test_set.get_data()
                    logger.info("Classify individuals and crossings")
                    crossings_predictor = GetPredictionCrossigns(net)
                    predictions = crossings_predictor.get_all_predictions(test_set)
                    for blob, prediction in zip(toassign_blobs, predictions):
                        if prediction == 1:
                            blob._is_a_crossing = True
                            blob._is_an_individual = False
                        else:
                            blob._is_a_crossing = False
                            blob._is_an_individual = True
                    logger.debug("Freeing memory. Test crossings set deleted")
                    del test_set
                    if return_store_objects:
                        return trainer
            else:
                logger.debug("There are not enough crossings to train the crossing detector")
                video._there_are_crossings = False
                return list_of_blobs

    elif video.number_of_animals == 1:
        video._there_are_crossings = False
        return list_of_blobs
