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

import numpy as np
import h5py
import tensorflow as tf

from confapp import conf

from idtrackerai.list_of_blobs import ListOfBlobs
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

# FIXME: This function returns either a TrainDeepCrossing object or a
# ListOfBlobs. Not good practice.


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
    logger.info("Discriminating blobs representing individuals from blobs associated to crossings")

    image_shape = video.identification_image_size[0]
    with h5py.File(video.identification_images_file_path, 'w') as f:
        f.create_dataset("identification_images", ((0, image_shape, image_shape)),
                         chunks=(1, image_shape, image_shape),
                         maxshape=(video.number_of_animals * video.number_of_frames * 5,
                                   image_shape, image_shape))
        f.attrs['number_of_animals'] = video.number_of_animals
        f.attrs['video_path'] = video.video_path

    list_of_blobs.apply_model_area_to_video(video, model_area, video.identification_image_size[0], video.number_of_animals)

    if use_network:
        tf.reset_default_graph()
        video.create_crossings_detector_folder()
        logger.info("Get individual and crossing images labelled data")
        training_set = CrossingDataset(list_of_blobs.blobs_in_video,
                                       video,
                                       scope='training')
        if not training_set.there_are_crossings:
            logger.debug("There are not enough crossings to train the crossing detector")
            video._there_are_crossings = False
            return list_of_blobs
        else:
            training_set.get_data(sampling_ratio_start=0,
                                  sampling_ratio_end=.9)
            validation_set = CrossingDataset(list_of_blobs.blobs_in_video,
                                             video,
                                             scope='validation',
                                             crossings=training_set.crossing_blobs,
                                             individual_blobs=training_set.individual_blobs,
                                            image_size=training_set.image_size)
            validation_set.get_data(sampling_ratio_start=.9, sampling_ratio_end=1.)
            logger.info("Start crossing detector training")
            logger.info("Crossing detector training finished")
            crossing_image_size = training_set.image_size
            crossing_image_shape = training_set.images.shape[1:]
            logger.info("crossing image shape %s" %str(crossing_image_shape))
            crossings_detector_network_params = \
                NetworkParams_crossings(number_of_classes=2,
                                        learning_rate=conf.LEARNING_RATE_DCD,
                                        architecture=cnn_model_crossing_detector,
                                        keep_prob=conf.KEEP_PROB_IDCNN_PRETRAINING,
                                        save_folder=video._crossings_detector_folder,
                                        image_size=crossing_image_shape)
            net = ConvNetwork_crossings(crossings_detector_network_params)
            trainer = TrainDeepCrossing(net, training_set, validation_set,
                                        num_epochs=95,
                                        plot_flag=plot_flag,
                                        return_store_objects=return_store_objects)
        if not trainer.model_diverged:
            logger.debug("crossing image size %s" %str(crossing_image_size))
            video.crossing_image_shape = crossing_image_shape
            video.crossing_image_size = crossing_image_size
            video.save()
            # ind_blob_images = np.asarray(validation_set.generate_individual_blobs_images())
            # crossing_images = validation_set.generate_crossing_images()
            validation_set = None
            training_set = None
            logger.debug("Freeing memory. Validation and training crossings sets deleted")
            test_set = CrossingDataset(list_of_blobs.blobs_in_video, video,
                                       scope='test',
                                       image_size=video.crossing_image_size)
            logger.debug("Classify individuals and crossings")
            crossings_predictor = GetPredictionCrossigns(net)
            predictions = crossings_predictor.get_all_predictions(test_set)
            for blob, prediction in zip(test_set.test, predictions):
                if prediction == 1:
                    blob._is_a_crossing = True
                    blob._is_an_individual = False
                else:
                    blob._is_a_crossing = False
                    blob._is_an_individual = True
                    blob.set_image_for_identification(video)
            # [(setattr(blob, '_is_a_crossing', True), setattr(blob, '_is_an_individual', False)) if prediction == 1
            #     else (setattr(blob, '_is_a_crossing', False), setattr(blob, '_is_an_individual', True), blob.set_image_for_identification(video))
            #     for blob, prediction in zip(test_set.test, predictions)]
            logger.debug("Freeing memory. Test crossings set deleted")
            test_set = None
            if return_store_objects:
                return trainer
    if video.number_of_animals == 1:
        video._there_are_crossings = False
        return list_of_blobs
