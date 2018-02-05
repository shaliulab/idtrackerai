from __future__ import absolute_import, division, print_function
import logging
import numpy as np
import cv2

from list_of_blobs import ListOfBlobs
from get_crossings_data_set import CrossingDataset
from network_params_crossings import NetworkParams_crossings
from cnn_architectures import cnn_model_crossing_detector
from crossings_detector_model import ConvNetwork_crossings
from train_crossings_detector import TrainDeepCrossing
from get_predictions_crossings import GetPredictionCrossigns

logger = logging.getLogger("__main__.crossing_detector")

def detect_crossings(list_of_blobs, video, model_area, use_network = True, return_store_objects = False, plot_flag = True):
    logger.info("Discriminating blobs representing individuals from blobs associated to crossings")
    list_of_blobs.apply_model_area_to_video(video, model_area, video.identification_image_size[0], video.number_of_animals)
    if use_network:
        video.create_crossings_detector_folder()
        logger.info("Get individual and crossing images labelled data")
        training_set = CrossingDataset(list_of_blobs.blobs_in_video, video, scope = 'training')
        training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .9)
        validation_set = CrossingDataset(list_of_blobs.blobs_in_video, video, scope = 'validation',
                                                        crossings = training_set.crossing_blobs,
                                                        individual_blobs = training_set.individual_blobs,
                                                        image_size = training_set.image_size)
        validation_set.get_data(sampling_ratio_start = .9, sampling_ratio_end = 1.)
        logger.info("Start crossing detector training")
        logger.info("Crossing detector training finished")
        crossing_image_size = training_set.image_size
        crossing_image_shape = training_set.images.shape[1:]
        logger.info("crossing image shape %s" %str(crossing_image_shape))
        crossings_detector_network_params = NetworkParams_crossings(number_of_classes = 2,
                                                                    learning_rate = 0.001,
                                                                    architecture = cnn_model_crossing_detector,
                                                                    keep_prob = 1.0,
                                                                    save_folder = video._crossings_detector_folder,
                                                                    image_size = crossing_image_shape)
        net = ConvNetwork_crossings(crossings_detector_network_params)
        trainer = TrainDeepCrossing(net, training_set, validation_set,
                                    num_epochs = 95, plot_flag = plot_flag,
                                    return_store_objects = return_store_objects)
        if not trainer.model_diverged:
            logger.debug("crossing image size %s" %str(crossing_image_size))
            video.crossing_image_shape = crossing_image_shape
            video.crossing_image_size = crossing_image_size
            video.save()
            ind_blob_images = np.asarray(validation_set.generate_individual_blobs_images())
            crossing_images = validation_set.generate_crossing_images()
            validation_set = None
            training_set = None
            logger.debug("Freeing memory. Validation and training crossings sets deleted")
            test_set = CrossingDataset(list_of_blobs.blobs_in_video, video,
                                        scope = 'test',
                                        image_size = video.crossing_image_size)
            logger.debug("Classify individuals and crossings")
            crossings_predictor = GetPredictionCrossigns(net)
            predictions = crossings_predictor.get_all_predictions(test_set)
            [(setattr(blob,'_is_a_crossing', True), setattr(blob, '_is_an_individual', False)) if prediction == 1
                else (setattr(blob,'_is_a_crossing', False), setattr(blob, '_is_an_individual', True), blob.set_image_for_identification(video))
                for blob, prediction in zip(test_set.test, predictions)]
            logger.debug("Freeing memory. Test crossings set deleted")
            test_set = None
            if return_store_objects:
                return trainer
