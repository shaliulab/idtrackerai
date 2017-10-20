from __future__ import absolute_import, division, print_function
import logging

from list_of_blobs import ListOfBlobs
from get_crossings_data_set import CrossingDataset
from network_params_crossings import NetworkParams_crossings
from cnn_architectures import cnn_model_crossing_detector
from crossings_detector_model import ConvNetwork_crossings
from train_crossings_detector import TrainDeepCrossing
from get_predictions_crossings import GetPredictionCrossigns

logger = logging.getLogger("__main__.crossing_detector")

def detect_crossings(list_of_blobs, video, model_area):
    logger.info("Discriminating blobs representing individuals from blobs associated to crossings")
    list_of_blobs.apply_model_area_to_video(model_area, video.portrait_size[0])
    video.create_crossings_detector_folder()
    logger.info("Get individual and crossing images labelled data")
    training_set = CrossingDataset(list_of_blobs.blobs_in_video, video, scope = 'training')
    training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .9)
    validation_set = CrossingDataset(list_of_blobs.blobs_in_video, video, scope = 'validation',
                                                    crossings = training_set.crossings,
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
    TrainDeepCrossing(net, training_set, validation_set, num_epochs = 95, plot_flag = True)
    logger.debug("crossing image size %s" %str(crossing_image_size))
    video.crossing_image_shape = crossing_image_shape
    video.crossing_image_size = crossing_image_size
    video.save()
    logger.debug("Freeing memory. Validation and training crossings sets deleted")
    validation_set = None
    training_set = None
    test_set = CrossingDataset(list_of_blobs.blobs_in_video, video,
                                scope = 'test',
                                image_size = video.crossing_image_size)
    logger.debug("Classify individuals and crossings")
    crossings_predictor = GetPredictionCrossigns(net)
    predictions = crossings_predictor.get_all_predictions(test_set)
    # set blobs as crossings by deleting the portrait
    [(setattr(blob,'_is_a_crossing', True), setattr(blob, '_is_an_individual', False)) if prediction == 1
        else (setattr(blob,'_is_a_crossing', False), setattr(blob, '_is_an_individual', True))
        for blob, prediction in zip(test_set.test, predictions)]
    logger.debug("Freeing memory. Test crossings set deleted")
    test_set = None
