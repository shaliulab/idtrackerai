from __future__ import absolute_import, division, print_function

import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import psutil

BATCH_SIZE = 100

class GetPredictionCrossigns(object):
    def __init__(self, net):
        # Data set
        self.net = net
        self._softmax_probs = []
        self._predictions = []
        self._fc_vectors = []
        self.batch_size = BATCH_SIZE


    def next_batch_test(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.number_of_image_predicted
        self.number_of_image_predicted += batch_size
        end = self.number_of_image_predicted
        return self.test_images[start:end]

    def predict(self, test_images):
        self.test_images = test_images
        self.number_of_image_predicted = 0
        predictions = []
        while self.number_of_image_predicted < len(test_images):
            predictions.extend(self.net.prediction(self.next_batch_test(batch_size = BATCH_SIZE)))

        return predictions

    def get_all_predictions(self, test_set):
        # compute maximum number of images given the available RAM
        image_size_bytes = np.prod(test_set.image_size)*4 #XXX:Check it out!!!! np.prod(test_set.image_size**2)*4
        number_of_images_to_be_fitted_in_RAM = len(test_set.test)
        num_images_that_can_fit_in_RAM = 100000#int(psutil.virtual_memory().available*.2/image_size_bytes)
        if number_of_images_to_be_fitted_in_RAM > num_images_that_can_fit_in_RAM:
            print("There is NOT enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
            number_of_predictions_retrieved = 0
            predictions = []
            i = 0
            while number_of_predictions_retrieved < number_of_images_to_be_fitted_in_RAM:
                images = test_set.generate_test_images(interval = (i*num_images_that_can_fit_in_RAM, (i+1)*num_images_that_can_fit_in_RAM))
                predictions.extend(self.predict(images))
                number_of_predictions_retrieved = len(predictions)
                i += 1
        else:
            print("There is enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
            print("getting predictions...")
            test_images = test_set.generate_test_images()
            predictions = self.predict(test_images)
        return predictions

    def get_predictions_from_images(self, images):
        image_size_bytes = np.prod(images.shape[1:])*4
        number_of_images_to_be_fitted_in_RAM = len(images)
        num_images_that_can_fit_in_RAM = int(psutil.virtual_memory().available*.9/image_size_bytes)
        if number_of_images_to_be_fitted_in_RAM > num_images_that_can_fit_in_RAM:
            print("There is NOT enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
            number_of_predictions_retrieved = 0
            predictions = []
            i = 0
            while number_of_predictions_retrieved < number_of_images_to_be_fitted_in_RAM:
                images_batch = images[i*num_images_that_can_fit_in_RAM : (i+1)*num_images_that_can_fit_in_RAM]
                predictions.extend(self.predict(images_batch))
                number_of_predictions_retrieved = len(predictions)
                i += 1
        else:
            print("There is enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
            print("getting predictions...")
            predictions = self.predict(images)
        return predictions
