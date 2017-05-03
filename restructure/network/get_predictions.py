from __future__ import absolute_import, division, print_function

import itertools
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 5000 # 32x32 = 1024bytes x BATCH_SIZE ~ 100MB

class GetPrediction(object):
    def __init__(self, data_set,
                print_flag = True):
        # Data set
        self.data_set = data_set
        self._softmax_probs = []
        self._predictions = []

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.data_set.images[start:end]

    def get_predictions(self, name, store_loss_and_accuracy, batch_operation):
        self._index_in_epoch = 0
        while self._index_in_epoch < self.data_set._num_images:
            softmax_probs_batch, predictions_batch = batch_operation(self.next_batch(BATCH_SIZE))
            self._softmax_probs.append(softmax_probs_batch)
            self._predictions.append(predictions_batch)
