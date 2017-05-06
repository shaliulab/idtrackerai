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
        self._conv_vectors = []

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.data_set.images[start:end]

    def get_predictions_softmax(self, batch_operation):
        self._index_in_epoch = 0
        while self._index_in_epoch < self.data_set._num_images:
            softmax_probs_batch, predictions_batch = batch_operation(self.next_batch(BATCH_SIZE))
            self._softmax_probs.append(softmax_probs_batch)
            self._predictions.append(predictions_batch)
        self._softmax_probs = np.concatenate(self._softmax_probs, axis = 0)
        self._predictions = np.concatenate(self._predictions, axis = 0)

    def get_predictions_conv_embedding(self, batch_operation, number_of_animals):
        self._index_in_epoch = 0
        while self._index_in_epoch < self.data_set._num_images:
            conv_vectors = batch_operation(self.next_batch(BATCH_SIZE))
            print(conv_vectors.shape)
            self._conv_vectors.append(conv_vectors)
        self._conv_vectors = np.concatenate(self._conv_vectors, axis = 0)
        _, self._predictions = kMeansCluster(self._conv_vectors, number_of_animals, 100)


def kMeansCluster(vector_values, num_clusters, max_num_steps, stop_coeficient = 0.0):
  vectors = tf.constant(vector_values)
  centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors),
                                   [0,0],[num_clusters,-1]))
  old_centroids = tf.Variable(tf.zeros([num_clusters,vector_values.shape[1]]))
  centroid_distance = tf.Variable(tf.zeros([num_clusters,vector_values.shape[1]]))

  expanded_vectors = tf.expand_dims(vectors, 0)
  expanded_centroids = tf.expand_dims(centroids, 1)

  # print 'vectors shape: '+ str(expanded_vectors.get_shape())
  # print 'centroids shape: ' + str(expanded_centroids.get_shape())

  distances = tf.reduce_sum(
    tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
  assignments = tf.argmin(distances, 0)

  means = tf.stack([
    tf.reduce_mean(
        tf.boolean_mask(
            vectors, tf.equal(assignments, c)
        ), 0)
    for c in xrange(num_clusters)])

  save_old_centroids = tf.assign(old_centroids, centroids)

  update_centroids = tf.assign(centroids, means)
  init_op = tf.initialize_all_variables()

  performance = tf.assign(centroid_distance, tf.subtract(centroids, old_centroids))
  check_stop = tf.reduce_sum(tf.abs(performance))

  with tf.Session() as sess:
    sess.run(init_op)
    for step in xrange(max_num_steps):
    #   print "Running step " + str(step)
      sess.run(save_old_centroids)
      _, centroid_values, assignment_values = sess.run([update_centroids,
                                                        centroids,
                                                        assignments])
      sess.run(check_stop)
      current_stop_coeficient = check_stop.eval()
    #   print "coeficient:", current_stop_coeficient
      if current_stop_coeficient <= stop_coeficient:
        break

    return centroid_values, assignment_values
